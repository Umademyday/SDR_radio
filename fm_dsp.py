#!/usr/bin/env python3
"""The broadcast-FM signal chain, shared by the offline and live receivers.

`decode_fm.py` (capture file -> WAV) and `play_fm.py` (dongle -> speakers) are
different programs with different constraints, but the radio inside them is the
same radio. Everything that is genuinely the same lives here: the capture
format, the sample rates, the filter designs, the discriminator and the
de-emphasis curve.

What deliberately does *not* live here, because the two really do differ:

* **Filter state.** Offline, a whole capture is filtered in one pass. Live,
  samples arrive in blocks and every filter must carry its state across the
  seams (see `play_fm.PolyphaseDecimator`).
* **Output scaling.** Offline can normalise against the peak of the whole file;
  live has no "whole file" and uses a fixed gain against the 75 kHz broadcast
  deviation limit.

Rates through the chain:

    1.8 MS/s complex --(/6)--> 300 kS/s complex --discriminator-->
    300 kS/s deviation in Hz --> 48 kS/s (offline WAV) or 50 kS/s (live audio)
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

# --- Capture format: what rtl_sdr writes, to a file or to a pipe -------------
CAPTURE_SAMPLE_RATE_HZ = 1_800_000
CU8_ZERO_LEVEL = 127.5  # midpoint of the unsigned 8-bit range 0..255
CU8_FULL_SCALE = 127.5  # divisor that maps 0..255 onto about -1.0..+1.0
BYTES_PER_COMPLEX_SAMPLE = 2  # one byte of I, one byte of Q

# --- Isolating the centred station -------------------------------------------
# Both decimation stages use 6: 1.8 MS/s -> 300 kS/s, and (live) 300 -> 50 kS/s.
BASEBAND_DECIMATION_FACTOR = 6
BASEBAND_SAMPLE_RATE_HZ = CAPTURE_SAMPLE_RATE_HZ // BASEBAND_DECIMATION_FACTOR  # 300 kS/s
STATION_HALF_BANDWIDTH_HZ = 100_000  # a broadcast FM channel occupies about +/-100 kHz
STATION_FILTER_BASE_TAPS = 129  # before the polyphase rounding below

# --- Mono audio ---------------------------------------------------------------
MONO_AUDIO_CUTOFF_HZ = 15_000  # top of the mono (L+R) band
STEREO_PILOT_HZ = 19_000  # must be in the stopband or it whistles in the audio
PILOT_ATTENUATION_DB = 70.0
DEEMPHASIS_TIME_CONSTANT_S = 50e-6  # Europe (the Americas and Korea use 75 us)

# --- PCM ----------------------------------------------------------------------
INT16_MAX = 32_767
INT16_MIN = -32_768


def cu8_bytes_to_iq(raw_bytes: np.ndarray) -> np.ndarray:
    """Convert interleaved unsigned 8-bit bytes into centred complex I/Q samples.

    The bytes are widened to float *before* the midpoint is subtracted: doing it
    the other way round would wrap around in unsigned 8-bit arithmetic and turn
    every negative sample into a large positive one.
    """
    centred_samples = (raw_bytes.astype(np.float32) - CU8_ZERO_LEVEL) / CU8_FULL_SCALE
    in_phase = centred_samples[0::2]
    quadrature = centred_samples[1::2]
    return (in_phase + 1j * quadrature).astype(np.complex64)


def polyphase_tap_count(preferred_tap_count: int, decimation_factor: int) -> int:
    """Round a tap count up until `(taps - 1)` is a multiple of the factor.

    Block-wise overlap-save needs this: the number of outputs to discard at the
    start of each block is `(taps - 1) / factor`, which has to be a whole
    number or the decimation phase slips from block to block.

    Offline filtering does not need the constraint, but both receivers call
    this anyway. Using one rule everywhere is what keeps the two from drifting
    apart -- they used to differ by a couple of taps for no good reason.
    """
    return preferred_tap_count + -(preferred_tap_count - 1) % decimation_factor


def design_station_filter() -> np.ndarray:
    """Anti-alias taps for 1.8 MS/s -> 300 kS/s, keeping the station's +/-100 kHz.

    Everything beyond +/-100 kHz is a neighbouring channel: it must be filtered
    out *before* decimating, or dropping five of every six samples folds it back
    on top of the wanted station.
    """
    return signal.firwin(
        polyphase_tap_count(STATION_FILTER_BASE_TAPS, BASEBAND_DECIMATION_FACTOR),
        cutoff=STATION_HALF_BANDWIDTH_HZ,
        fs=CAPTURE_SAMPLE_RATE_HZ,
    )


def design_mono_audio_filter() -> np.ndarray:
    """15 kHz taps that pass mono audio and bury the 19 kHz stereo pilot.

    The pilot sits only 4 kHz above the mono band, so the transition has to be
    steep -- a Kaiser design, which lands around 330 taps.

    Live, this filter doubles as the anti-alias filter for the second
    decimation, which is also by 6, so the same polyphase rounding applies.
    Offline does not need the rounding but takes it anyway: identical taps mean
    the two receivers sound identical, which matters more than four taps.
    """
    transition_width_normalised = (STEREO_PILOT_HZ - MONO_AUDIO_CUTOFF_HZ) / (
        BASEBAND_SAMPLE_RATE_HZ / 2
    )
    tap_count, kaiser_beta = signal.kaiserord(
        PILOT_ATTENUATION_DB, transition_width_normalised
    )
    return signal.firwin(
        polyphase_tap_count(tap_count | 1, BASEBAND_DECIMATION_FACTOR),
        cutoff=MONO_AUDIO_CUTOFF_HZ,
        window=("kaiser", kaiser_beta),
        fs=BASEBAND_SAMPLE_RATE_HZ,
    )


def discriminate_fm_to_hz(iq: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Recover instantaneous frequency deviation, in Hz, from complex I/Q.

    FM stores the audio in the *rate of change* of the carrier's phase, so the
    receiver has to differentiate the phase. Each sample is A*exp(j*phi[n]).
    Multiplying a sample by the conjugate of its predecessor subtracts the
    previous angle (exp(j*phi[n]) * exp(-j*phi[n-1]) == exp(j*(phi[n]-phi[n-1]))),
    so np.angle of that product is exactly the phase advanced during one sample
    period, already wrapped into (-pi, +pi] -- which is why no explicit phase
    unwrapping is needed.

    Converting that to Hz is just a change of units: one full cycle is 2*pi
    radians and consecutive samples are 1/sample_rate seconds apart, so

        deviation_hz = phase_change_rad / (2*pi) * sample_rate_hz

    The result is the audio waveform: silence sits at 0 Hz deviation and a loud
    passage swings out towards the +/-75 kHz broadcast limit. Note the output is
    one sample shorter than the input -- a difference needs two samples.
    """
    phase_change_rad = np.angle(iq[1:] * np.conj(iq[:-1]))
    return phase_change_rad * (sample_rate_hz / (2.0 * math.pi))


def deemphasis_coefficients(sample_rate_hz: float) -> tuple[list[float], list[float]]:
    """Single-pole IIR undoing the transmitter's treble boost, as (b, a).

    FM broadcasters pre-emphasise high frequencies to lift them above the noise;
    the receiver has to apply the mirror-image roll-off or everything sounds
    harsh. A 50 us time constant corresponds to a -3 dB corner near 3.2 kHz.
    The numerator is scaled for unity gain at DC.
    """
    pole = math.exp(-1.0 / (sample_rate_hz * DEEMPHASIS_TIME_CONSTANT_S))
    return [1.0 - pole], [1.0, -pole]
