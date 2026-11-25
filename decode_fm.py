#!/usr/bin/env python3
"""Offline broadcast-FM receiver: a raw CU8 I/Q capture becomes a mono WAV.

Milestone 1 of the SDR_radio project. This script is deliberately offline and
mono: it reads a file that was already captured from an RTL-SDR Blog V4, and
walks the samples through the same stages a hardware FM radio uses.

    python decode_fm.py radio_nacional_88_2.cu8 radio_nacional_88_2.wav

Signal chain, with the sample rate at every stage:

    file  1.8 MS/s complex  raw unsigned bytes -> centred complex I/Q
      |   anti-alias FIR, decimate by 6 (keeps the station's +/-100 kHz)
    base  300 kS/s complex  one station, centred at 0 Hz
      |   phase-difference discriminator (FM -> instantaneous frequency)
    disc  300 kS/s real     frequency deviation in Hz == the audio waveform
      |   15 kHz low-pass (kills the 19 kHz stereo pilot), 50 us de-emphasis
      |   rational resample 300 kS/s -> 48 kS/s (up 8, down 50)
    out   48 kS/s real      mono audio -> signed 16-bit PCM WAV
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

# --- Stage 1: the capture format, as documented for this recording -----------
CAPTURE_SAMPLE_RATE_HZ = 1_800_000
CU8_ZERO_LEVEL = 127.5  # midpoint of the unsigned 8-bit range 0..255
CU8_FULL_SCALE = 127.5  # divisor that maps 0..255 onto about -1.0..+1.0
BYTES_PER_COMPLEX_SAMPLE = 2  # one byte of I, one byte of Q

# --- Stage 2: isolating the centred station ---------------------------------
BASEBAND_DECIMATION_FACTOR = 6
BASEBAND_SAMPLE_RATE_HZ = CAPTURE_SAMPLE_RATE_HZ // BASEBAND_DECIMATION_FACTOR  # 300 kS/s
STATION_HALF_BANDWIDTH_HZ = 100_000  # a broadcast FM channel occupies about +/-100 kHz
STATION_FILTER_TAP_COUNT = 129  # odd, so the FIR has an integer group delay

# --- Stage 4: mono audio ----------------------------------------------------
AUDIO_SAMPLE_RATE_HZ = 48_000
MONO_AUDIO_CUTOFF_HZ = 15_000  # top of the mono (L+R) band
STEREO_PILOT_HZ = 19_000  # must be in the stopband or it whistles in the audio
PILOT_ATTENUATION_DB = 70.0
DEEMPHASIS_TIME_CONSTANT_S = 50e-6  # Europe (the Americas and Korea use 75 us)

# --- Stage 5: PCM output ----------------------------------------------------
INT16_MAX = 32_767
INT16_MIN = -32_768
TARGET_PEAK_FRACTION = 0.95  # leave headroom so rounding cannot clip
SILENCE_PEAK_THRESHOLD = 1e-9  # below this the capture is silence, not audio


class CaptureFormatError(ValueError):
    """The input file is not a usable interleaved unsigned 8-bit I/Q capture."""


def read_cu8_iq(capture_path: Path) -> np.ndarray:
    """Read interleaved unsigned 8-bit bytes (I0, Q0, I1, Q1, ...) as complex I/Q.

    The bytes are widened to float *before* the midpoint is subtracted: doing it
    the other way round would wrap around in unsigned 8-bit arithmetic and turn
    every negative sample into a large positive one.
    """
    raw_bytes = np.fromfile(capture_path, dtype=np.uint8)

    if raw_bytes.size == 0:
        raise CaptureFormatError(
            f"{capture_path} is empty: expected interleaved I/Q bytes, got 0 bytes."
        )
    if raw_bytes.size % BYTES_PER_COMPLEX_SAMPLE != 0:
        raise CaptureFormatError(
            f"{capture_path} holds {raw_bytes.size} bytes, which is odd. "
            "CU8 data is one I byte followed by one Q byte, so the length must be even. "
            "The file is probably truncated."
        )

    return cu8_bytes_to_iq(raw_bytes)


def cu8_bytes_to_iq(raw_bytes: np.ndarray) -> np.ndarray:
    """Convert interleaved unsigned 8-bit bytes into centred complex I/Q samples.

    Shared by the offline decoder and the live player, because `rtl_sdr` emits
    exactly the same bytes to a pipe as it writes to a file.
    """
    centred_samples = (raw_bytes.astype(np.float32) - CU8_ZERO_LEVEL) / CU8_FULL_SCALE
    in_phase = centred_samples[0::2]
    quadrature = centred_samples[1::2]
    return (in_phase + 1j * quadrature).astype(np.complex64)


def isolate_station(iq_at_capture_rate: np.ndarray) -> np.ndarray:
    """Low-pass to the station's own bandwidth, then decimate 1.8 MS/s -> 300 kS/s.

    The tuner was parked on the station, so the wanted signal already sits at
    0 Hz and no mixing is needed. Everything beyond +/-100 kHz is a neighbouring
    channel: it has to be filtered out *before* decimating, otherwise dropping
    five of every six samples folds it back on top of our station (aliasing).
    """
    anti_alias_taps = signal.firwin(
        STATION_FILTER_TAP_COUNT,
        cutoff=STATION_HALF_BANDWIDTH_HZ,
        fs=CAPTURE_SAMPLE_RATE_HZ,
    )
    # upfirdn filters and throws away the unused outputs in one polyphase pass.
    iq_at_baseband_rate = signal.upfirdn(
        anti_alias_taps, iq_at_capture_rate, down=BASEBAND_DECIMATION_FACTOR
    )
    return iq_at_baseband_rate.astype(np.complex64)


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
    passage swings out towards the +/-75 kHz broadcast limit.
    """
    phase_change_rad = np.angle(iq[1:] * np.conj(iq[:-1]))
    return phase_change_rad * (sample_rate_hz / (2.0 * math.pi))


def design_mono_audio_filter() -> np.ndarray:
    """FIR taps that pass mono audio to 15 kHz and bury the 19 kHz pilot tone."""
    transition_width_normalised = (STEREO_PILOT_HZ - MONO_AUDIO_CUTOFF_HZ) / (
        BASEBAND_SAMPLE_RATE_HZ / 2
    )
    tap_count, kaiser_beta = signal.kaiserord(
        PILOT_ATTENUATION_DB, transition_width_normalised
    )
    tap_count |= 1  # force odd for a symmetric, integer-delay filter
    return signal.firwin(
        tap_count,
        cutoff=MONO_AUDIO_CUTOFF_HZ,
        window=("kaiser", kaiser_beta),
        fs=BASEBAND_SAMPLE_RATE_HZ,
    )


def apply_deemphasis(audio: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Undo the transmitter's treble boost with a 50 us single-pole RC filter.

    FM broadcasters pre-emphasise high frequencies to lift them above the noise;
    the receiver has to apply the mirror-image roll-off or everything sounds
    harsh. A 50 us time constant corresponds to a -3 dB corner at about 3.2 kHz.
    """
    pole = math.exp(-1.0 / (sample_rate_hz * DEEMPHASIS_TIME_CONSTANT_S))
    numerator = [1.0 - pole]  # scaled for unity gain at DC
    denominator = [1.0, -pole]
    return signal.lfilter(numerator, denominator, audio)


def extract_mono_audio(deviation_hz_at_baseband_rate: np.ndarray) -> np.ndarray:
    """Band-limit, de-emphasise and resample the discriminator output to 48 kHz."""
    band_limited = signal.oaconvolve(
        deviation_hz_at_baseband_rate, design_mono_audio_filter(), mode="same"
    )
    deemphasised = apply_deemphasis(band_limited, BASEBAND_SAMPLE_RATE_HZ)

    # 300 kS/s -> 48 kS/s is the exact ratio 8/50, so a rational resampler is
    # enough; no interpolation guesswork.
    rate_gcd = math.gcd(AUDIO_SAMPLE_RATE_HZ, BASEBAND_SAMPLE_RATE_HZ)
    return signal.resample_poly(
        deemphasised,
        up=AUDIO_SAMPLE_RATE_HZ // rate_gcd,
        down=BASEBAND_SAMPLE_RATE_HZ // rate_gcd,
    )


def to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Remove residual DC and scale to signed 16-bit PCM without clipping.

    The discriminator leaves a DC offset whenever the tuner is a few kHz off
    centre; as audio that is inaudible but it wastes headroom, so it goes first.
    A silent capture has a peak of zero, so the scale factor is only computed
    once the signal is known to be above the silence threshold.
    """
    if audio.size == 0:
        return np.zeros(0, dtype=np.int16)

    without_dc = audio - np.mean(audio)
    peak_amplitude = float(np.max(np.abs(without_dc)))

    if peak_amplitude < SILENCE_PEAK_THRESHOLD:
        return np.zeros(without_dc.size, dtype=np.int16)

    scale = TARGET_PEAK_FRACTION * INT16_MAX / peak_amplitude
    return np.clip(np.round(without_dc * scale), INT16_MIN, INT16_MAX).astype(np.int16)


def decode_capture_to_wav(capture_path: Path, wav_path: Path) -> None:
    """Run the whole receiver chain and report each stage as it completes."""
    iq_at_capture_rate = read_cu8_iq(capture_path)
    print(
        f"capture       : {iq_at_capture_rate.size:>10,} complex samples @ "
        f"{CAPTURE_SAMPLE_RATE_HZ:,} S/s  "
        f"({iq_at_capture_rate.size / CAPTURE_SAMPLE_RATE_HZ:.3f} s)"
    )

    iq_at_baseband_rate = isolate_station(iq_at_capture_rate)
    print(
        f"station       : {iq_at_baseband_rate.size:>10,} complex samples @ "
        f"{BASEBAND_SAMPLE_RATE_HZ:,} S/s  "
        f"(kept +/-{STATION_HALF_BANDWIDTH_HZ / 1e3:.0f} kHz, "
        f"{STATION_FILTER_TAP_COUNT}-tap FIR, decimated by {BASEBAND_DECIMATION_FACTOR})"
    )

    deviation_hz = discriminate_fm_to_hz(iq_at_baseband_rate, BASEBAND_SAMPLE_RATE_HZ)
    print(
        f"discriminator : {deviation_hz.size:>10,} real samples    @ "
        f"{BASEBAND_SAMPLE_RATE_HZ:,} S/s  "
        f"(deviation rms {np.std(deviation_hz) / 1e3:.1f} kHz, "
        f"peak {np.max(np.abs(deviation_hz)) / 1e3:.1f} kHz)"
    )

    audio = extract_mono_audio(deviation_hz)
    print(
        f"audio         : {audio.size:>10,} real samples    @ "
        f"{AUDIO_SAMPLE_RATE_HZ:,} S/s  "
        f"({audio.size / AUDIO_SAMPLE_RATE_HZ:.3f} s, "
        f"{MONO_AUDIO_CUTOFF_HZ / 1e3:.0f} kHz low-pass, "
        f"{DEEMPHASIS_TIME_CONSTANT_S * 1e6:.0f} us de-emphasis)"
    )

    pcm16_audio = to_pcm16(audio)
    wavfile.write(wav_path, AUDIO_SAMPLE_RATE_HZ, pcm16_audio)
    print(f"wrote         : {wav_path}  (mono, 16-bit PCM, {AUDIO_SAMPLE_RATE_HZ:,} Hz)")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode a raw CU8 FM capture into a mono 48 kHz WAV file."
    )
    parser.add_argument(
        "capture_path",
        type=Path,
        help=f"input CU8 I/Q capture recorded at {CAPTURE_SAMPLE_RATE_HZ:,} S/s",
    )
    parser.add_argument("wav_path", type=Path, help="output mono WAV file to write")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        decode_capture_to_wav(arguments.capture_path, arguments.wav_path)
    except FileNotFoundError:
        print(f"error: no such capture file: {arguments.capture_path}", file=sys.stderr)
        return 2
    except IsADirectoryError:
        print(f"error: {arguments.capture_path} is a directory, not a capture", file=sys.stderr)
        return 2
    except CaptureFormatError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
