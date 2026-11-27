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

# The radio itself lives in fm_dsp, shared with the live receiver in play_fm.
from fm_dsp import (
    BASEBAND_DECIMATION_FACTOR,
    BASEBAND_SAMPLE_RATE_HZ,
    BYTES_PER_COMPLEX_SAMPLE,
    CAPTURE_SAMPLE_RATE_HZ,
    DEEMPHASIS_TIME_CONSTANT_S,
    INT16_MAX,
    INT16_MIN,
    MONO_AUDIO_CUTOFF_HZ,
    STATION_HALF_BANDWIDTH_HZ,
    cu8_bytes_to_iq,
    deemphasis_coefficients,
    design_mono_audio_filter,
    design_station_filter,
    discriminate_fm_to_hz,
)

# --- what is specific to decoding a file ------------------------------------
AUDIO_SAMPLE_RATE_HZ = 48_000  # the live receiver uses 50 kHz; see fm_dsp
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


def isolate_station(iq_at_capture_rate: np.ndarray) -> np.ndarray:
    """Low-pass to the station's own bandwidth, then decimate 1.8 MS/s -> 300 kS/s.

    The tuner was parked on the station, so the wanted signal already sits at
    0 Hz and no mixing is needed. `upfirdn` filters and throws away the unused
    outputs in one polyphase pass.
    """
    iq_at_baseband_rate = signal.upfirdn(
        design_station_filter(), iq_at_capture_rate, down=BASEBAND_DECIMATION_FACTOR
    )
    return iq_at_baseband_rate.astype(np.complex64)


def extract_mono_audio(deviation_hz_at_baseband_rate: np.ndarray) -> np.ndarray:
    """Band-limit, de-emphasise and resample the discriminator output to 48 kHz.

    Offline the whole capture is filtered in one pass, so no filter state has
    to survive anything -- the one real difference from the live receiver.
    """
    band_limited = signal.oaconvolve(
        deviation_hz_at_baseband_rate, design_mono_audio_filter(), mode="same"
    )
    deemphasised = signal.lfilter(
        *deemphasis_coefficients(BASEBAND_SAMPLE_RATE_HZ), band_limited
    )

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
        f"{design_station_filter().size}-tap FIR, decimated by {BASEBAND_DECIMATION_FACTOR})"
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
