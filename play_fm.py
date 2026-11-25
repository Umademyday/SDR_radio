#!/usr/bin/env python3
"""Live broadcast-FM receiver: dongle -> speakers, with no file in between.

Same signal chain as decode_fm.py, but fed by the RTL-SDR Blog V4 in real time
and piped to the Mac's speakers through sox's `play`.

Samples arrive from the `rtl_sdr` command as raw CU8 bytes -- byte for byte the
same format decode_fm.py reads from a file -- so both paths share one reader.

    python play_fm.py                 # Radio Nacional, 88.2 MHz
    python play_fm.py --station 101.3
    python play_fm.py --seconds 10    # stop by itself instead of Ctrl-C

The only real difference from offline decoding is *continuity*: samples arrive in
blocks, so every filter has to remember where the previous block left off.
A stateless filter restarted on each block clicks 6 times a second. Hence the
small stateful wrappers below.

The live chain also decimates by 6 twice, which lands on 50 kHz audio rather
than 48 kHz. That keeps every stage an exact integer ratio -- no rational
resampler to carry state across -- and `play` handles 50 kHz output fine.

    1.8 MS/s complex --(/6)--> 300 kS/s complex --discriminator-->
    300 kS/s deviation --(15 kHz LPF, /6)--> 50 kS/s audio --> speakers
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys

import numpy as np
from scipy import signal

from decode_fm import (
    BASEBAND_DECIMATION_FACTOR,
    cu8_bytes_to_iq,
    BASEBAND_SAMPLE_RATE_HZ,
    CAPTURE_SAMPLE_RATE_HZ,
    DEEMPHASIS_TIME_CONSTANT_S,
    INT16_MAX,
    MONO_AUDIO_CUTOFF_HZ,
    PILOT_ATTENUATION_DB,
    STATION_HALF_BANDWIDTH_HZ,
    STEREO_PILOT_HZ,
)

BYTES_PER_COMPLEX_SAMPLE = 2
AUTO_GAIN_FLAG = "0"  # rtl_sdr reads a gain of 0 as "let the tuner decide"

DEFAULT_STATION_MHZ = 88.2  # Radio Nacional, Madrid
DEFAULT_TUNER_GAIN_DB = 33.8  # a good starting point for strong local FM

AUDIO_DECIMATION_FACTOR = 6
LIVE_AUDIO_SAMPLE_RATE_HZ = BASEBAND_SAMPLE_RATE_HZ // AUDIO_DECIMATION_FACTOR  # 50 kS/s

# One USB read. Must be a multiple of 512 (librtlsdr) and of 36 (so both
# decimators keep their phase from block to block). 294912 = 512*576 = 36*8192.
READ_BLOCK_SAMPLES = 294_912
BLOCK_DURATION_S = READ_BLOCK_SAMPLES / CAPTURE_SAMPLE_RATE_HZ  # ~164 ms

BROADCAST_PEAK_DEVIATION_HZ = 75_000.0  # full modulation -> full audio scale
DC_BLOCKER_POLE = 0.999  # high-pass corner at about 8 Hz at 50 kS/s
DEFAULT_VOLUME = 0.8
SHUTDOWN_GRACE_S = 2.0  # how long a child gets to exit before it is killed


class PolyphaseDecimator:
    """Anti-alias FIR + decimation that survives block boundaries.

    `upfirdn` is stateless: on its own it would start each block with an empty
    filter and click. Overlap-save removes the seam -- keep the last
    len(taps)-1 input samples, prepend them to the next block, and discard the
    outputs they are responsible for.
    """

    def __init__(self, taps: np.ndarray, factor: int, dtype: np.dtype) -> None:
        if (taps.size - 1) % factor != 0:
            raise ValueError(
                f"{taps.size}-tap filter cannot hold decimation phase for factor {factor}: "
                "len(taps) - 1 must be a multiple of the factor."
            )
        self._taps = taps
        self._factor = factor
        self._overlap = np.zeros(taps.size - 1, dtype=dtype)
        self._outputs_to_discard = (taps.size - 1) // factor

    def process(self, block: np.ndarray) -> np.ndarray:
        extended_block = np.concatenate((self._overlap, block))
        self._overlap = extended_block[-(self._taps.size - 1):]
        decimated = signal.upfirdn(self._taps, extended_block, down=self._factor)
        first_valid = self._outputs_to_discard
        return decimated[first_valid : first_valid + block.size // self._factor]


class FmDiscriminator:
    """Phase-difference FM discriminator that remembers the previous sample.

    Identical maths to decode_fm.discriminate_fm_to_hz -- angle of
    iq[n] * conj(iq[n-1]), scaled by sample_rate / 2*pi -- except that the
    "previous sample" for the first sample of a block is the last sample of the
    block before it, so no sample is lost at the seam.
    """

    def __init__(self, sample_rate_hz: float) -> None:
        self._radians_to_hz = sample_rate_hz / (2.0 * math.pi)
        self._previous_sample: complex | None = None

    def process(self, iq: np.ndarray) -> np.ndarray:
        if self._previous_sample is None:
            self._previous_sample = iq[0]
        joined = np.concatenate(([self._previous_sample], iq))
        self._previous_sample = iq[-1]
        return np.angle(joined[1:] * np.conj(joined[:-1])) * self._radians_to_hz


class StatefulIirFilter:
    """scipy lfilter that carries its delay line between blocks."""

    def __init__(self, numerator: list[float], denominator: list[float]) -> None:
        self._numerator = numerator
        self._denominator = denominator
        self._state = np.zeros(max(len(numerator), len(denominator)) - 1)

    def process(self, block: np.ndarray) -> np.ndarray:
        filtered, self._state = signal.lfilter(
            self._numerator, self._denominator, block, zi=self._state
        )
        return filtered


def design_station_filter() -> np.ndarray:
    """Anti-alias taps for 1.8 MS/s -> 300 kS/s, keeping the station's +/-100 kHz."""
    return signal.firwin(
        127,  # 126 = 21*6, so the decimation phase is preserved across blocks
        cutoff=STATION_HALF_BANDWIDTH_HZ,
        fs=CAPTURE_SAMPLE_RATE_HZ,
    )


def design_audio_filter() -> np.ndarray:
    """15 kHz taps that also anti-alias the 300 kS/s -> 50 kS/s decimation.

    One filter does both jobs: it keeps mono audio, buries the 19 kHz stereo
    pilot, and clears everything above the new 25 kHz Nyquist.
    """
    transition_width_normalised = (STEREO_PILOT_HZ - MONO_AUDIO_CUTOFF_HZ) / (
        BASEBAND_SAMPLE_RATE_HZ / 2
    )
    tap_count, kaiser_beta = signal.kaiserord(
        PILOT_ATTENUATION_DB, transition_width_normalised
    )
    tap_count += -(tap_count - 1) % AUDIO_DECIMATION_FACTOR  # keep (taps-1) % 6 == 0
    return signal.firwin(
        tap_count,
        cutoff=MONO_AUDIO_CUTOFF_HZ,
        window=("kaiser", kaiser_beta),
        fs=BASEBAND_SAMPLE_RATE_HZ,
    )


def make_deemphasis_filter() -> StatefulIirFilter:
    """European 50 us de-emphasis as a single-pole IIR at the audio rate."""
    pole = math.exp(-1.0 / (LIVE_AUDIO_SAMPLE_RATE_HZ * DEEMPHASIS_TIME_CONSTANT_S))
    return StatefulIirFilter([1.0 - pole], [1.0, -pole])


def make_dc_blocker() -> StatefulIirFilter:
    """Gentle high-pass that removes the tuning-offset DC the discriminator leaves.

    Live audio cannot be normalised against a whole-file peak, so the DC has to
    go continuously rather than as one mean subtraction.
    """
    return StatefulIirFilter([1.0, -1.0], [1.0, -DC_BLOCKER_POLE])


class FmReceiver:
    """The whole live chain: one complex I/Q block in, one PCM audio block out."""

    def __init__(self, volume: float) -> None:
        self._station_decimator = PolyphaseDecimator(
            design_station_filter(), BASEBAND_DECIMATION_FACTOR, np.complex128
        )
        self._discriminator = FmDiscriminator(BASEBAND_SAMPLE_RATE_HZ)
        self._audio_decimator = PolyphaseDecimator(
            design_audio_filter(), AUDIO_DECIMATION_FACTOR, np.float64
        )
        self._deemphasis = make_deemphasis_filter()
        self._dc_blocker = make_dc_blocker()
        self._volume = volume
        self.last_deviation_rms_hz = 0.0
        self.last_audio_peak_fraction = 0.0

    def process(self, iq_at_capture_rate: np.ndarray) -> np.ndarray:
        iq_at_baseband_rate = self._station_decimator.process(iq_at_capture_rate)
        deviation_hz = self._discriminator.process(iq_at_baseband_rate)
        audio_hz = self._audio_decimator.process(deviation_hz)
        audio = self._deemphasis.process(audio_hz)
        audio = self._dc_blocker.process(audio)

        # Fixed scaling, not per-block normalisation: a block-by-block automatic
        # gain would pump the volume up during quiet passages.
        normalised = audio * (self._volume / BROADCAST_PEAK_DEVIATION_HZ)

        self.last_deviation_rms_hz = float(np.std(deviation_hz))
        self.last_audio_peak_fraction = float(np.max(np.abs(normalised)))
        return (np.clip(normalised, -1.0, 1.0) * INT16_MAX).astype(np.int16)


def open_speakers() -> subprocess.Popen:
    """Start sox's `play` reading raw mono 16-bit PCM from its stdin."""
    command = [
        "play", "-q",
        "-t", "raw",
        "-r", str(LIVE_AUDIO_SAMPLE_RATE_HZ),
        "-e", "signed",
        "-b", "16",
        "-c", "1",
        "-",
    ]
    try:
        return subprocess.Popen(command, stdin=subprocess.PIPE)
    except FileNotFoundError:
        raise SystemExit(
            "error: `play` not found. Install sox with:  brew install sox"
        )


def open_dongle_stream(station_hz: float, gain_db: float | str) -> subprocess.Popen:
    """Start `rtl_sdr` writing raw CU8 to a pipe.

    The dongle is driven through the `rtl_sdr` command rather than the pyrtlsdr
    bindings: pyrtlsdr 0.5.1 insists on the rtlsdr_set_dithering symbol, which
    the V4-aware librtlsdr 2.0.3 does not export, so importing it fails outright.
    Piping costs nothing and has a real advantage -- `rtl_sdr` emits the very
    same CU8 bytes that decode_fm.py reads from a file, so live and offline
    input share one reader.
    """
    gain_argument = AUTO_GAIN_FLAG if gain_db == "auto" else f"{float(gain_db):.1f}"
    command = [
        "rtl_sdr",
        "-f", str(int(station_hz)),
        "-s", str(int(CAPTURE_SAMPLE_RATE_HZ)),
        "-g", gain_argument,
        "-",
    ]
    try:
        return subprocess.Popen(command, stdout=subprocess.PIPE)
    except FileNotFoundError:
        raise SystemExit(
            "error: `rtl_sdr` not found. Install it with:  brew install librtlsdr"
        )


def read_exactly(stream, byte_count: int) -> bytes:
    """Read a whole block, since a pipe is free to hand back short reads."""
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break  # rtl_sdr exited
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_until_interrupted(
    station_hz: float, gain_db: float | str, receiver: FmReceiver, seconds: float | None
) -> None:
    """Pump blocks from the dongle to the speakers until Ctrl-C or the time limit."""
    dongle = open_dongle_stream(station_hz, gain_db)
    speakers = open_speakers()
    block_bytes = READ_BLOCK_SAMPLES * BYTES_PER_COMPLEX_SAMPLE
    blocks_to_play = None if seconds is None else math.ceil(seconds / BLOCK_DURATION_S)
    blocks_played = 0

    print(
        f"tuned to {station_hz / 1e6:.1f} MHz, gain {gain_db} dB, "
        f"{CAPTURE_SAMPLE_RATE_HZ / 1e6:.1f} MS/s -> "
        f"{LIVE_AUDIO_SAMPLE_RATE_HZ:,} Hz audio"
    )
    print("Ctrl-C to stop.\n")

    try:
        while blocks_to_play is None or blocks_played < blocks_to_play:
            raw_block = read_exactly(dongle.stdout, block_bytes)
            if len(raw_block) < block_bytes:
                print("\nthe dongle stopped sending samples.", file=sys.stderr)
                break

            iq_block = cu8_bytes_to_iq(np.frombuffer(raw_block, dtype=np.uint8))
            speakers.stdin.write(receiver.process(iq_block).tobytes())
            blocks_played += 1
            print(
                f"\r{blocks_played * BLOCK_DURATION_S:6.1f} s   "
                f"deviation {receiver.last_deviation_rms_hz / 1e3:5.1f} kHz rms   "
                f"audio {receiver.last_audio_peak_fraction * 100:5.1f} % of full scale",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nstopped.")
    except BrokenPipeError:
        print("\nerror: the audio player exited.", file=sys.stderr)
    finally:
        shut_down(dongle, speakers)


def shut_down(dongle: subprocess.Popen, speakers: subprocess.Popen) -> None:
    """Stop both child processes without deadlocking on a full pipe.

    `rtl_sdr` can be parked inside a write to a pipe nobody is draining any
    more, where a polite SIGTERM may never be acted on. Closing our read end
    first breaks that write, and anything still alive after the grace period
    is killed outright.
    """
    dongle.terminate()
    if dongle.stdout:
        dongle.stdout.close()

    if speakers.stdin:
        try:
            speakers.stdin.close()  # lets `play` drain what is already buffered
        except BrokenPipeError:
            pass

    for process in (dongle, speakers):
        try:
            process.wait(timeout=SHUTDOWN_GRACE_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive a broadcast FM station live and play it on the speakers."
    )
    parser.add_argument(
        "--station", type=float, default=DEFAULT_STATION_MHZ,
        help=f"station frequency in MHz (default: {DEFAULT_STATION_MHZ})",
    )
    parser.add_argument(
        "--gain", default=DEFAULT_TUNER_GAIN_DB,
        help=f"tuner gain in dB, or 'auto' (default: {DEFAULT_TUNER_GAIN_DB})",
    )
    parser.add_argument(
        "--volume", type=float, default=DEFAULT_VOLUME,
        help=f"output level, 0.0 to 1.0 (default: {DEFAULT_VOLUME})",
    )
    parser.add_argument(
        "--seconds", type=float, default=None,
        help="stop after this many seconds (default: run until Ctrl-C)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    gain_db = "auto" if str(arguments.gain).lower() == "auto" else float(arguments.gain)

    receiver = FmReceiver(volume=arguments.volume)
    receive_until_interrupted(
        arguments.station * 1e6, gain_db, receiver, arguments.seconds
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
