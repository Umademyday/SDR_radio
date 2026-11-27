#!/usr/bin/env python3
"""Optional live oscilloscope and spectrum analyser for the decoded FM audio.

Imported only when `play_fm.py --scope` is used, so matplotlib stays an optional
dependency: without the flag the receiver never touches this module.

Two rules shape the design:

* **macOS insists the GUI runs on the main thread.** So `play_fm.main` keeps the
  main thread for `AudioScope.run()` and moves the receive loop to a worker.
* **Plotting must stay out of the audio path.** The receive loop only drops a
  reference into `LatestAudioBlock`; all drawing and every FFT happens on the
  GUI timer, once per displayed block.
"""

from __future__ import annotations

import threading

import numpy as np
from matplotlib import pyplot

from fm_dsp import INT16_MAX

SCOPE_WINDOW_S = 0.020  # 20 ms on screen; 1,000 samples at the current 50 kS/s
REDRAW_INTERVAL_MS = 40  # the receiver produces a block every ~41 ms
LINE_WIDTH = 0.8

# --- spectrum ---------------------------------------------------------------
# 4,096 samples at 50 kS/s is 12.2 Hz per bin. One block only carries 2,048
# samples, so the scope keeps the last two blocks and transforms them together:
# real resolution, rather than the interpolation that zero-padding would give.
SPECTRUM_SAMPLES = 4_096
SPECTRUM_MAX_HZ = 15_000  # the mono audio band; everything above it was filtered out
SPECTRUM_FLOOR_DBFS = -90.0
AMPLITUDE_FLOOR = 1e-12  # keeps log10 away from zero on digital silence


class LatestAudioBlock:
    """Single-slot handoff from the audio thread to the GUI thread.

    Publishing overwrites whatever has not been drawn yet, which is exactly the
    "skip stale display frames" rule: the audio path never blocks on the GUI,
    and a slow redraw can never build up a backlog of frames to catch up on.
    The newest block simply wins.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: np.ndarray | None = None

    def publish(self, audio_pcm16: np.ndarray) -> None:
        """Called from the audio thread. Costs a lock and a reference store."""
        with self._lock:
            self._pending = audio_pcm16

    def take(self) -> np.ndarray | None:
        """Called from the GUI thread. Returns None when nothing is new."""
        with self._lock:
            pending, self._pending = self._pending, None
        return pending


def window_sample_count(sample_rate_hz: int) -> int:
    """How many samples make up the visible window, derived from the audio rate."""
    return int(round(SCOPE_WINDOW_S * sample_rate_hz))


def latest_window(audio_pcm16: np.ndarray, sample_count: int) -> np.ndarray:
    """The newest `sample_count` samples, normalised to a fixed -1..+1 scale.

    The scale is deliberately fixed rather than fitted to each block: dividing
    by the full-scale PCM value means loud speech draws a taller trace than
    quiet speech, which is the whole point of watching the waveform.
    """
    newest = audio_pcm16[-sample_count:]
    normalised = newest.astype(np.float64) / INT16_MAX

    if normalised.size < sample_count:  # a short final block: pad on the left
        padding = np.zeros(sample_count - normalised.size)
        normalised = np.concatenate((padding, normalised))
    return normalised


def triggered_window(audio_pcm16: np.ndarray, sample_count: int) -> np.ndarray:
    """Like `latest_window`, but aligned to a rising zero crossing.

    Without this the window starts wherever the block happens to end, so the
    waveform slides sideways from frame to frame and reads as jitter even when
    the audio is steady. A bench oscilloscope solves it the same way: start
    drawing at a consistent point in the signal.

    The newest crossing that still leaves a full window is chosen, so the
    display stays as fresh as possible. Silence has no crossings; then this
    falls back to simply showing the newest samples.
    """
    if audio_pcm16.size < 2 * sample_count:
        return latest_window(audio_pcm16, sample_count)

    recent = audio_pcm16[-2 * sample_count:]
    candidate_starts = recent[: sample_count + 1]
    rising_edges = np.flatnonzero(
        (candidate_starts[:-1] <= 0) & (candidate_starts[1:] > 0)
    )
    start = int(rising_edges[-1]) if rising_edges.size else sample_count
    return recent[start : start + sample_count].astype(np.float64) / INT16_MAX


def spectrum_frequencies_hz(sample_rate_hz: float) -> np.ndarray:
    """Bin centre frequencies for the spectrum, in Hz (about 12.2 Hz apart)."""
    return np.fft.rfftfreq(SPECTRUM_SAMPLES, 1.0 / sample_rate_hz)


def amplitude_spectrum_dbfs(audio_pcm16: np.ndarray) -> np.ndarray:
    """One-sided amplitude spectrum of the newest samples, in dBFS.

    Calibrated so a bin-centred sine of peak amplitude A reads 20*log10(A):
    half scale is -6 dBFS. Two corrections get it there.

    * **Window gain.** A Hann window throws away energy, so the raw bin
      magnitude is divided by `sum(window)` rather than by the sample count.
    * **One-sided folding.** A real signal splits its energy between positive
      and negative frequencies; dropping the negative half means doubling what
      is left. DC and Nyquist have no twin to fold in, so they are not doubled.

    Digital silence would be log10(0), so magnitudes are floored first.
    """
    newest = audio_pcm16[-SPECTRUM_SAMPLES:].astype(np.float64) / INT16_MAX
    if newest.size < SPECTRUM_SAMPLES:  # not enough history yet: pad on the left
        newest = np.concatenate((np.zeros(SPECTRUM_SAMPLES - newest.size), newest))

    window = np.hanning(SPECTRUM_SAMPLES)
    bins = np.fft.rfft(newest * window)
    amplitude = 2.0 * np.abs(bins) / np.sum(window)
    amplitude[0] /= 2.0
    if SPECTRUM_SAMPLES % 2 == 0:
        amplitude[-1] /= 2.0

    return 20.0 * np.log10(np.maximum(amplitude, AMPLITUDE_FLOOR))


class AudioScope:
    """One window, two plots: the audio waveform and its spectrum."""

    def __init__(
        self,
        sample_rate_hz: int,
        station_mhz: float,
        latest_block: LatestAudioBlock,
        stop,
    ) -> None:
        self._latest_block = latest_block
        self._stop = stop
        self._sample_count = window_sample_count(sample_rate_hz)
        time_axis_ms = np.arange(self._sample_count) / sample_rate_hz * 1000.0

        # The spectrum needs more samples than one block carries, so the two
        # most recent blocks are kept and transformed together.
        self._history = np.zeros(SPECTRUM_SAMPLES, dtype=np.int16)
        frequencies_hz = spectrum_frequencies_hz(sample_rate_hz)
        self._audible_bins = frequencies_hz <= SPECTRUM_MAX_HZ

        self._figure, (waveform_axes, spectrum_axes) = pyplot.subplots(
            2, 1, figsize=(8.0, 6.5)
        )
        self._figure.canvas.manager.set_window_title(
            f"FM audio scope - {station_mhz:.1f} MHz"
        )

        (self._waveform_line,) = waveform_axes.plot(
            time_axis_ms, np.zeros(self._sample_count), linewidth=LINE_WIDTH
        )
        waveform_axes.set_xlim(0.0, float(time_axis_ms[-1]))
        waveform_axes.set_ylim(-1.0, 1.0)  # fixed for the session, never autoscaled
        waveform_axes.set_xlabel("Time (ms)")
        waveform_axes.set_ylabel("Amplitude")
        waveform_axes.grid(True, alpha=0.3)

        (self._spectrum_line,) = spectrum_axes.plot(
            frequencies_hz[self._audible_bins] / 1000.0,
            np.full(int(np.count_nonzero(self._audible_bins)), SPECTRUM_FLOOR_DBFS),
            linewidth=LINE_WIDTH,
        )
        spectrum_axes.set_xlim(0.0, SPECTRUM_MAX_HZ / 1000.0)
        spectrum_axes.set_ylim(SPECTRUM_FLOOR_DBFS, 0.0)
        spectrum_axes.set_xlabel("Audio frequency (kHz)")
        spectrum_axes.set_ylabel("Level (dBFS)")
        spectrum_axes.grid(True, alpha=0.3)

        self._figure.tight_layout()

        self._timer = self._figure.canvas.new_timer(interval=REDRAW_INTERVAL_MS)
        self._timer.add_callback(self._redraw_if_new_audio)
        self._figure.canvas.mpl_connect("close_event", self._on_window_closed)

    def _redraw_if_new_audio(self) -> None:
        if self._stop.requested:
            pyplot.close(self._figure)  # unblocks run() on Ctrl-C
            return

        pending = self._latest_block.take()
        if pending is None:
            return  # nothing new since the last tick, so nothing to redraw

        # One block, both plots: decoded once, drawn once.
        self._waveform_line.set_ydata(triggered_window(pending, self._sample_count))

        self._history = np.concatenate((self._history, pending))[-SPECTRUM_SAMPLES:]
        self._spectrum_line.set_ydata(
            amplitude_spectrum_dbfs(self._history)[self._audible_bins]
        )

        self._figure.canvas.draw_idle()  # both lines reused, nothing replotted

    def _on_window_closed(self, _event) -> None:
        self._stop.request()

    def run(self) -> None:
        """Hold the main thread until the window closes or a stop is requested."""
        self._timer.start()
        pyplot.show()
        self._stop.request()  # the window was closed: tell the receiver to finish
