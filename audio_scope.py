#!/usr/bin/env python3
"""Optional live oscilloscope for the decoded FM audio.

Imported only when `play_fm.py --scope` is used, so matplotlib stays an optional
dependency: without the flag the receiver never touches this module.

Two rules shape the design:

* **macOS insists the GUI runs on the main thread.** So `play_fm.main` keeps the
  main thread for `AudioScope.run()` and moves the receive loop to a worker.
* **Plotting must stay out of the audio path.** The receive loop only drops a
  reference into `LatestAudioBlock`; all drawing happens on the GUI timer.
"""

from __future__ import annotations

import threading

import numpy as np
from matplotlib import pyplot

from fm_dsp import INT16_MAX

SCOPE_WINDOW_S = 0.020  # 20 ms on screen; 1,000 samples at the current 50 kS/s
REDRAW_INTERVAL_MS = 40  # the receiver only produces a block every ~164 ms
LINE_WIDTH = 0.8


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


class AudioScope:
    """A window showing the most recent 20 ms of the audio being played."""

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

        self._figure, axes = pyplot.subplots(figsize=(8.0, 3.5))
        self._figure.canvas.manager.set_window_title(
            f"FM audio scope - {station_mhz:.1f} MHz"
        )
        (self._line,) = axes.plot(
            time_axis_ms, np.zeros(self._sample_count), linewidth=LINE_WIDTH
        )
        axes.set_xlim(0.0, float(time_axis_ms[-1]))
        axes.set_ylim(-1.0, 1.0)  # fixed for the whole session, never autoscaled
        axes.set_xlabel("Time (ms)")
        axes.set_ylabel("Amplitude")
        axes.grid(True, alpha=0.3)
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

        self._line.set_ydata(latest_window(pending, self._sample_count))
        self._figure.canvas.draw_idle()  # reuses the same line, no replotting

    def _on_window_closed(self, _event) -> None:
        self._stop.request()

    def run(self) -> None:
        """Hold the main thread until the window closes or a stop is requested."""
        self._timer.start()
        pyplot.show()
        self._stop.request()  # the window was closed: tell the receiver to finish
