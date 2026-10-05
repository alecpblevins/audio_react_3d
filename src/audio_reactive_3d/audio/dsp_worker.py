"""Background DSP thread: ring buffer -> features -> smoothing -> publish.

Pulls the latest audio window from the capture ring buffer, runs it through
:class:`~audio_reactive_3d.audio.features.FeatureExtractor`, applies
attack/release smoothing via :class:`~audio_reactive_3d.audio.smoother.Smoother`,
and publishes the result as a single atomic reference swap -- never
blocking (or being blocked by) the render thread.
"""

from __future__ import annotations

import logging
import threading
import time

from audio_reactive_3d import config
from audio_reactive_3d.audio.features import BPM_INDEX, FeatureExtractor, FeatureFrame
from audio_reactive_3d.audio.ring_buffer import RingBuffer
from audio_reactive_3d.audio.smoother import Smoother

logger = logging.getLogger(__name__)

#: Number of leading elements (spectrum + low/mid/high + RMS) that receive
#: attack/release smoothing. The onset flag (index ONSET_INDEX) is an
#: instantaneous event and must not be smoothed; BPM is smoothed separately
#: with slower ballistics.
_SMOOTHED_PREFIX_LENGTH = config.SPECTRUM_BINS + 4


class SharedFeatureState:
    """Cross-thread handoff point for the latest feature frame.

    A plain attribute holding an immutable object reference is swapped
    atomically under the GIL, so the render thread can read ``.latest``
    without a lock while the DSP thread writes a brand new
    :class:`FeatureFrame` each tick -- no torn reads, no blocking either
    side.
    """

    def __init__(self) -> None:
        self.latest: FeatureFrame | None = None


class DSPWorker:
    """Runs :class:`FeatureExtractor` on live audio on a background thread.

    Parameters
    ----------
    ring_buffer:
        Source of raw audio frames (written by the capture thread).
    shared:
        Destination for the latest computed :class:`FeatureFrame`.
    target_hz:
        Desired feature-extraction rate; the loop paces itself to this
        cadence using the monotonic clock (no audio-driven blocking).
    """

    def __init__(
        self,
        ring_buffer: RingBuffer,
        shared: SharedFeatureState,
        target_hz: float = config.DSP_TARGET_HZ,
        fft_size: int = config.FFT_SIZE,
        sample_rate: int = config.SAMPLE_RATE,
    ) -> None:
        self._ring_buffer = ring_buffer
        self._shared = shared
        self._target_hz = target_hz
        self._fft_size = fft_size

        self._extractor = FeatureExtractor(sample_rate=sample_rate, fft_size=fft_size)
        self._value_smoother = Smoother(shape=(_SMOOTHED_PREFIX_LENGTH,))
        self._bpm_smoother = Smoother(
            attack_seconds=config.BPM_ATTACK_SECONDS,
            release_seconds=config.BPM_RELEASE_SECONDS,
        )

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        #: Frames successfully computed and published.
        self.frames_processed = 0
        #: Loop iterations skipped because not enough audio was buffered yet
        #: (e.g. at startup) -- not the same as "dropped" real-time frames.
        self.frames_skipped = 0

    def start(self) -> None:
        """Start the background DSP thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="dsp-worker", daemon=True)
        self._thread.start()
        logger.info("Started DSP worker at target %.1f Hz", self._target_hz)

    def stop(self) -> None:
        """Signal the DSP thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("Stopped DSP worker")

    def _run(self) -> None:
        period = 1.0 / self._target_hz
        last_time = time.perf_counter()

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            chunk = self._ring_buffer.peek_latest(self._fft_size)
            if chunk is None:
                self.frames_skipped += 1
                time.sleep(period)
                continue

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            raw = self._extractor.process(chunk, timestamp=now)
            raw[:_SMOOTHED_PREFIX_LENGTH] = self._value_smoother.process(
                raw[:_SMOOTHED_PREFIX_LENGTH], dt
            )
            raw[BPM_INDEX] = self._bpm_smoother.process(raw[BPM_INDEX], dt)

            # Atomic reference swap: safe to read concurrently without a lock.
            self._shared.latest = FeatureFrame(vector=raw, timestamp=now)
            self.frames_processed += 1

            elapsed = time.perf_counter() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.debug("DSP worker running behind schedule by %.1f ms", -sleep_time * 1000)
