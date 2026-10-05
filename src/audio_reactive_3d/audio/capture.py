"""System-audio loopback capture using SoundCard, backed by a ring buffer.

The capture callback (``_capture_loop``) is intentionally dumb: it only
pulls raw float32 frames from the OS loopback device and pushes them into a
preallocated :class:`~audio_reactive_3d.audio.ring_buffer.RingBuffer`. It
performs zero DSP and zero heap allocation per iteration (SoundCard's
``record()`` call does allocate its own return buffer internally, but we
hand that same array straight to the ring buffer without any further
copying/processing here).

On Windows this uses WASAPI loopback on the default speaker. On Linux it
relies on a PulseAudio/PipeWire monitor source; on macOS it requires a
loopback driver such as BlackHole. If no loopback-capable device can be
found, :func:`open_default_loopback_microphone` raises a clear,
non-crashing ``RuntimeError`` that callers can catch and report.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

try:
    import soundcard as sc
except Exception as exc:  # pragma: no cover - import-time environment issue
    sc = None  # type: ignore[assignment]
    _IMPORT_ERROR: Exception | None = exc
else:
    _IMPORT_ERROR = None

from audio_reactive_3d import config
from audio_reactive_3d.audio.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Lightweight, picklable description of a capturable loopback device."""

    name: str
    id: str
    is_default: bool


def list_loopback_devices() -> list[DeviceInfo]:
    """Return every speaker (output) device available for loopback capture.

    Raises
    ------
    RuntimeError
        If SoundCard failed to import (e.g. unsupported platform/backend
        missing) or no audio backend is available on this system.
    """
    if sc is None:
        raise RuntimeError(
            "SoundCard is unavailable on this system "
            f"({_IMPORT_ERROR!r}). Loopback capture cannot proceed."
        )

    try:
        speakers = sc.all_speakers()
        default = sc.default_speaker()
    except Exception as exc:  # pragma: no cover - depends on host audio stack
        raise RuntimeError(f"Could not enumerate audio output devices: {exc!r}") from exc

    default_name = default.name if default is not None else None
    return [
        DeviceInfo(name=s.name, id=s.id, is_default=(s.name == default_name)) for s in speakers
    ]


class LoopbackCapture:
    """Captures system audio output into a ring buffer on a background thread.

    Parameters
    ----------
    ring_buffer:
        Preallocated buffer the capture thread writes into.
    device_name:
        Substring to match against output device names, or ``None`` to use
        the system default output device.
    sample_rate, channels, block_size:
        Requested capture format; see ``config.py`` for defaults.
    """

    def __init__(
        self,
        ring_buffer: RingBuffer,
        device_name: str | None = None,
        sample_rate: int = config.SAMPLE_RATE,
        channels: int = config.CHANNELS,
        block_size: int = config.BLOCK_SIZE,
    ) -> None:
        if sc is None:
            raise RuntimeError(
                "SoundCard is unavailable on this system "
                f"({_IMPORT_ERROR!r}). Loopback capture cannot proceed."
            )

        self._ring_buffer = ring_buffer
        self._sample_rate = sample_rate
        self._channels = channels
        self._block_size = block_size

        self._speaker = self._resolve_speaker(device_name)
        self.device_name: str = self._speaker.name

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        #: Cleared by the capture thread if the underlying stream raises
        #: unexpectedly (e.g. the output device was unplugged/disabled).
        #: Callers (e.g. the render loop) can poll this to shut down
        #: cleanly instead of silently freezing with stale audio features.
        self.alive = threading.Event()
        self.alive.set()

    @staticmethod
    def _resolve_speaker(device_name: str | None):
        if device_name:
            try:
                return sc.get_speaker(device_name)
            except Exception as exc:
                raise RuntimeError(
                    f"No output device matching {device_name!r} was found: {exc!r}"
                ) from exc
        try:
            return sc.default_speaker()
        except Exception as exc:
            raise RuntimeError(f"Could not resolve default output device: {exc!r}") from exc

    def start(self) -> None:
        """Start the background capture thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="audio-capture", daemon=True
        )
        self._thread.start()
        logger.info("Started loopback capture on %r", self.device_name)

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("Stopped loopback capture on %r", self.device_name)

    def _capture_loop(self) -> None:
        """Runs on the capture thread. No DSP; push-only into ring buffer."""
        mic = sc.get_microphone(id=str(self._speaker.name), include_loopback=True)
        try:
            with mic.recorder(
                samplerate=self._sample_rate, channels=self._channels
            ) as recorder:
                while not self._stop_event.is_set():
                    # record() blocks until `block_size` frames are ready;
                    # it returns a fresh (block_size, channels) float32 array.
                    frames = recorder.record(numframes=self._block_size)
                    self._ring_buffer.write(frames.astype(np.float32, copy=False))
        except Exception:
            logger.exception("Audio capture loop terminated unexpectedly")
            self.alive.clear()
