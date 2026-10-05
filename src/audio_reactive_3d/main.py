"""Entry point for audio_reactive_3d.

Wires together audio capture, the DSP feature pipeline, and the GLSL
renderer. By default, running the module opens the moderngl-window
visualizer; ``--headless`` keeps the Milestone 1/2 console-only workflow
(no GL context needed) for environments without a display, e.g. CI or a
quick device/DSP sanity check.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time

from audio_reactive_3d import config
from audio_reactive_3d.audio.capture import LoopbackCapture, list_loopback_devices
from audio_reactive_3d.audio.dsp_worker import DSPWorker, SharedFeatureState
from audio_reactive_3d.audio.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


def _print_devices() -> int:
    """Implements ``--list-devices``. Returns a process exit code."""
    try:
        devices = list_loopback_devices()
    except RuntimeError as exc:
        print(f"Could not list audio devices: {exc}")
        return 1

    if not devices:
        print("No output (loopback-capable) devices were found.")
        return 1

    print("Available loopback-capable output devices:")
    for device in devices:
        marker = " (default)" if device.is_default else ""
        print(f"  - {device.name}{marker}")
    return 0


def _log_features_forever(
    shared: SharedFeatureState,
    worker: DSPWorker,
    stop_event: threading.Event,
    log_every_tick: bool,
) -> None:
    """Console readout of extracted features for manual verification.

    If ``log_every_tick`` is set (``--log-features``), logs the full raw
    feature vector on every DSP tick (~``config.DSP_TARGET_HZ`` times per
    second) -- this is the Milestone 2 acceptance check. Otherwise logs a
    condensed one-line summary once per second, which is friendlier for
    everyday use.
    """
    period = 1.0 / config.DSP_TARGET_HZ if log_every_tick else 1.0
    last_frame_count = -1
    while not stop_event.is_set():
        time.sleep(period)
        frame = shared.latest
        if frame is None:
            logger.info("No features yet (buffering audio)...")
            continue

        if log_every_tick:
            if worker.frames_processed == last_frame_count:
                continue  # avoid re-logging a stale frame while waiting
            last_frame_count = worker.frames_processed
            logger.info("features=%s", frame.vector.tolist())
        else:
            logger.info(
                "low=%.3f mid=%.3f high=%.3f rms=%.3f onset=%d bpm=%.1f "
                "(processed=%d skipped=%d)",
                frame.low_band,
                frame.mid_band,
                frame.high_band,
                frame.rms,
                frame.onset,
                frame.bpm,
                worker.frames_processed,
                worker.frames_skipped,
            )


def _start_capture_and_dsp(
    device_name: str | None,
) -> tuple[LoopbackCapture, DSPWorker, SharedFeatureState] | None:
    """Create and start the ring buffer, capture, and DSP worker.

    Returns ``None`` (after logging an error) if audio capture could not be
    initialized, so callers can degrade gracefully instead of crashing.
    """
    ring_buffer = RingBuffer(capacity=config.RING_BUFFER_FRAMES, channels=config.CHANNELS)

    try:
        capture = LoopbackCapture(ring_buffer, device_name=device_name)
    except RuntimeError as exc:
        logger.error("Failed to initialize audio capture: %s", exc)
        return None

    shared = SharedFeatureState()
    dsp_worker = DSPWorker(ring_buffer, shared)

    capture.start()
    dsp_worker.start()
    return capture, dsp_worker, shared


def _run_headless(device_name: str | None, log_every_tick: bool) -> int:
    """Capture loopback audio, run the DSP pipeline, and log features.

    No GL context or window is created -- useful for devices/sessions
    without a display, or for quickly sanity-checking audio/DSP in
    isolation from rendering.
    """
    started = _start_capture_and_dsp(device_name)
    if started is None:
        return 1
    capture, dsp_worker, shared = started

    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received signal %s, shutting down...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        signal.signal(signal.SIGTERM, _handle_signal)
    except (AttributeError, ValueError):
        pass  # SIGTERM not available on this platform.

    logger.info("Capturing from %r. Press Ctrl+C to stop.", capture.device_name)
    try:
        _log_features_forever(shared, dsp_worker, stop_event, log_every_tick)
    finally:
        dsp_worker.stop()
        capture.stop()

    return 0


def _run_render(device_name: str | None) -> int:
    """Capture loopback audio, run the DSP pipeline, and open the visualizer.

    Degrades gracefully (logs an error, returns a nonzero exit code) if
    either audio capture or GL window/context creation fails, per the
    project's "must not hard-crash" requirement -- this matters most on
    Linux/macOS or headless hosts where WASAPI-style loopback or a GPU
    context may be unavailable.
    """
    import moderngl_window as mglw

    from audio_reactive_3d.render.app import build_visualizer_app

    started = _start_capture_and_dsp(device_name)
    if started is None:
        return 1
    capture, dsp_worker, shared = started

    try:
        app_cls = build_visualizer_app(shared, capture.device_name, capture_alive=capture.alive)
        logger.info("Opening visualizer window for device %r. Close it to stop.", capture.device_name)
        # args=[] bypasses moderngl_window's own argv parsing (window
        # backend/size/fullscreen flags) since our CLI already owns argv.
        mglw.run_window_config(app_cls, args=[])
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
    except Exception as exc:  # noqa: BLE001 - must not hard-crash on GL failures
        logger.error("Visualizer failed to start or crashed: %s", exc)
        return 1
    finally:
        dsp_worker.stop()
        capture.stop()

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="audio_reactive_3d",
        description="Real-time system-audio visualizer.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List loopback-capable output devices and exit.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Substring of the output device name to capture (default: system default).",
    )
    parser.add_argument(
        "--log-features",
        action="store_true",
        help=(
            "In --headless mode, log the full raw feature vector on every DSP "
            f"tick (~{config.DSP_TARGET_HZ:.0f} Hz) instead of a 1 Hz summary."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run audio capture + DSP only, logging features to the console "
            "instead of opening the GLSL visualizer window (no GPU/display "
            "required)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    _configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        return _print_devices()

    if args.headless:
        return _run_headless(args.device, args.log_features)

    return _run_render(args.device)


if __name__ == "__main__":
    raise SystemExit(main())

