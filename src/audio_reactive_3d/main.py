"""Entry point for audio_reactive_3d.

Wires together audio capture, (eventually) DSP, and (eventually) rendering.
Milestone 1 only implements loopback capture plus a console RMS readout and
a ``--list-devices`` CLI flag; later milestones hang off the same entry
point without changing this module's public shape.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading

from audio_reactive_3d import config
from audio_reactive_3d.audio.capture import LoopbackCapture, list_loopback_devices, log_rms_forever
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


def _run_capture(device_name: str | None) -> int:
    """Milestone 1 behavior: capture loopback audio and log RMS once/sec."""
    ring_buffer = RingBuffer(capacity=config.RING_BUFFER_FRAMES, channels=config.CHANNELS)

    try:
        capture = LoopbackCapture(ring_buffer, device_name=device_name)
    except RuntimeError as exc:
        logger.error("Failed to initialize audio capture: %s", exc)
        return 1

    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received signal %s, shutting down...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        signal.signal(signal.SIGTERM, _handle_signal)
    except (AttributeError, ValueError):
        pass  # SIGTERM not available on this platform.

    capture.start()
    logger.info("Capturing from %r. Press Ctrl+C to stop.", capture.device_name)
    try:
        log_rms_forever(ring_buffer, stop_event=stop_event)
    finally:
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    _configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        return _print_devices()

    return _run_capture(args.device)


if __name__ == "__main__":
    raise SystemExit(main())
