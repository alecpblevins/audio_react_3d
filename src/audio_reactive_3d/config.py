"""Central configuration for audio_reactive_3d.

All tunable constants (device selection, FFT parameters, smoothing
ballistics, shader paths, window settings) live here so no module reaches
into another to hardcode a magic number. Nothing in this module holds
mutable runtime state -- it is pure configuration.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Audio capture
# --------------------------------------------------------------------------

#: Sample rate (Hz) requested from the loopback device.
SAMPLE_RATE: int = 48_000

#: Number of audio channels captured (loopback devices are typically stereo).
CHANNELS: int = 2

#: Number of frames pulled from the device per callback/record() call.
#: Smaller blocks reduce capture latency at the cost of more thread wakeups.
BLOCK_SIZE: int = 512

#: Capacity of the lock-free ring buffer, in frames. Large enough to absorb
#: scheduling jitter in the DSP thread without ever blocking the audio
#: capture thread.
RING_BUFFER_FRAMES: int = SAMPLE_RATE * 2  # 2 seconds of audio

#: Name substring used to select a non-default loopback device, or ``None``
#: to use the system default output device's loopback.
DEVICE_NAME: str | None = None

#: How often (seconds) the capture thread logs an RMS level to the console
#: while running standalone (``--list-devices`` is unaffected).
RMS_LOG_INTERVAL_SECONDS: float = 1.0

# --------------------------------------------------------------------------
# DSP (Milestone 2)
# --------------------------------------------------------------------------

#: FFT window size, in samples. Must be a power of two for numpy.fft.rfft
#: efficiency.
FFT_SIZE: int = 2048

#: Number of log-spaced spectrum bins exposed in the feature vector.
SPECTRUM_BINS: int = 64

#: Frequency band edges (Hz) for low/mid/high energy aggregation.
LOW_BAND_HZ: tuple[float, float] = (20.0, 250.0)
MID_BAND_HZ: tuple[float, float] = (250.0, 2_000.0)
HIGH_BAND_HZ: tuple[float, float] = (2_000.0, 16_000.0)

#: Attack/release smoothing time constants, in seconds.
ATTACK_SECONDS: float = 0.010
RELEASE_SECONDS: float = 0.200

#: Target DSP feature-extraction rate (Hz).
DSP_TARGET_HZ: float = 30.0

#: Total length (floats) of the shared feature vector, padded for GPU
#: uniform-buffer alignment. See docstring in ``audio/features.py`` for the
#: exact layout.
FEATURE_VECTOR_LENGTH: int = 128

# --------------------------------------------------------------------------
# Rendering (Milestone 3+)
# --------------------------------------------------------------------------

WINDOW_SIZE: tuple[int, int] = (1280, 720)
TARGET_FPS: int = 60
VSYNC: bool = True

SHADERS_DIR: Path = Path(__file__).resolve().parent / "render" / "shaders"

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL: str = "INFO"
