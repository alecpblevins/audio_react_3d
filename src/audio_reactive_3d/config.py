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

#: Frequency range covered by the 64 log-spaced spectrum bins.
SPECTRUM_MIN_HZ: float = 20.0
SPECTRUM_MAX_HZ: float = 20_000.0

#: dB range used to normalize magnitude/RMS values to 0..1. 0 dBFS is full
#: scale for a float32 signal; -80 dB is a practical noise floor.
DB_FLOOR: float = -80.0
DB_CEILING: float = 0.0

#: Attack/release smoothing time constants, in seconds, applied to the
#: spectrum + band energies + RMS.
ATTACK_SECONDS: float = 0.010
RELEASE_SECONDS: float = 0.200

#: Separate (slower) ballistics for the BPM estimate, which should not
#: visibly flicker between beats.
BPM_ATTACK_SECONDS: float = 0.5
BPM_RELEASE_SECONDS: float = 2.0

#: Onset (spectral flux) detection tuning.
#: Number of recent frames kept to compute the adaptive flux threshold.
ONSET_HISTORY_FRAMES: int = 43  # ~1.4s of history at 30 Hz
#: Flux must exceed mean + K * stddev of recent history to count as onset.
ONSET_THRESHOLD_K: float = 1.5
#: Minimum time between two reported onsets, to avoid double-triggering.
ONSET_REFRACTORY_SECONDS: float = 0.1

#: Plausible BPM range for tempo estimation.
BPM_MIN: float = 60.0
BPM_MAX: float = 200.0

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

#: Number of icosahedron subdivision passes. Each pass quadruples the
#: triangle count; level 4 yields 2562 vertices / 5120 triangles, matching
#: the spec's "~2k verts, no external assets" target.
ICOSPHERE_SUBDIVISIONS: int = 4

#: How far vertices are pushed outward along their normal, scaled by the
#: (smoothed, 0..1) low-band energy. In mesh-radius units (mesh is a unit
#: sphere), so 0.35 means the surface can bulge up to 35% of its radius.
DISPLACEMENT_SCALE: float = 0.35

#: Camera / projection setup for the Milestone 3 icosphere scene.
CAMERA_POSITION: tuple[float, float, float] = (0.0, 0.0, 3.0)
CAMERA_FOV_DEGREES: float = 45.0
CAMERA_NEAR: float = 0.1
CAMERA_FAR: float = 100.0

#: Unlit base surface color (r, g, b), bright enough to stay clearly visible
#: even at silence; band energies additively brighten it on top.
BASE_COLOR: tuple[float, float, float] = (0.12, 0.14, 0.26)

#: Minimum lighting factor applied even on faces pointing away from the key
#: light, so the mesh never reads as a black silhouette.
AMBIENT_INTENSITY: float = 0.45

# --------------------------------------------------------------------------
# Rendering: particle-field visual mode (Milestone 4)
# --------------------------------------------------------------------------

#: Number of particles in the spectrum particle field, spread evenly over a
#: sphere (Fibonacci lattice) and assigned round-robin to spectrum bins so
#: every bin is represented by roughly ``PARTICLE_COUNT / SPECTRUM_BINS``
#: particles.
PARTICLE_COUNT: int = 2048

#: How far each particle is pushed outward from its "home" sphere position,
#: scaled by its assigned spectrum bin's normalized (0..1) value.
PARTICLE_DISPLACEMENT_SCALE: float = 0.9

#: Point sprite size range (pixels): rendered size is
#: ``base + bin_value * scale + rms * rms_scale``.
PARTICLE_POINT_BASE_SIZE: float = 3.0
PARTICLE_POINT_SIZE_SCALE: float = 10.0
PARTICLE_POINT_RMS_SCALE: float = 6.0

#: Hot-reload: how often (seconds) to stat() shader files for changes.
SHADER_RELOAD_POLL_SECONDS: float = 0.5

#: Slow idle auto-rotation so the mesh reads as 3D even with a static
#: camera (radians/second about the Y axis).
ROTATION_SPEED_RAD_PER_SEC: float = 0.3

#: Background clear color (r, g, b, a).
CLEAR_COLOR: tuple[float, float, float, float] = (0.02, 0.02, 0.05, 1.0)

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL: str = "INFO"
