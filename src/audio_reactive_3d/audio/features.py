"""FFT-based feature extraction: spectrum, band energies, onset, RMS, BPM.

Feature vector layout (index -> meaning), total length
``config.FEATURE_VECTOR_LENGTH`` (128 floats, padded for GPU uniform-buffer
alignment)::

    [0:64]   log-spaced magnitude spectrum, dB-scaled, normalized to 0..1
    [64]     low band energy  (20-250 Hz),     normalized 0..1
    [65]     mid band energy  (250-2000 Hz),   normalized 0..1
    [66]     high band energy (2000-16000 Hz), normalized 0..1
    [67]     RMS level, normalized 0..1 (dB-scaled)
    [68]     onset flag: 1.0 on a detected onset frame, else 0.0
    [69]     smoothed BPM estimate (0.0 if not yet known)
    [70:128] reserved, always 0.0

This module is pure signal processing: it has no knowledge of threads,
ring buffers, or rendering. See ``audio/dsp_worker.py`` for the background
thread that drives this on live audio, and ``audio/smoother.py`` for the
ballistics applied to the raw values this module produces.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import windows

from audio_reactive_3d import config

_EPS = 1e-10

#: Index of the first non-spectrum element in the feature vector.
BAND_LOW_INDEX = config.SPECTRUM_BINS
BAND_MID_INDEX = config.SPECTRUM_BINS + 1
BAND_HIGH_INDEX = config.SPECTRUM_BINS + 2
RMS_INDEX = config.SPECTRUM_BINS + 3
ONSET_INDEX = config.SPECTRUM_BINS + 4
BPM_INDEX = config.SPECTRUM_BINS + 5


def _db_normalize(magnitude: np.ndarray, floor_db: float, ceiling_db: float) -> np.ndarray:
    """Convert linear magnitude to dB and normalize to 0..1, clipped."""
    db = 20.0 * np.log10(magnitude + _EPS)
    normalized = (db - floor_db) / (ceiling_db - floor_db)
    return np.clip(normalized, 0.0, 1.0)


def _band_energy(power: np.ndarray, mask: np.ndarray) -> float:
    """Mean power within ``mask``, converted to a normalized dB-scale RMS."""
    if not mask.any():
        return 0.0
    band_rms = float(np.sqrt(np.mean(power[mask]) + _EPS))
    return float(_db_normalize(np.array([band_rms]), config.DB_FLOOR, config.DB_CEILING)[0])


@dataclass(frozen=True)
class FeatureFrame:
    """An immutable snapshot of one extracted (and possibly smoothed) frame."""

    vector: np.ndarray  # shape (config.FEATURE_VECTOR_LENGTH,), float32
    timestamp: float

    @property
    def spectrum(self) -> np.ndarray:
        return self.vector[: config.SPECTRUM_BINS]

    @property
    def low_band(self) -> float:
        return float(self.vector[BAND_LOW_INDEX])

    @property
    def mid_band(self) -> float:
        return float(self.vector[BAND_MID_INDEX])

    @property
    def high_band(self) -> float:
        return float(self.vector[BAND_HIGH_INDEX])

    @property
    def rms(self) -> float:
        return float(self.vector[RMS_INDEX])

    @property
    def onset(self) -> bool:
        return bool(self.vector[ONSET_INDEX])

    @property
    def bpm(self) -> float:
        return float(self.vector[BPM_INDEX])


class FeatureExtractor:
    """Stateful FFT feature extractor.

    Not thread-safe: holds per-frame state (previous spectrum, onset
    history) and is intended to be owned and called exclusively by a
    single DSP worker thread/loop.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        fft_size: int = config.FFT_SIZE,
    ) -> None:
        self._sample_rate = sample_rate
        self._fft_size = fft_size
        self._window = windows.hann(fft_size, sym=False).astype(np.float32)

        freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
        self._freqs = freqs

        spectrum_max_hz = min(config.SPECTRUM_MAX_HZ, sample_rate / 2.0)
        self._bin_edges = np.geomspace(
            config.SPECTRUM_MIN_HZ, spectrum_max_hz, config.SPECTRUM_BINS + 1
        )
        bin_indices = np.digitize(freqs, self._bin_edges) - 1
        bin_indices[(bin_indices < 0) | (bin_indices >= config.SPECTRUM_BINS)] = -1
        self._bin_indices = bin_indices

        self._low_mask = (freqs >= config.LOW_BAND_HZ[0]) & (freqs < config.LOW_BAND_HZ[1])
        self._mid_mask = (freqs >= config.MID_BAND_HZ[0]) & (freqs < config.MID_BAND_HZ[1])
        self._high_mask = (freqs >= config.HIGH_BAND_HZ[0]) & (freqs < config.HIGH_BAND_HZ[1])

        self._prev_magnitude: np.ndarray | None = None
        self._flux_history: deque[float] = deque(maxlen=config.ONSET_HISTORY_FRAMES)
        self._onset_times: deque[float] = deque(maxlen=8)
        self._last_onset_time: float = -1e9

    def bin_for_frequency(self, freq_hz: float) -> int:
        """Return the log-spectrum bin index that ``freq_hz`` falls into.

        Returns -1 if ``freq_hz`` is outside the covered range
        (``config.SPECTRUM_MIN_HZ`` .. ``config.SPECTRUM_MAX_HZ``).
        """
        idx = int(np.digitize([freq_hz], self._bin_edges)[0]) - 1
        if idx < 0 or idx >= config.SPECTRUM_BINS:
            return -1
        return idx

    def process(self, samples: np.ndarray, timestamp: float) -> np.ndarray:
        """Compute one raw (unsmoothed) feature vector from an audio block.

        ``samples`` must have exactly ``fft_size`` frames, shape
        ``(fft_size,)`` (mono) or ``(fft_size, channels)`` (downmixed to
        mono by averaging channels). ``timestamp`` should be a monotonic
        clock reading in seconds, used for onset refractory timing and BPM
        estimation.

        Returns a freshly allocated float32 array of length
        ``config.FEATURE_VECTOR_LENGTH``.
        """
        if samples.shape[0] != self._fft_size:
            raise ValueError(f"expected {self._fft_size} samples, got {samples.shape[0]}")

        mono = samples.mean(axis=1) if samples.ndim == 2 else samples
        windowed = mono.astype(np.float32) * self._window

        magnitude = np.abs(np.fft.rfft(windowed))
        power = magnitude * magnitude

        spectrum = np.zeros(config.SPECTRUM_BINS, dtype=np.float32)
        counts = np.zeros(config.SPECTRUM_BINS, dtype=np.int32)
        valid = self._bin_indices >= 0
        np.add.at(spectrum, self._bin_indices[valid], magnitude[valid])
        np.add.at(counts, self._bin_indices[valid], 1)
        nonzero = counts > 0
        spectrum[nonzero] /= counts[nonzero]
        spectrum = _db_normalize(spectrum, config.DB_FLOOR, config.DB_CEILING)

        low = _band_energy(power, self._low_mask)
        mid = _band_energy(power, self._mid_mask)
        high = _band_energy(power, self._high_mask)

        rms = float(np.sqrt(np.mean(np.square(mono)) + _EPS))
        rms_norm = float(_db_normalize(np.array([rms]), config.DB_FLOOR, config.DB_CEILING)[0])

        onset = self._detect_onset(magnitude, timestamp)
        bpm = self._estimate_bpm()

        vector = np.zeros(config.FEATURE_VECTOR_LENGTH, dtype=np.float32)
        vector[: config.SPECTRUM_BINS] = spectrum
        vector[BAND_LOW_INDEX] = low
        vector[BAND_MID_INDEX] = mid
        vector[BAND_HIGH_INDEX] = high
        vector[RMS_INDEX] = rms_norm
        vector[ONSET_INDEX] = 1.0 if onset else 0.0
        vector[BPM_INDEX] = bpm
        return vector

    def _detect_onset(self, magnitude: np.ndarray, timestamp: float) -> bool:
        """Spectral-flux onset detection with an adaptive threshold.

        Flux is the sum of positive (rising) bin-to-bin magnitude
        differences vs. the previous frame. An onset fires when flux
        exceeds ``mean + K * stddev`` of recent flux history and the
        refractory period has elapsed, avoiding double-triggers.
        """
        if self._prev_magnitude is None:
            self._prev_magnitude = magnitude
            return False

        diff = magnitude - self._prev_magnitude
        flux = float(np.sum(diff[diff > 0]))
        self._prev_magnitude = magnitude
        self._flux_history.append(flux)

        min_history = max(8, self._flux_history.maxlen // 4)
        if len(self._flux_history) < min_history:
            return False

        history = np.array(self._flux_history)
        threshold = float(history.mean() + config.ONSET_THRESHOLD_K * history.std())

        is_candidate = flux > threshold and flux > 0.0
        refractory_ok = (timestamp - self._last_onset_time) >= config.ONSET_REFRACTORY_SECONDS
        onset = is_candidate and refractory_ok
        if onset:
            self._last_onset_time = timestamp
            self._onset_times.append(timestamp)
        return onset

    def _estimate_bpm(self) -> float:
        """Median inter-onset-interval tempo estimate, 0.0 if not enough data."""
        if len(self._onset_times) < 3:
            return 0.0

        intervals = np.diff(np.array(self._onset_times))
        plausible = intervals[
            (intervals > 60.0 / config.BPM_MAX) & (intervals < 60.0 / config.BPM_MIN)
        ]
        if plausible.size == 0:
            return 0.0

        median_interval = float(np.median(plausible))
        bpm = 60.0 / median_interval
        return float(np.clip(bpm, config.BPM_MIN, config.BPM_MAX))
