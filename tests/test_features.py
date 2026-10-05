"""Tests for FFT-based feature extraction (spectrum, bands, RMS)."""

from __future__ import annotations

import numpy as np

from audio_reactive_3d import config
from audio_reactive_3d.audio.features import RMS_INDEX, FeatureExtractor


def _sine_wave(freq_hz: float, n_samples: int, sample_rate: int, amplitude: float = 0.8) -> np.ndarray:
    t = np.arange(n_samples) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_sine_peak_in_expected_log_bin() -> None:
    extractor = FeatureExtractor(sample_rate=config.SAMPLE_RATE, fft_size=config.FFT_SIZE)
    tone = _sine_wave(440.0, config.FFT_SIZE, config.SAMPLE_RATE)

    vector = extractor.process(tone, timestamp=0.0)
    spectrum = vector[: config.SPECTRUM_BINS]

    expected_bin = extractor.bin_for_frequency(440.0)
    assert expected_bin != -1

    peak_bin = int(np.argmax(spectrum))
    # Log bins near a few hundred Hz are only one or two linear FFT bins
    # wide, so windowing/leakage can nudge the measured peak by a couple of
    # bins relative to the raw edge-based lookup; allow a small tolerance.
    assert abs(peak_bin - expected_bin) <= 2


def test_silence_has_near_zero_rms() -> None:
    extractor = FeatureExtractor(sample_rate=config.SAMPLE_RATE, fft_size=config.FFT_SIZE)
    silence = np.zeros(config.FFT_SIZE, dtype=np.float32)

    vector = extractor.process(silence, timestamp=0.0)

    assert vector[RMS_INDEX] < 1e-3


def test_stereo_input_is_downmixed() -> None:
    extractor = FeatureExtractor(sample_rate=config.SAMPLE_RATE, fft_size=config.FFT_SIZE)
    mono_tone = _sine_wave(1000.0, config.FFT_SIZE, config.SAMPLE_RATE)
    stereo_tone = np.stack([mono_tone, mono_tone], axis=1)

    mono_vector = extractor.process(mono_tone, timestamp=0.0)
    stereo_vector = extractor.process(stereo_tone, timestamp=1.0 / config.DSP_TARGET_HZ)

    # Same signal on both channels should downmix to (numerically) the same
    # spectrum shape as mono.
    np.testing.assert_allclose(
        mono_vector[: config.SPECTRUM_BINS], stereo_vector[: config.SPECTRUM_BINS], atol=1e-5
    )


def test_feature_vector_has_expected_length_and_padding() -> None:
    extractor = FeatureExtractor(sample_rate=config.SAMPLE_RATE, fft_size=config.FFT_SIZE)
    tone = _sine_wave(220.0, config.FFT_SIZE, config.SAMPLE_RATE)

    vector = extractor.process(tone, timestamp=0.0)

    assert vector.shape == (config.FEATURE_VECTOR_LENGTH,)
    assert vector.dtype == np.float32
    # Reserved padding region (after BPM) must stay zero.
    np.testing.assert_array_equal(vector[70:], np.zeros(config.FEATURE_VECTOR_LENGTH - 70))


def test_loud_transient_triggers_onset_after_quiet_passage() -> None:
    extractor = FeatureExtractor(sample_rate=config.SAMPLE_RATE, fft_size=config.FFT_SIZE)
    silence = np.zeros(config.FFT_SIZE, dtype=np.float32)
    loud = _sine_wave(440.0, config.FFT_SIZE, config.SAMPLE_RATE, amplitude=0.9)

    onset_detected = False
    t = 0.0
    dt = config.FFT_SIZE / config.SAMPLE_RATE
    # Feed enough quiet frames to build up onset-detection history, with a
    # loud transient appearing after it (well past the refractory period).
    for i in range(20):
        frame = loud if i == 15 else silence
        vector = extractor.process(frame, timestamp=t)
        if vector[config.SPECTRUM_BINS + 4] > 0.5:
            onset_detected = True
        t += dt

    assert onset_detected
