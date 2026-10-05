"""Tests for the attack/release ballistics Smoother."""

from __future__ import annotations

import numpy as np
import pytest

from audio_reactive_3d.audio.smoother import Smoother


def test_first_call_snaps_to_value() -> None:
    smoother = Smoother(attack_seconds=0.01, release_seconds=0.2)
    result = smoother.process(0.8, dt=0.0)
    assert float(result) == pytest.approx(0.8)


def test_rising_value_uses_fast_attack() -> None:
    fast = Smoother(attack_seconds=0.001, release_seconds=1.0)
    slow = Smoother(attack_seconds=1.0, release_seconds=1.0)

    fast.process(0.0, dt=0.0)
    slow.process(0.0, dt=0.0)

    fast_result = float(fast.process(1.0, dt=0.01))
    slow_result = float(slow.process(1.0, dt=0.01))

    # A much faster attack time constant should catch up to the target
    # value noticeably more than the slow one in the same dt.
    assert fast_result > slow_result


def test_falling_value_uses_release_not_attack() -> None:
    smoother = Smoother(attack_seconds=0.001, release_seconds=5.0)
    smoother.process(1.0, dt=0.0)

    result = float(smoother.process(0.0, dt=0.01))

    # With a very slow release, the value should barely have decayed.
    assert result > 0.9


def test_vector_shape_smoothing_independent_per_element() -> None:
    smoother = Smoother(attack_seconds=0.01, release_seconds=0.2, shape=(3,))
    smoother.process(np.array([0.0, 0.0, 0.0]), dt=0.0)

    result = smoother.process(np.array([1.0, 0.0, 0.5]), dt=0.05)

    assert result.shape == (3,)
    assert result[0] > result[2] > result[1]


def test_reset_clears_state() -> None:
    smoother = Smoother()
    smoother.process(1.0, dt=0.0)
    smoother.reset()

    result = smoother.process(0.3, dt=0.01)
    assert float(result) == pytest.approx(0.3)
