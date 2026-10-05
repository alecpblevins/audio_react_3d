"""Attack/release (ballistics) smoothing for audio feature values.

Implements the classic envelope-follower approach: when an incoming value
is rising, approach it using a fast "attack" time constant; when falling,
use a slower "release" time constant. This keeps visuals responsive to
transients (attack) while avoiding flicker on sustained decay (release).

This module has no knowledge of audio, threads, or the feature-vector
layout -- it is a generic, reusable exponential envelope follower.
"""

from __future__ import annotations

import numpy as np

from audio_reactive_3d import config


class Smoother:
    """Per-element attack/release smoother, usable on scalars or arrays.

    Parameters
    ----------
    attack_seconds, release_seconds:
        Time constants (seconds) for rising vs. falling values. Smaller
        values track the input faster.
    shape:
        Shape of the internal state array. Use ``()`` for a scalar or e.g.
        ``(64,)`` for a vector such as the spectrum.
    """

    def __init__(
        self,
        attack_seconds: float = config.ATTACK_SECONDS,
        release_seconds: float = config.RELEASE_SECONDS,
        shape: tuple[int, ...] = (),
    ) -> None:
        self._attack_seconds = max(attack_seconds, 1e-6)
        self._release_seconds = max(release_seconds, 1e-6)
        self._state = np.zeros(shape, dtype=np.float32)
        self._initialized = False

    def reset(self) -> None:
        """Clear state so the next ``process()`` call snaps to its input."""
        self._state[...] = 0.0
        self._initialized = False

    def process(self, value: np.ndarray | float, dt: float) -> np.ndarray:
        """Advance the smoother by ``dt`` seconds toward ``value``.

        On the very first call (or after :meth:`reset`), the state snaps
        directly to ``value`` to avoid a slow ramp-up from zero. Returns
        the updated internal state array (not a copy-free view into
        caller data -- safe to hold onto).
        """
        value_arr = np.asarray(value, dtype=np.float32)
        if not self._initialized:
            self._state = np.array(value_arr, dtype=np.float32, copy=True)
            self._initialized = True
            return self._state

        dt = max(dt, 0.0)
        rising = value_arr > self._state
        tau = np.where(rising, self._attack_seconds, self._release_seconds)
        # Exponential approach: coeff -> 0 means "fully caught up this tick".
        coeff = np.exp(-dt / tau)
        self._state = (coeff * self._state + (1.0 - coeff) * value_arr).astype(np.float32)
        return self._state
