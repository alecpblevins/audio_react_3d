"""audio_reactive_3d: real-time system-audio visualizer.

A single-process application that captures system audio output, analyzes it
in real time (FFT, band energies, onset detection), and renders a reactive
3D/GLSL visualization. See ``docs/math.md`` for the underlying DSP and
graphics math, and ``config.py`` for all tunable constants.
"""

__version__ = "0.1.0"
