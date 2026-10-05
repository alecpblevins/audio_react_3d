# The math behind audio_reactive_3d

This document explains the signal-processing and graphics math used
throughout the pipeline, in the order data actually flows: capture → FFT →
log-frequency binning → smoothing → vertex displacement → color.

## 1. Sampling and Nyquist

The loopback device delivers a stream of discrete samples at
`config.SAMPLE_RATE` (48,000 Hz by default). The Nyquist–Shannon sampling
theorem says a sampled signal can only unambiguously represent frequencies
up to half the sample rate — the **Nyquist frequency**, here 24,000 Hz.
Any true frequency content above that would be indistinguishable from
(“alias as”) a lower frequency after sampling, but since humans can't hear
much above ~20 kHz anyway, 48 kHz capture is comfortably sufficient and
`config.SPECTRUM_MAX_HZ = 20_000.0` stays safely under Nyquist.

`numpy.fft.rfft` of an `N`-sample real-valued window produces `N / 2 + 1`
complex bins, evenly spaced from 0 Hz to the Nyquist frequency:

```
bin_hz(k) = k * SAMPLE_RATE / FFT_SIZE,   k = 0 .. FFT_SIZE/2
```

With `FFT_SIZE = 2048` and `SAMPLE_RATE = 48_000`, each raw FFT bin covers
`48_000 / 2048 ≈ 23.4 Hz` — fine resolution at low frequencies, but far
coarser than how humans perceive pitch (see §3).

## 2. Windowed FFT and spectral leakage

The DFT (and FFT) implicitly assumes the analyzed block repeats forever —
it treats the `FFT_SIZE`-sample window as one period of a periodic signal.
If the true waveform doesn't have an exact whole number of cycles inside
that window, there's a discontinuity at the wrap-around point, which the
FFT represents as energy smeared across *many* nearby frequency bins
instead of a single sharp peak. This smearing is **spectral leakage**.

A **window function** tapers the samples to (near) zero at both edges of
the block before the FFT, so the discontinuity at the wrap point is gone
(or much smaller), trading a bit of frequency resolution for dramatically
less leakage. This project uses a **Hann window**
(`scipy.signal.windows.hann`):

```
w[n] = 0.5 * (1 - cos(2*pi*n / (N-1))),   n = 0 .. N-1
```

Applied as `samples * w` before `numpy.fft.rfft(samples * w)`.

## 3. Log-frequency binning and why it matches human hearing

Human pitch perception is roughly *logarithmic*, not linear — the
perceived difference between 100 Hz and 200 Hz (one octave) feels similar
to the difference between 1000 Hz and 2000 Hz (also one octave), even
though the second gap spans 10x more raw Hz. A linear-frequency spectrum
display wastes most of its bins on the (perceptually cramped) high end and
gives almost no resolution to the bass, where most of the perceptually
interesting structure lives.

`audio/features.py` instead builds `config.SPECTRUM_BINS` (64) **edges**
evenly spaced in `log`-space between `SPECTRUM_MIN_HZ` (20 Hz) and
`SPECTRUM_MAX_HZ` (20,000 Hz):

```
edges[i] = SPECTRUM_MIN_HZ * (SPECTRUM_MAX_HZ / SPECTRUM_MIN_HZ) ** (i / SPECTRUM_BINS)
```

Each output bin aggregates (via mean magnitude) every raw FFT bin whose
center frequency falls between `edges[i]` and `edges[i+1]`. The result is
a 64-value spectrum where equal pixel-width corresponds to equal *musical*
interval, which is why the icosphere/particle visuals read as "the bass is
on one side, the treble on the other" in a way that feels musically
natural rather than arbitrary.

Magnitudes are then converted to decibels and normalized to `0..1` against
`config.DB_FLOOR` / `config.DB_CEILING`:

```
db = 20 * log10(magnitude + epsilon)
normalized = clip((db - DB_FLOOR) / (DB_CEILING - DB_FLOOR), 0, 1)
```

The low/mid/high band energies (§ feature vector layout in `features.py`)
are computed the same way, just aggregated over the three fixed Hz ranges
instead of 64 log-spaced ones.

## 4. Attack/release smoothing (ballistics)

A raw per-frame spectrum is visually noisy — it can swing wildly between
consecutive ~33 ms analysis frames even during sustained, steady-sounding
audio, because phase and windowing effects make bin magnitudes jitter.
Smoothing each value over time with different speeds for "getting louder"
vs. "getting quieter" mirrors how analog VU meters and audio compressors
behave, and is implemented in `audio/smoother.py` as a one-pole IIR filter
with asymmetric time constants:

```
if new_value > smoothed_value:
    coefficient = attack_coefficient      # responds fast to transients
else:
    coefficient = release_coefficient     # decays slowly/gracefully

smoothed_value += coefficient * (new_value - smoothed_value)
```

The per-tick coefficient is derived from a desired time constant `tau`
(`config.ATTACK_SECONDS` ≈ 10 ms, `config.RELEASE_SECONDS` ≈ 200 ms) and
the DSP tick period `dt`:

```
coefficient = 1 - exp(-dt / tau)
```

A short attack means the visuals react almost instantly to a kick drum or
onset; a long release means they fade out smoothly afterward instead of
chattering, which is exactly the "ballistics" behavior real VU meters and
compressors use.

## 5. Vertex displacement math (icosphere mode)

The icosphere mesh is a unit sphere (every vertex position has length 1,
and — conveniently — its own normalized position *is* its outward normal,
since the mesh is centered at the origin). The vertex shader
(`render/shaders/basic.vert`) displaces each vertex outward along that
normal, scaled by the smoothed low-band energy:

```glsl
vec3 displaced_position = in_position + in_normal * (low_band * u_displacement_scale);
```

Because `in_normal` is a unit vector, this is a pure radial "inflate the
sphere" operation — bass energy makes the whole mesh bulge outward
uniformly, and it relaxes back down as the bass fades, driven directly by
the attack/release-smoothed `low_band` feature.

## 6. Color mapping (icosphere mode)

The fragment shader (`render/shaders/basic.frag`) computes a Lambertian
diffuse term against a fixed key light, adds a constant ambient floor
(`config.AMBIENT_INTENSITY`) so no face ever reads as pure black, and adds
a Fresnel-style **rim light** — a `(1 - dot(normal, view_direction))`
term — so silhouette edges stay visible even on faces angled away from the
key light:

```glsl
float diffuse = max(dot(normal, light_dir), 0.0);
float rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
float lighting = u_ambient + (1.0 - u_ambient) * diffuse + 0.35 * rim;
vec3 color = u_base_color * lighting;
color += vec3(mid_band, 0.0, high_band) * 0.6;  // treble tints the surface
```

`mid_band` and `high_band` are added on top as a color tint (mid → warmer,
high → cooler/brighter), so the sphere's hue visibly shifts with the
treble content of whatever is playing, independent of the bass-driven
geometry displacement.

## 7. Particle-field mode: per-particle spectrum sampling

The particle mode (`render/shaders/particles.vert`) needs each of the 2048
particles to read a *different*, runtime-determined spectrum bin (particle
`i` is assigned bin `i % 64`). Static uniform-buffer indexing (as the
icosphere shader uses for the four fixed low/mid/high/rms scalars) only
supports compile-time-constant indices portably across GPUs; a *dynamic*,
per-vertex index needs `texelFetch` against a texture instead. The feature
vector is therefore also uploaded as a `(128 x 1)` single-channel floating
point texture, and each particle does:

```glsl
float bin_value = texelFetch(u_features_tex, ivec2(in_bin_index, 0), 0).r;
vec3 pos = in_direction * (1.0 + bin_value * u_displacement_scale);
```

— the same "radial displacement by energy" idea as the icosphere, but
per-particle and per-bin instead of one global low-band value for the
whole mesh. Point size and color are likewise driven by that particle's
own `bin_value`, so the particle field reads as a spherical bar-spectrum
analyzer.
