"""The Milestone 4 second visual mode: a reactive spectrum particle field.

Particles are spread evenly over a sphere (Fibonacci lattice) and assigned
round-robin to the 64 spectrum bins. Each particle is pushed outward from
its "home" position by its own bin's live energy and colored by the same
value, so the whole field reads as a spherical spectrum analyzer. Unlike
the icosphere scene, per-particle bin lookups need dynamic (runtime)
indexing, which is done via ``texelFetch`` on a feature-vector texture
rather than the icosphere's static uniform-buffer indices.
"""

from __future__ import annotations

import glm
import moderngl
import numpy as np

from audio_reactive_3d import config
from audio_reactive_3d.render.geometry import generate_fibonacci_sphere

#: Texture unit used for the feature-vector texture.
_FEATURES_TEXTURE_UNIT = 0


class ParticleScene:
    """Owns the particle point cloud, shader program, and feature texture."""

    def __init__(
        self,
        ctx: moderngl.Context,
        program: moderngl.Program,
        particle_count: int = config.PARTICLE_COUNT,
    ) -> None:
        self._ctx = ctx
        self.set_program(program)

        directions = generate_fibonacci_sphere(particle_count)
        bin_indices = (np.arange(particle_count) % config.SPECTRUM_BINS).astype(np.float32)
        interleaved = np.hstack([directions, bin_indices.reshape(-1, 1)]).astype(np.float32)
        self._vbo = ctx.buffer(interleaved.tobytes())
        self.particle_count = particle_count

        # Feature vector as a (128 x 1) single-channel float texture so the
        # vertex shader can texelFetch an arbitrary, per-particle bin index
        # at runtime (dynamic indexing isn't available for UBO arrays on
        # all GL 3.3 hardware, but texelFetch always supports it).
        self._feature_tex = ctx.texture(
            (config.FEATURE_VECTOR_LENGTH, 1), components=1, dtype="f4"
        )
        self._feature_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self._rebuild_vao()

    def set_program(self, program: moderngl.Program) -> None:
        """Swap in a (re)compiled shader program, e.g. after a hot-reload."""
        self._program = program
        if hasattr(self, "_vbo"):
            self._rebuild_vao()

    def _rebuild_vao(self) -> None:
        self._vao = self._ctx.vertex_array(
            self._program,
            [(self._vbo, "3f 1f", "in_direction", "in_bin_index")],
        )

    def render(self, aspect_ratio: float, feature_vector: np.ndarray) -> None:
        """Upload the latest feature vector and draw one frame of particles.

        Args:
            aspect_ratio: Current framebuffer width / height.
            feature_vector: float32 array of length
                ``config.FEATURE_VECTOR_LENGTH``.
        """
        self._feature_tex.write(
            np.ascontiguousarray(feature_vector, dtype=np.float32).tobytes()
        )
        self._feature_tex.use(location=_FEATURES_TEXTURE_UNIT)

        view = glm.lookAt(
            glm.vec3(*config.CAMERA_POSITION),
            glm.vec3(0.0, 0.0, 0.0),
            glm.vec3(0.0, 1.0, 0.0),
        )
        projection = glm.perspective(
            glm.radians(config.CAMERA_FOV_DEGREES),
            aspect_ratio,
            config.CAMERA_NEAR,
            config.CAMERA_FAR,
        )
        mvp = projection * view

        self._program["u_mvp"].write(bytes(mvp))
        self._program["u_features_tex"].value = _FEATURES_TEXTURE_UNIT
        self._program["u_displacement_scale"].value = config.PARTICLE_DISPLACEMENT_SCALE
        self._program["u_point_base_size"].value = config.PARTICLE_POINT_BASE_SIZE
        self._program["u_point_size_scale"].value = config.PARTICLE_POINT_SIZE_SCALE
        self._program["u_point_rms_scale"].value = config.PARTICLE_POINT_RMS_SCALE

        self._ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self._ctx.enable(moderngl.BLEND)
        self._vao.render(mode=moderngl.POINTS)
