"""The Milestone 3 scene: a feature-reactive icosphere.

Owns the mesh buffers, the compiled shader program, and the per-frame
uniform buffer upload of the live audio feature vector. Pure rendering
state -- it has no knowledge of audio capture, threads, or the window
system; ``render/app.py`` wires those together and calls :meth:`Scene.render`
once per frame with the latest feature vector.
"""

from __future__ import annotations

import glm
import moderngl
import numpy as np

from audio_reactive_3d import config
from audio_reactive_3d.render.geometry import generate_icosphere

#: Index into the UBO's ``vec4 data[32]`` array holding
#: ``(low_band, mid_band, high_band, rms)`` -- feature indices 64..67 fall
#: exactly on a vec4 boundary (64 / 4 == 16), so no shader-side bit-twiddling
#: is needed to unpack them.
_FEATURES_UBO_BINDING = 0


class Scene:
    """Icosphere mesh + shader program reacting to the live feature vector."""

    def __init__(self, ctx: moderngl.Context, program: moderngl.Program) -> None:
        self._ctx = ctx
        self._program = program

        vertices, normals, indices = generate_icosphere(config.ICOSPHERE_SUBDIVISIONS)
        interleaved = np.hstack([vertices, normals]).astype(np.float32)

        self._vbo = ctx.buffer(interleaved.tobytes())
        self._ibo = ctx.buffer(indices.astype(np.uint32).tobytes())
        self._vao = ctx.vertex_array(
            program,
            [(self._vbo, "3f 3f", "in_position", "in_normal")],
            self._ibo,
        )
        self.triangle_count = indices.shape[0]
        self.vertex_count = vertices.shape[0]

        # Feature vector UBO. std140 array-of-vec4 has a 16-byte stride with
        # no padding, which is exactly a contiguous float32[128] buffer --
        # the raw feature vector bytes can be written straight through.
        self._feature_ubo = ctx.buffer(reserve=config.FEATURE_VECTOR_LENGTH * 4)
        self._program["Features"].binding = _FEATURES_UBO_BINDING
        self._feature_ubo.bind_to_uniform_block(_FEATURES_UBO_BINDING)

        # Static lighting/camera uniforms that don't change per frame.
        self._program["u_base_color"].value = tuple(config.BASE_COLOR)
        self._program["u_ambient"].value = config.AMBIENT_INTENSITY
        self._program["u_camera_pos"].value = tuple(config.CAMERA_POSITION)

        ctx.enable(moderngl.DEPTH_TEST)

    def render(self, time_s: float, aspect_ratio: float, feature_vector: np.ndarray) -> None:
        """Upload the latest feature vector and draw one frame.

        Args:
            time_s: Seconds since the render loop started (drives idle
                auto-rotation).
            aspect_ratio: Current framebuffer width / height.
            feature_vector: float32 array of length
                ``config.FEATURE_VECTOR_LENGTH``.
        """
        self._feature_ubo.write(np.ascontiguousarray(feature_vector, dtype=np.float32).tobytes())

        model = glm.rotate(
            glm.mat4(1.0),
            time_s * config.ROTATION_SPEED_RAD_PER_SEC,
            glm.vec3(0.0, 1.0, 0.0),
        )
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
        mvp = projection * view * model

        self._program["u_model"].write(bytes(model))
        self._program["u_mvp"].write(bytes(mvp))
        self._program["u_displacement_scale"].value = config.DISPLACEMENT_SCALE

        self._vao.render(mode=moderngl.TRIANGLES)
