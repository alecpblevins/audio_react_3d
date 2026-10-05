"""moderngl-window application wiring live audio features to the GPU scene.

Design note on dependency injection: ``moderngl_window.run_window_config``
constructs the ``WindowConfig`` itself (``config_cls(ctx=..., wnd=..., ...)``)
after creating the GL context and window, so there is no constructor hook to
pass arbitrary objects straight through. :func:`build_visualizer_app` instead
builds the ``WindowConfig`` subclass dynamically inside a closure, so each
run can bind its own :class:`SharedFeatureState` without any module-level
mutable state.
"""

from __future__ import annotations

import logging

import moderngl_window as mglw
import numpy as np
from moderngl_window.text.bitmapped import TextWriter2D

from audio_reactive_3d import config
from audio_reactive_3d.audio.dsp_worker import SharedFeatureState
from audio_reactive_3d.render.scene import Scene

logger = logging.getLogger(__name__)

#: Fallback feature vector used before the DSP worker has produced its first
#: frame, so the shader always receives a well-formed (silent) buffer.
_SILENT_FEATURES = np.zeros(config.FEATURE_VECTOR_LENGTH, dtype=np.float32)

#: Rolling window (seconds) over which the on-screen FPS counter is averaged.
_FPS_AVERAGING_WINDOW_SECONDS = 0.5


def build_visualizer_app(
    shared: SharedFeatureState, device_label: str
) -> type[mglw.WindowConfig]:
    """Build a ``WindowConfig`` subclass bound to one capture session.

    Args:
        shared: The :class:`SharedFeatureState` the DSP worker publishes
            into; read (never written) once per rendered frame.
        device_label: Human-readable device name, shown in the on-screen
            overlay for context.
    """

    class VisualizerApp(mglw.WindowConfig):
        gl_version = (3, 3)
        title = "audio_reactive_3d"
        window_size = config.WINDOW_SIZE
        aspect_ratio = config.WINDOW_SIZE[0] / config.WINDOW_SIZE[1]
        resource_dir = config.SHADERS_DIR
        vsync = config.VSYNC
        resizable = True
        samples = 4
        clear_color = config.CLEAR_COLOR

        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            program = self.load_program(vertex_shader="basic.vert", fragment_shader="basic.frag")
            self._scene = Scene(self.ctx, program)
            self._text_writer = TextWriter2D()

            self._fps_accum_time = 0.0
            self._fps_accum_frames = 0
            self._fps_display = 0.0

            logger.info(
                "Visualizer ready: %d verts / %d tris, device=%r",
                self._scene.vertex_count,
                self._scene.triangle_count,
                device_label,
            )

        def on_render(self, time_s: float, frame_time: float) -> None:
            frame = shared.latest
            vector = frame.vector if frame is not None else _SILENT_FEATURES

            self._scene.render(time_s, self.wnd.aspect_ratio, vector)
            self._draw_overlay(frame_time)

        def _draw_overlay(self, frame_time: float) -> None:
            self._fps_accum_time += frame_time
            self._fps_accum_frames += 1
            if self._fps_accum_time >= _FPS_AVERAGING_WINDOW_SECONDS:
                self._fps_display = self._fps_accum_frames / self._fps_accum_time
                self._fps_accum_time = 0.0
                self._fps_accum_frames = 0

            self._text_writer.text = f"FPS: {self._fps_display:5.1f}   {device_label}"
            self._text_writer.draw((10, self.wnd.height - 30), size=20.0)

        def on_key_event(self, key: object, action: object, modifiers: object) -> None:
            keys = self.wnd.keys
            if action == keys.ACTION_PRESS and key == keys.ESCAPE:
                self.wnd.close()

    return VisualizerApp
