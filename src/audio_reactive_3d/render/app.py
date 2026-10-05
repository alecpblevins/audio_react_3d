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
import threading

import moderngl_window as mglw
import numpy as np
from moderngl_window.text.bitmapped import TextWriter2D

from audio_reactive_3d import config
from audio_reactive_3d.audio.dsp_worker import SharedFeatureState
from audio_reactive_3d.render.particles import ParticleScene
from audio_reactive_3d.render.scene import Scene
from audio_reactive_3d.render.shader_reload import HotReloadProgram

logger = logging.getLogger(__name__)

#: Fallback feature vector used before the DSP worker has produced its first
#: frame, so the shader always receives a well-formed (silent) buffer.
_SILENT_FEATURES = np.zeros(config.FEATURE_VECTOR_LENGTH, dtype=np.float32)

#: Rolling window (seconds) over which the on-screen FPS counter is averaged.
_FPS_AVERAGING_WINDOW_SECONDS = 0.5

_MODE_NAMES = {1: "Icosphere", 2: "Particles"}


def build_visualizer_app(
    shared: SharedFeatureState,
    device_label: str,
    capture_alive: threading.Event | None = None,
    low_power: bool = False,
) -> type[mglw.WindowConfig]:
    """Build a ``WindowConfig`` subclass bound to one capture session.

    Args:
        shared: The :class:`SharedFeatureState` the DSP worker publishes
            into; read (never written) once per rendered frame.
        device_label: Human-readable device name, shown in the on-screen
            overlay for context.
        capture_alive: Optional event that the audio capture thread clears
            if its stream dies unexpectedly (e.g. device disconnected).
            Polled once per frame so the window closes cleanly instead of
            freezing on stale audio when the device goes away.
        low_power: If set (``--low-power`` on the CLI), use a smaller
            window, no MSAA, a much lower-poly icosphere, and fewer
            particles -- a workaround for machines with no dedicated GPU,
            where the default settings can render too slowly to look
            animated (see ``config.py``'s "low-power / CPU" section).
    """

    window_size_value = config.LOW_POWER_WINDOW_SIZE if low_power else config.WINDOW_SIZE
    msaa_samples_value = config.LOW_POWER_MSAA_SAMPLES if low_power else config.MSAA_SAMPLES
    icosphere_subdivisions = (
        config.LOW_POWER_ICOSPHERE_SUBDIVISIONS if low_power else config.ICOSPHERE_SUBDIVISIONS
    )
    particle_count = config.LOW_POWER_PARTICLE_COUNT if low_power else config.PARTICLE_COUNT

    #: Substrings of ``GL_RENDERER`` that indicate a software/CPU rasterizer
    #: rather than a real GPU, used only to suggest ``--low-power`` to the
    #: user -- never to silently change already-fixed-at-creation settings
    #: like window size or MSAA samples.
    _SOFTWARE_RENDERER_HINTS = (
        "llvmpipe",
        "softpipe",
        "swiftshader",
        "basic render",
        "microsoft basic",
        "warp",
    )

    class VisualizerApp(mglw.WindowConfig):
        gl_version = (3, 3)
        title = "audio_reactive_3d"
        window_size = window_size_value
        aspect_ratio = window_size_value[0] / window_size_value[1]
        resource_dir = config.SHADERS_DIR
        vsync = config.VSYNC
        resizable = True
        samples = msaa_samples_value
        clear_color = config.CLEAR_COLOR

        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)

            renderer = str(self.ctx.info.get("GL_RENDERER", ""))
            if not low_power and any(
                hint in renderer.lower() for hint in _SOFTWARE_RENDERER_HINTS
            ):
                logger.warning(
                    "Detected a software/CPU renderer (%r). If the icosphere mode "
                    "(key 1) looks frozen or very slow, rerun with --low-power.",
                    renderer,
                )

            self._icosphere_reload = HotReloadProgram(
                self.ctx,
                config.SHADERS_DIR / "basic.vert",
                config.SHADERS_DIR / "basic.frag",
                on_reload=self._on_icosphere_reload,
                poll_interval_seconds=config.SHADER_RELOAD_POLL_SECONDS,
            )
            self._particle_reload = HotReloadProgram(
                self.ctx,
                config.SHADERS_DIR / "particles.vert",
                config.SHADERS_DIR / "particles.frag",
                on_reload=self._on_particle_reload,
                poll_interval_seconds=config.SHADER_RELOAD_POLL_SECONDS,
            )
            self._mode = 1
            self._text_writer = TextWriter2D()

            self._fps_accum_time = 0.0
            self._fps_accum_frames = 0
            self._fps_display = 0.0

            logger.info(
                "Visualizer ready: %d verts / %d tris, %d particles, device=%r, "
                "low_power=%s, renderer=%r",
                self._scene.vertex_count,
                self._scene.triangle_count,
                self._particles.particle_count,
                device_label,
                low_power,
                renderer,
            )

        def _on_icosphere_reload(self, program: object) -> None:
            if hasattr(self, "_scene"):
                self._scene.set_program(program)
            else:
                self._scene = Scene(self.ctx, program, subdivisions=icosphere_subdivisions)

        def _on_particle_reload(self, program: object) -> None:
            if hasattr(self, "_particles"):
                self._particles.set_program(program)
            else:
                self._particles = ParticleScene(self.ctx, program, particle_count=particle_count)

        def on_render(self, time_s: float, frame_time: float) -> None:
            if capture_alive is not None and not capture_alive.is_set():
                logger.error("Audio device disconnected or capture failed; closing window.")
                self.wnd.close()
                return

            self._icosphere_reload.poll(time_s)
            self._particle_reload.poll(time_s)

            frame = shared.latest
            vector = frame.vector if frame is not None else _SILENT_FEATURES

            if self._mode == 1:
                self._scene.render(time_s, self.wnd.aspect_ratio, vector)
            else:
                self._particles.render(self.wnd.aspect_ratio, vector)

            self._draw_overlay(frame_time)

        def _draw_overlay(self, frame_time: float) -> None:
            self._fps_accum_time += frame_time
            self._fps_accum_frames += 1
            if self._fps_accum_time >= _FPS_AVERAGING_WINDOW_SECONDS:
                self._fps_display = self._fps_accum_frames / self._fps_accum_time
                self._fps_accum_time = 0.0
                self._fps_accum_frames = 0

            mode_name = _MODE_NAMES[self._mode]
            self._text_writer.text = (
                f"FPS: {self._fps_display:5.1f}   Mode: {self._mode} ({mode_name})   "
                f"{device_label}"
            )
            self._text_writer.draw((10, self.wnd.height - 30), size=20.0)

        def on_key_event(self, key: object, action: object, modifiers: object) -> None:
            keys = self.wnd.keys
            if action != keys.ACTION_PRESS:
                return
            if key == keys.ESCAPE:
                self.wnd.close()
            elif key == keys.NUMBER_1:
                self._mode = 1
            elif key == keys.NUMBER_2:
                self._mode = 2

        def on_close(self) -> None:
            logger.info("Visualizer window closing.")

    return VisualizerApp

