"""GLSL shader hot-reload (Milestone 4).

Polls each shader file's modification time and, when it changes, attempts
to recompile the program. A failed recompile (syntax error mid-edit) is
logged and the last good program is kept -- the app never crashes from a
broken shader save.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import moderngl

logger = logging.getLogger(__name__)


class HotReloadProgram:
    """Watches a vertex+fragment shader pair and recompiles on change.

    Parameters
    ----------
    ctx:
        The moderngl context to compile new programs against.
    vertex_path, fragment_path:
        Paths to the GLSL source files to watch.
    on_reload:
        Called with the newly compiled :class:`moderngl.Program` each time
        a recompile succeeds (including the very first load), so callers
        can rebuild dependent VAOs/uniforms.
    poll_interval_seconds:
        Minimum time between ``stat()`` checks, to avoid hammering the
        filesystem every single frame.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        vertex_path: Path,
        fragment_path: Path,
        on_reload: Callable[[moderngl.Program], None],
        poll_interval_seconds: float = 0.5,
    ) -> None:
        self._ctx = ctx
        self._vertex_path = vertex_path
        self._fragment_path = fragment_path
        self._on_reload = on_reload
        self._poll_interval = poll_interval_seconds

        self._last_check_time = 0.0
        self._last_mtimes = (0.0, 0.0)
        self.program: moderngl.Program = self._compile()
        self._last_mtimes = self._current_mtimes()
        self._on_reload(self.program)

    def _current_mtimes(self) -> tuple[float, float]:
        return (self._vertex_path.stat().st_mtime, self._fragment_path.stat().st_mtime)

    def _compile(self) -> moderngl.Program:
        return self._ctx.program(
            vertex_shader=self._vertex_path.read_text(encoding="utf-8"),
            fragment_shader=self._fragment_path.read_text(encoding="utf-8"),
        )

    def poll(self, now_seconds: float) -> None:
        """Check for shader-file changes and recompile if needed.

        Args:
            now_seconds: Monotonic clock value (e.g. ``time.perf_counter()``)
                used to throttle how often the filesystem is touched.
        """
        if now_seconds - self._last_check_time < self._poll_interval:
            return
        self._last_check_time = now_seconds

        try:
            mtimes = self._current_mtimes()
        except OSError:
            return  # File briefly missing mid-save; try again next poll.

        if mtimes == self._last_mtimes:
            return
        self._last_mtimes = mtimes

        try:
            new_program = self._compile()
        except Exception:
            logger.exception(
                "Shader reload failed for %s / %s -- keeping previous program",
                self._vertex_path.name,
                self._fragment_path.name,
            )
            return

        self.program = new_program
        self._on_reload(new_program)
        logger.info(
            "Reloaded shaders: %s / %s", self._vertex_path.name, self._fragment_path.name
        )
