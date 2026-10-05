"""Procedural icosphere generation (no external mesh assets).

Starts from a regular icosahedron (12 vertices, 20 triangular faces) and
repeatedly subdivides each triangle into four, projecting new vertices onto
the unit sphere each pass. Since the mesh is centered at the origin with
radius 1, a vertex's outward normal is simply its (normalized) position --
no separate normal computation is needed.
"""

from __future__ import annotations

import numpy as np

_GOLDEN_RATIO = (1.0 + 5.0**0.5) / 2.0

#: The 12 vertices of a regular icosahedron, before normalizing to the unit
#: sphere.
_BASE_VERTICES: tuple[tuple[float, float, float], ...] = (
    (-1.0, _GOLDEN_RATIO, 0.0),
    (1.0, _GOLDEN_RATIO, 0.0),
    (-1.0, -_GOLDEN_RATIO, 0.0),
    (1.0, -_GOLDEN_RATIO, 0.0),
    (0.0, -1.0, _GOLDEN_RATIO),
    (0.0, 1.0, _GOLDEN_RATIO),
    (0.0, -1.0, -_GOLDEN_RATIO),
    (0.0, 1.0, -_GOLDEN_RATIO),
    (_GOLDEN_RATIO, 0.0, -1.0),
    (_GOLDEN_RATIO, 0.0, 1.0),
    (-_GOLDEN_RATIO, 0.0, -1.0),
    (-_GOLDEN_RATIO, 0.0, 1.0),
)

#: The 20 triangular faces of the base icosahedron, as index triples into
#: ``_BASE_VERTICES``.
_BASE_FACES: tuple[tuple[int, int, int], ...] = (
    (0, 11, 5),
    (0, 5, 1),
    (0, 1, 7),
    (0, 7, 10),
    (0, 10, 11),
    (1, 5, 9),
    (5, 11, 4),
    (11, 10, 2),
    (10, 7, 6),
    (7, 1, 8),
    (3, 9, 4),
    (3, 4, 2),
    (3, 2, 6),
    (3, 6, 8),
    (3, 8, 9),
    (4, 9, 5),
    (2, 4, 11),
    (6, 2, 10),
    (8, 6, 7),
    (9, 8, 1),
)


def generate_icosphere(
    subdivisions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate an indexed, unit-radius icosphere mesh.

    Args:
        subdivisions: Number of times to subdivide each triangle into four.
            Vertex count follows ``2 + 10 * 4**subdivisions`` (e.g. 4 ->
            2562 vertices, 5120 triangles).

    Returns:
        A ``(vertices, normals, indices)`` tuple:

        - ``vertices``: float32 array, shape ``(N, 3)``, unit-sphere
          positions.
        - ``normals``: float32 array, shape ``(N, 3)``, equal to the
          normalized vertex positions (the mesh is a sphere centered at the
          origin).
        - ``indices``: uint32 array, shape ``(M, 3)``, triangle winding
          (counter-clockwise when viewed from outside).
    """
    vertices: list[list[float]] = [_normalize(list(v)) for v in _BASE_VERTICES]
    faces: list[list[int]] = [list(f) for f in _BASE_FACES]

    for _ in range(subdivisions):
        vertices, faces = _subdivide(vertices, faces)

    vertex_array = np.asarray(vertices, dtype=np.float32)
    index_array = np.asarray(faces, dtype=np.uint32)
    # Unit sphere centered at the origin: normal == normalized position.
    normal_array = vertex_array.copy()
    return vertex_array, normal_array, index_array


def generate_fibonacci_sphere(count: int) -> np.ndarray:
    """Generate ``count`` unit vectors spread evenly over a sphere.

    Uses the Fibonacci lattice method, which distributes points with
    near-uniform density without any iterative relaxation. Used for the
    particle-field visual mode (Milestone 4), where each point is one
    particle's "home" direction.

    Returns:
        float32 array, shape ``(count, 3)``, each row a unit-length vector.
    """
    indices = np.arange(0, count, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / count)
    golden_angle = np.pi * (1.0 + 5.0**0.5)
    theta = golden_angle * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _normalize(vertex: list[float]) -> list[float]:
    arr = np.asarray(vertex, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    return (arr / norm).tolist()


def _subdivide(
    vertices: list[list[float]], faces: list[list[int]]
) -> tuple[list[list[float]], list[list[int]]]:
    """Split every triangle into four, reusing shared edge midpoints."""
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint_index(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        cached = midpoint_cache.get(key)
        if cached is not None:
            return cached

        mid = [(vertices[a][i] + vertices[b][i]) / 2.0 for i in range(3)]
        vertices.append(_normalize(mid))
        new_index = len(vertices) - 1
        midpoint_cache[key] = new_index
        return new_index

    new_faces: list[list[int]] = []
    for a, b, c in faces:
        ab = midpoint_index(a, b)
        bc = midpoint_index(b, c)
        ca = midpoint_index(c, a)
        new_faces.append([a, ab, ca])
        new_faces.append([b, bc, ab])
        new_faces.append([c, ca, bc])
        new_faces.append([ab, bc, ca])

    return vertices, new_faces
