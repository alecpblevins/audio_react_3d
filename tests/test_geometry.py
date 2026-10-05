"""Tests for procedural icosphere mesh generation."""

from __future__ import annotations

import numpy as np

from audio_reactive_3d.render.geometry import generate_fibonacci_sphere, generate_icosphere


def test_base_icosahedron_has_expected_counts() -> None:
    vertices, normals, indices = generate_icosphere(subdivisions=0)

    assert vertices.shape == (12, 3)
    assert normals.shape == (12, 3)
    assert indices.shape == (20, 3)


def test_subdivision_counts_match_closed_form() -> None:
    # Indexed-vertex count after n subdivisions: 2 + 10 * 4**n.
    for subdivisions in (1, 2, 3, 4):
        vertices, _normals, indices = generate_icosphere(subdivisions)
        expected_vertices = 2 + 10 * 4**subdivisions
        expected_triangles = 20 * 4**subdivisions
        assert vertices.shape[0] == expected_vertices
        assert indices.shape[0] == expected_triangles


def test_all_vertices_lie_on_unit_sphere() -> None:
    vertices, _normals, _indices = generate_icosphere(subdivisions=3)
    radii = np.linalg.norm(vertices, axis=1)
    assert np.allclose(radii, 1.0, atol=1e-5)


def test_normals_equal_normalized_positions() -> None:
    vertices, normals, _indices = generate_icosphere(subdivisions=2)
    assert np.allclose(normals, vertices, atol=1e-6)


def test_indices_reference_valid_vertices() -> None:
    vertices, _normals, indices = generate_icosphere(subdivisions=2)
    assert indices.min() >= 0
    assert indices.max() < vertices.shape[0]
    assert indices.dtype == np.uint32


def test_default_subdivision_level_is_near_2k_vertices() -> None:
    from audio_reactive_3d import config

    vertices, _normals, _indices = generate_icosphere(config.ICOSPHERE_SUBDIVISIONS)
    assert 1500 <= vertices.shape[0] <= 4000


def test_fibonacci_sphere_returns_requested_point_count() -> None:
    points = generate_fibonacci_sphere(2048)
    assert points.shape == (2048, 3)
    assert points.dtype == np.float32


def test_fibonacci_sphere_points_are_unit_length() -> None:
    points = generate_fibonacci_sphere(500)
    radii = np.linalg.norm(points, axis=1)
    assert np.allclose(radii, 1.0, atol=1e-5)


def test_fibonacci_sphere_points_are_evenly_distributed() -> None:
    # Nearest-neighbor spacing should be roughly uniform across the sphere
    # (no large gaps or clusters): the ratio of max to min nearest-neighbor
    # distance should stay modest for a well-distributed lattice.
    points = generate_fibonacci_sphere(1000)
    dot = points @ points.T
    np.fill_diagonal(dot, -1.0)  # exclude self-matches from the "nearest" search
    nearest_cos = dot.max(axis=1)
    nearest_angle = np.arccos(np.clip(nearest_cos, -1.0, 1.0))
    assert nearest_angle.max() / nearest_angle.min() < 3.0
