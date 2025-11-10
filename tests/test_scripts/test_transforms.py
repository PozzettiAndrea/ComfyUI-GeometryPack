"""Tests for transform nodes (CenterMeshNode)."""

import pytest
import numpy as np
from nodes.transforms import CenterMeshNode


@pytest.mark.unit
def test_center_mesh(cube_mesh, save_mesh_helper, render_helper):
    """Test centering mesh at origin."""
    # Translate cube away from origin
    cube_mesh.vertices += [5.0, 3.0, -2.0]

    node = CenterMeshNode()
    centered = node.center_mesh(mesh=cube_mesh)[0]

    # Check that centroid is at origin
    centroid = centered.vertices.mean(axis=0)
    assert np.allclose(centroid, [0, 0, 0], atol=1e-6)

    save_mesh_helper(centered, "centered_cube", "obj")
    render_helper(centered, "centered_cube")


@pytest.mark.unit
def test_center_mesh_preserves_shape(sphere_mesh):
    """Test that centering preserves mesh shape."""
    original_bounds_extent = sphere_mesh.bounds[1] - sphere_mesh.bounds[0]

    node = CenterMeshNode()
    centered = node.center_mesh(mesh=sphere_mesh)[0]

    centered_extent = centered.bounds[1] - centered.bounds[0]
    assert np.allclose(original_bounds_extent, centered_extent, atol=1e-6)
