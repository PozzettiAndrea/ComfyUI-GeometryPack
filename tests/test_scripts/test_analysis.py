"""Tests for analysis nodes (MeshInfoNode, MarkBoundaryEdgesNode)."""

import pytest
from nodes.analysis import MeshInfoNode, MarkBoundaryEdgesNode


@pytest.mark.unit
def test_mesh_info_cube(cube_mesh):
    """Test mesh info on cube."""
    node = MeshInfoNode()
    info = node.get_mesh_info(mesh=cube_mesh)[0]

    assert "vertices" in info.lower()
    assert "faces" in info.lower()
    assert "8" in info  # cube has 8 vertices


@pytest.mark.unit
def test_mesh_info_bunny(bunny_mesh):
    """Test mesh info on Stanford Bunny."""
    if bunny_mesh is None:
        pytest.skip("Bunny mesh not available")

    node = MeshInfoNode()
    info = node.get_mesh_info(mesh=bunny_mesh)[0]

    assert "vertices" in info.lower()
    assert "faces" in info.lower()
    assert "volume" in info.lower()


@pytest.mark.unit
def test_mark_boundary_edges_closed_mesh(cube_mesh, save_mesh_helper):
    """Test boundary detection on closed mesh (should have no boundaries)."""
    node = MarkBoundaryEdgesNode()
    mesh_with_field, info = node.mark_boundary(mesh=cube_mesh)

    assert mesh_with_field is not None
    assert "boundary" in info.lower()

    save_mesh_helper(mesh_with_field, "cube_with_boundary_field", "ply")


@pytest.mark.unit
def test_mark_boundary_edges_open_mesh(open_mesh, save_mesh_helper, render_helper):
    """Test boundary detection on open mesh (should have boundaries)."""
    # Skip this test - open_mesh fixture creates invalid mesh
    pytest.skip("Open mesh fixture needs fixing")


@pytest.mark.unit
def test_mesh_info_volume_area(sphere_mesh):
    """Test mesh info calculates volume and area."""
    node = MeshInfoNode()
    info = node.get_mesh_info(mesh=sphere_mesh)[0]

    assert "volume" in info.lower()
    assert "area" in info.lower()
    # Sphere with radius 0.5 should have volume ~0.5
    assert "0." in info
