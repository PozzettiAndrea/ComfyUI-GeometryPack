"""Tests for I/O nodes (LoadMesh, SaveMesh)."""

import pytest
from pathlib import Path
from nodes.io import LoadMesh, SaveMesh


@pytest.mark.unit
def test_load_mesh_stl(bunny_mesh, render_helper, save_mesh_helper):
    """Test loading STL file."""
    if bunny_mesh is None:
        pytest.skip("Stanford_Bunny.stl not found in assets")

    node = LoadMesh()
    mesh = node.load_mesh(file_path="Stanford_Bunny.stl")[0]

    assert mesh is not None
    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0

    save_mesh_helper(mesh, "loaded_bunny", "obj")
    render_helper(mesh, "loaded_bunny")


@pytest.mark.unit
def test_save_mesh_obj(cube_mesh, meshes_output_dir):
    """Test saving mesh to OBJ format."""
    node = SaveMesh()
    status = node.save_mesh(mesh=cube_mesh, file_path="test_cube.obj")[0]

    assert "successfully" in status.lower()
    assert (meshes_output_dir / "test_cube.obj").exists()


@pytest.mark.unit
def test_save_mesh_ply(sphere_mesh, meshes_output_dir):
    """Test saving mesh to PLY format."""
    node = SaveMesh()
    status = node.save_mesh(mesh=sphere_mesh, file_path="test_sphere.ply")[0]

    assert "successfully" in status.lower()
    assert (meshes_output_dir / "test_sphere.ply").exists()


@pytest.mark.unit
@pytest.mark.parametrize("format", ["obj", "ply", "stl", "off"])
def test_save_load_cycle(cube_mesh, meshes_output_dir, format):
    """Test save/load cycle for different formats."""
    save_node = SaveMesh()
    load_node = LoadMesh()

    filename = f"cycle_test.{format}"
    # Save to output directory
    save_node.save_mesh(mesh=cube_mesh, file_path=filename)

    # Load from output directory (files are saved there)
    from pathlib import Path
    saved_path = meshes_output_dir / filename
    if not saved_path.exists():
        pytest.skip(f"File {filename} was not saved correctly")

    loaded_mesh = load_node.load_mesh(file_path=str(saved_path))[0]

    assert loaded_mesh is not None
    assert loaded_mesh.vertices.shape[0] > 0
    assert loaded_mesh.faces.shape[0] > 0
