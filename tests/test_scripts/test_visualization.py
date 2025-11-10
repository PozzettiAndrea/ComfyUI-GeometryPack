"""Tests for visualization nodes."""

import pytest
from pathlib import Path
from nodes.visualization import (
    PreviewMeshNode,
    PreviewMeshVTKNode,
    PreviewMeshVTKFiltersNode,
    PreviewMeshVTKFieldsNode,
)


@pytest.mark.unit
def test_preview_mesh_threejs(sphere_mesh, test_output_dir):
    """Test Three.js mesh preview (GLB export)."""
    node = PreviewMeshNode()
    result = node.preview_mesh(mesh=sphere_mesh)

    # Check that result contains UI data
    assert result is not None
    assert "ui" in result
    assert "mesh_file" in result["ui"]

    # Verify GLB file was created
    glb_files = list(test_output_dir.rglob("*.glb"))
    assert len(glb_files) > 0


@pytest.mark.unit
def test_preview_mesh_vtk(sphere_mesh, test_output_dir):
    """Test VTK.js mesh preview (STL export)."""
    node = PreviewMeshVTKNode()
    result = node.preview_mesh_vtk(mesh=sphere_mesh)

    assert result is not None
    assert "ui" in result

    # Verify STL file was created
    stl_files = list(test_output_dir.rglob("*.stl"))
    assert len(stl_files) > 0


@pytest.mark.unit
def test_preview_mesh_vtk_filters(sphere_mesh, test_output_dir):
    """Test VTK.js preview with filters."""
    node = PreviewMeshVTKFiltersNode()
    result = node.preview_mesh_vtk_filters(mesh=sphere_mesh)

    assert result is not None
    assert "ui" in result


@pytest.mark.unit
def test_preview_mesh_vtk_fields(sphere_mesh, test_output_dir):
    """Test VTK.js preview with scalar fields."""
    # Add a simple scalar field
    if not hasattr(sphere_mesh, 'vertex_attributes'):
        sphere_mesh.vertex_attributes = {}
    sphere_mesh.vertex_attributes['test_field'] = sphere_mesh.vertices[:, 2]

    node = PreviewMeshVTKFieldsNode()
    result = node.preview_mesh_vtk_fields(mesh=sphere_mesh)

    assert result is not None
    assert "ui" in result

    # Verify VTP file was created (supports fields)
    vtp_files = list(test_output_dir.rglob("*.vtp"))
    assert len(vtp_files) > 0
