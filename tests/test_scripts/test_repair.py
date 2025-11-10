"""Tests for repair nodes (FixNormalsNode, CheckNormalsNode, FillHolesNode, etc.)."""

import pytest
import numpy as np
from nodes.repair import (
    FixNormalsNode,
    CheckNormalsNode,
    FillHolesNode,
    ComputeNormalsNode,
    VisualizNormalFieldNode,
)


@pytest.mark.unit
def test_fix_normals(cube_mesh, save_mesh_helper, render_helper):
    """Test fixing mesh normals."""
    # Save original
    save_mesh_helper(cube_mesh, "00_original", "obj")
    render_helper(cube_mesh, "00_original")

    node = FixNormalsNode()
    fixed_mesh, info = node.fix_normals(mesh=cube_mesh)

    assert fixed_mesh is not None
    assert "normal" in info.lower()

    # Save fixed result
    save_mesh_helper(fixed_mesh, "01_fixed_normals", "obj")
    render_helper(fixed_mesh, "01_fixed_normals")


@pytest.mark.unit
def test_check_normals(sphere_mesh):
    """Test checking normal consistency."""
    node = CheckNormalsNode()
    report = node.check_normals(mesh=sphere_mesh)[0]

    assert report is not None
    assert "normal" in report.lower()


@pytest.mark.unit
def test_fill_holes(open_mesh, save_mesh_helper, render_helper):
    """Test filling holes in open mesh."""
    # Save original open mesh
    save_mesh_helper(open_mesh, "00_original_open", "obj")
    render_helper(open_mesh, "00_original_open")

    node = FillHolesNode()
    filled_mesh, info = node.fill_holes(mesh=open_mesh)

    assert filled_mesh is not None
    assert "hole" in info.lower()

    # Save filled result
    save_mesh_helper(filled_mesh, "01_holes_filled", "obj")
    render_helper(filled_mesh, "01_holes_filled")


@pytest.mark.unit
def test_compute_normals_faceted(cube_mesh, save_mesh_helper, render_helper):
    """Test computing faceted normals."""
    # Save original
    save_mesh_helper(cube_mesh, "00_original", "obj")
    render_helper(cube_mesh, "00_original")

    node = ComputeNormalsNode()
    mesh_with_normals = node.compute_normals(
        mesh=cube_mesh,
        smooth_vertex_normals="false"
    )[0]

    assert mesh_with_normals is not None
    assert hasattr(mesh_with_normals, 'face_normals')

    # Save with faceted normals
    save_mesh_helper(mesh_with_normals, "01_faceted_normals", "obj")
    render_helper(mesh_with_normals, "01_faceted_normals")


@pytest.mark.unit
def test_compute_normals_smooth(sphere_mesh, save_mesh_helper, render_helper):
    """Test computing smooth vertex normals."""
    # Save original
    save_mesh_helper(sphere_mesh, "00_original", "obj")
    render_helper(sphere_mesh, "00_original")

    node = ComputeNormalsNode()
    mesh_with_normals = node.compute_normals(
        mesh=sphere_mesh,
        smooth_vertex_normals="true"
    )[0]

    assert mesh_with_normals is not None

    # Save with smooth normals
    save_mesh_helper(mesh_with_normals, "01_smooth_normals", "obj")
    render_helper(mesh_with_normals, "01_smooth_normals")


@pytest.mark.unit
def test_visualize_normal_field(sphere_mesh, save_mesh_helper, render_helper):
    """Test visualizing normal field as scalar fields."""
    # Save original
    save_mesh_helper(sphere_mesh, "00_original", "obj")
    render_helper(sphere_mesh, "00_original")

    node = VisualizNormalFieldNode()
    mesh_with_fields, info = node.visualize_normals(mesh=sphere_mesh)

    assert mesh_with_fields is not None
    assert "normal" in info.lower()

    # Check that scalar fields were added
    if hasattr(mesh_with_fields, 'vertex_attributes'):
        attrs = mesh_with_fields.vertex_attributes
        assert any(k.startswith('normal_') for k in attrs.keys())

    # Save with fields (PLY supports vertex attributes)
    save_mesh_helper(mesh_with_fields, "01_with_normal_fields", "ply")
    render_helper(mesh_with_fields, "01_with_normal_fields")
