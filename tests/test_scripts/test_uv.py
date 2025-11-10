"""Tests for UV unwrapping nodes."""

import pytest
from nodes.uv import (
    XAtlasUVUnwrapNode,
    LibiglLSCMNode,
    LibiglHarmonicNode,
    LibiglARAPNode,
    BlenderUVUnwrapNode,
    BlenderCubeProjectionNode,
    BlenderCylinderProjectionNode,
    BlenderSphereProjectionNode,
)


@pytest.mark.optional
def test_xatlas_unwrap(sphere_mesh, save_mesh_helper, render_helper):
    """Test xAtlas UV unwrapping."""
    pytest.importorskip("xatlas")

    node = XAtlasUVUnwrapNode()
    unwrapped = node.unwrap(mesh=sphere_mesh)[0]

    assert unwrapped is not None
    assert hasattr(unwrapped.visual, 'uv') or 'uv' in unwrapped.metadata

    save_mesh_helper(unwrapped, "sphere_xatlas_uv", "obj")
    render_helper(unwrapped, "sphere_xatlas_uv")


@pytest.mark.optional
def test_libigl_lscm(sphere_mesh, save_mesh_helper, render_helper):
    """Test libigl LSCM UV unwrapping."""
    pytest.importorskip("igl")

    node = LibiglLSCMNode()
    unwrapped = node.lscm_unwrap(mesh=sphere_mesh)[0]

    assert unwrapped is not None

    save_mesh_helper(unwrapped, "sphere_lscm_uv", "obj")
    render_helper(unwrapped, "sphere_lscm_uv")


@pytest.mark.optional
def test_libigl_harmonic(open_mesh, save_mesh_helper, render_helper):
    """Test libigl harmonic UV unwrapping."""
    pytest.importorskip("igl")

    node = LibiglHarmonicNode()
    unwrapped = node.harmonic_unwrap(mesh=open_mesh)[0]

    assert unwrapped is not None

    save_mesh_helper(unwrapped, "open_harmonic_uv", "obj")
    render_helper(unwrapped, "open_harmonic_uv")


@pytest.mark.optional
def test_libigl_arap(open_mesh, save_mesh_helper, render_helper):
    """Test libigl ARAP UV unwrapping."""
    pytest.importorskip("igl")

    node = LibiglARAPNode()
    unwrapped = node.arap_unwrap(mesh=open_mesh, iterations=10)[0]

    assert unwrapped is not None

    save_mesh_helper(unwrapped, "open_arap_uv", "obj")
    render_helper(unwrapped, "open_arap_uv")


@pytest.mark.blender
@pytest.mark.slow
def test_blender_uv_unwrap(sphere_mesh, save_mesh_helper, render_helper):
    """Test Blender Smart UV Project."""
    node = BlenderUVUnwrapNode()
    unwrapped = node.smart_uv_project(
        mesh=sphere_mesh,
        angle_limit=66.0,
        island_margin=0.02
    )[0]

    assert unwrapped is not None

    save_mesh_helper(unwrapped, "sphere_blender_smart_uv", "obj")
    render_helper(unwrapped, "sphere_blender_smart_uv")


@pytest.mark.blender
@pytest.mark.slow
def test_blender_cube_projection(cube_mesh, save_mesh_helper):
    """Test Blender cube projection UV."""
    node = BlenderCubeProjectionNode()
    unwrapped = node.cube_projection(mesh=cube_mesh, cube_size=1.0)[0]

    assert unwrapped is not None

    save_mesh_helper(unwrapped, "cube_projection_uv", "obj")


@pytest.mark.blender
@pytest.mark.slow
def test_blender_cylinder_projection(sphere_mesh, save_mesh_helper):
    """Test Blender cylinder projection UV."""
    node = BlenderCylinderProjectionNode()
    unwrapped = node.cylinder_projection(mesh=sphere_mesh, radius=1.0)[0]

    assert unwrapped is not None

    save_mesh_helper(unwrapped, "sphere_cylinder_uv", "obj")


@pytest.mark.blender
@pytest.mark.slow
def test_blender_sphere_projection(sphere_mesh, save_mesh_helper, render_helper):
    """Test Blender sphere projection UV."""
    node = BlenderSphereProjectionNode()
    unwrapped = node.sphere_projection(mesh=sphere_mesh)[0]

    assert unwrapped is not None

    save_mesh_helper(unwrapped, "sphere_sphere_uv", "obj")
    render_helper(unwrapped, "sphere_sphere_uv")
