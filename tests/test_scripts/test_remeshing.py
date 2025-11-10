"""Tests for remeshing nodes."""

import pytest
from nodes.remeshing import (
    PyMeshLabRemeshNode,
    CGALIsotropicRemeshNode,
    MeshDecimationNode,
    MeshSubdivisionNode,
    LaplacianSmoothingNode,
    BlenderVoxelRemeshNode,
    BlenderQuadriflowRemeshNode,
    InstantMeshesRemeshNode,
)


@pytest.mark.unit
@pytest.mark.optional
def test_pymeshlab_remesh(sphere_mesh, save_mesh_helper, render_helper):
    """Test PyMeshLab isotropic remeshing."""
    pytest.importorskip("pymeshlab")

    # Save original
    save_mesh_helper(sphere_mesh, "00_original", "obj")
    render_helper(sphere_mesh, "00_original")

    node = PyMeshLabRemeshNode()
    remeshed = node.remesh(
        mesh=sphere_mesh,
        target_edge_length=0.1,
        iterations=10
    )[0]

    assert remeshed is not None
    assert remeshed.faces.shape[0] > 0

    # Save remeshed result
    save_mesh_helper(remeshed, "01_pymeshlab_remeshed", "obj")
    render_helper(remeshed, "01_pymeshlab_remeshed")


@pytest.mark.optional
def test_cgal_remesh(sphere_mesh, save_mesh_helper, render_helper):
    """Test CGAL isotropic remeshing."""
    pytest.importorskip("cgal")

    # Save original
    save_mesh_helper(sphere_mesh, "00_original", "obj")
    render_helper(sphere_mesh, "00_original")

    node = CGALIsotropicRemeshNode()
    remeshed = node.remesh(
        mesh=sphere_mesh,
        target_edge_length=0.1,
        iterations=5,
        protect_boundaries="false"
    )[0]

    assert remeshed is not None
    assert remeshed.faces.shape[0] > 0

    # Save remeshed result
    save_mesh_helper(remeshed, "01_cgal_remeshed", "obj")
    render_helper(remeshed, "01_cgal_remeshed")


@pytest.mark.unit
def test_mesh_decimation(sphere_mesh, save_mesh_helper, render_helper):
    """Test mesh decimation - reduces triangle count while preserving shape."""
    # Save original
    save_mesh_helper(sphere_mesh, "00_original", "obj")
    render_helper(sphere_mesh, "00_original")

    original_face_count = sphere_mesh.faces.shape[0]
    target_faces = original_face_count // 4

    node = MeshDecimationNode()
    decimated, info = node.decimate(
        mesh=sphere_mesh,
        target_face_count=target_faces
    )

    assert decimated is not None
    assert decimated.faces.shape[0] < original_face_count
    assert "face" in info.lower()

    # Save decimated result
    save_mesh_helper(decimated, "01_decimated", "obj")
    render_helper(decimated, "01_decimated")


@pytest.mark.unit
@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_mesh_subdivision(cube_mesh, iterations, save_mesh_helper, render_helper):
    """Test mesh subdivision - increases mesh resolution."""
    # Save original (only on first iteration to avoid duplication across params)
    save_mesh_helper(cube_mesh, "00_original", "obj")
    render_helper(cube_mesh, "00_original")

    original_face_count = cube_mesh.faces.shape[0]

    node = MeshSubdivisionNode()
    subdivided, info = node.subdivide(
        mesh=cube_mesh,
        iterations=iterations,
        method="loop"
    )

    assert subdivided is not None
    assert subdivided.faces.shape[0] > original_face_count

    # Save with iteration count in name
    save_mesh_helper(subdivided, f"01_subdivided_x{iterations}", "obj")
    render_helper(subdivided, f"01_subdivided_x{iterations}")


@pytest.mark.unit
def test_laplacian_smoothing(sphere_mesh, save_mesh_helper, render_helper):
    """Test Laplacian smoothing - smooths mesh surface."""
    # Save original
    save_mesh_helper(sphere_mesh, "00_original", "obj")
    render_helper(sphere_mesh, "00_original")

    node = LaplacianSmoothingNode()
    smoothed = node.smooth(
        mesh=sphere_mesh,
        iterations=10,
        lambda_factor=0.5
    )[0]

    assert smoothed is not None
    assert smoothed.vertices.shape == sphere_mesh.vertices.shape

    # Save smoothed result
    save_mesh_helper(smoothed, "01_smoothed", "obj")
    render_helper(smoothed, "01_smoothed")


@pytest.mark.blender
@pytest.mark.slow
def test_blender_voxel_remesh(sphere_mesh, save_mesh_helper, render_helper):
    """Test Blender voxel remeshing."""
    # Save original
    save_mesh_helper(sphere_mesh, "00_original", "obj")
    render_helper(sphere_mesh, "00_original")

    node = BlenderVoxelRemeshNode()
    remeshed = node.voxel_remesh(mesh=sphere_mesh, voxel_size=0.1)[0]

    assert remeshed is not None

    # Save remeshed result
    save_mesh_helper(remeshed, "01_voxel_remeshed", "obj")
    render_helper(remeshed, "01_voxel_remeshed")


@pytest.mark.blender
@pytest.mark.slow
def test_blender_quadriflow_remesh(sphere_mesh, save_mesh_helper, render_helper):
    """Test Blender Quadriflow remeshing."""
    # Save original
    save_mesh_helper(sphere_mesh, "00_original", "obj")
    render_helper(sphere_mesh, "00_original")

    node = BlenderQuadriflowRemeshNode()
    remeshed = node.quadriflow_remesh(
        mesh=sphere_mesh,
        target_face_count=500
    )[0]

    assert remeshed is not None

    # Save remeshed result
    save_mesh_helper(remeshed, "01_quadriflow_remeshed", "obj")
    render_helper(remeshed, "01_quadriflow_remeshed")


@pytest.mark.optional
@pytest.mark.slow
def test_instant_meshes_remesh(sphere_mesh, save_mesh_helper, render_helper):
    """Test InstantMeshes remeshing."""
    pytest.importorskip("PyNanoInstantMeshes")

    # Save original
    save_mesh_helper(sphere_mesh, "00_original", "obj")
    render_helper(sphere_mesh, "00_original")

    node = InstantMeshesRemeshNode()
    remeshed, info = node.instant_remesh(
        mesh=sphere_mesh,
        target_vertex_count=500,
        deterministic="true",
        crease_angle=30.0
    )

    assert remeshed is not None
    assert "vertices" in info.lower()

    # Save remeshed result
    save_mesh_helper(remeshed, "01_instant_remeshed", "obj")
    render_helper(remeshed, "01_instant_remeshed")
