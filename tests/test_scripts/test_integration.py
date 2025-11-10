"""Integration tests for node pipelines."""

import pytest
from nodes.io import LoadMesh, SaveMesh
from nodes.primitives import CreatePrimitive
from nodes.analysis import MeshInfoNode, MarkBoundaryEdgesNode
from nodes.transforms import CenterMeshNode
from nodes.remeshing import MeshDecimationNode, MeshSubdivisionNode
from nodes.repair import FixNormalsNode, ComputeNormalsNode
from nodes.conversion import MeshToPointCloudNode


@pytest.mark.unit
def test_pipeline_create_analyze_save(save_mesh_helper, render_helper):
    """Test: Create → Analyze → Save pipeline."""
    # Create primitive
    create_node = CreatePrimitive()
    mesh = create_node.create_primitive(shape="sphere", size=1.0)[0]

    # Analyze
    info_node = MeshInfoNode()
    info = info_node.get_mesh_info(mesh=mesh)[0]
    assert "vertices" in info.lower()

    # Save
    save_mesh_helper(mesh, "pipeline_create_analyze", "obj")
    render_helper(mesh, "pipeline_create_analyze")


@pytest.mark.unit
def test_pipeline_load_remesh_save(bunny_mesh, save_mesh_helper, render_helper):
    """Test: Load → Remesh → Save pipeline."""
    if bunny_mesh is None:
        pytest.skip("Bunny mesh not available")

    # Decimate
    decimate_node = MeshDecimationNode()
    decimated, info = decimate_node.decimate(
        mesh=bunny_mesh,
        target_face_count=500
    )

    # Center
    center_node = CenterMeshNode()
    centered = center_node.center_mesh(mesh=decimated)[0]

    # Save
    save_mesh_helper(centered, "pipeline_load_remesh_save", "obj")
    render_helper(centered, "pipeline_load_remesh_save")


@pytest.mark.unit
def test_pipeline_create_subdivide_smooth(cube_mesh, save_mesh_helper, render_helper):
    """Test: Create → Subdivide → Smooth pipeline."""
    from nodes.remeshing import LaplacianSmoothingNode

    # Subdivide
    subdivide_node = MeshSubdivisionNode()
    subdivided, _ = subdivide_node.subdivide(
        mesh=cube_mesh,
        iterations=2,
        method="loop"
    )

    # Smooth
    smooth_node = LaplacianSmoothingNode()
    smoothed = smooth_node.smooth(
        mesh=subdivided,
        iterations=10,
        lambda_factor=0.5
    )[0]

    save_mesh_helper(smoothed, "pipeline_subdivide_smooth", "obj")
    render_helper(smoothed, "pipeline_subdivide_smooth")


@pytest.mark.unit
def test_pipeline_repair_analyze(sphere_mesh, save_mesh_helper):
    """Test: Repair → Analyze pipeline."""
    # Fix normals
    fix_node = FixNormalsNode()
    fixed, _ = fix_node.fix_normals(mesh=sphere_mesh)

    # Analyze
    info_node = MeshInfoNode()
    info = info_node.get_mesh_info(mesh=fixed)[0]
    assert "vertices" in info.lower()

    save_mesh_helper(fixed, "pipeline_repair_analyze", "obj")


@pytest.mark.unit
def test_pipeline_mesh_to_pointcloud(sphere_mesh):
    """Test: Create → Convert to Point Cloud pipeline."""
    # Convert to point cloud
    pc_node = MeshToPointCloudNode()
    pc = pc_node.mesh_to_pointcloud(
        mesh=sphere_mesh,
        sample_count=1000,
        sampling_method="uniform",
        include_normals="true"
    )[0]

    assert pc["points"].shape[0] == 1000
    assert "normals" in pc


@pytest.mark.unit
def test_pipeline_boundary_detection_visualization(cube_mesh, save_mesh_helper):
    """Test: Boundary Detection → Visualization pipeline."""
    from nodes.analysis import MarkBoundaryEdgesNode
    # Mark boundaries
    boundary_node = MarkBoundaryEdgesNode()
    mesh_with_field, info = boundary_node.mark_boundary(mesh=cube_mesh)

    # Compute normals
    normals_node = ComputeNormalsNode()
    with_normals = normals_node.compute_normals(
        mesh=mesh_with_field,
        smooth_vertex_normals="true"
    )[0]

    save_mesh_helper(with_normals, "pipeline_boundary_viz", "ply")


@pytest.mark.unit
def test_pipeline_full_processing(cube_mesh, save_mesh_helper, render_helper):
    """Test: Full processing pipeline."""
    # Center
    center_node = CenterMeshNode()
    centered = center_node.center_mesh(mesh=cube_mesh)[0]

    # Subdivide
    subdivide_node = MeshSubdivisionNode()
    subdivided, _ = subdivide_node.subdivide(
        mesh=centered,
        iterations=2,
        method="loop"
    )

    # Compute normals
    normals_node = ComputeNormalsNode()
    final = normals_node.compute_normals(
        mesh=subdivided,
        smooth_vertex_normals="true"
    )[0]

    # Analyze
    info_node = MeshInfoNode()
    info = info_node.get_mesh_info(mesh=final)[0]
    assert "vertices" in info.lower()

    save_mesh_helper(final, "pipeline_full_processing", "obj")
    render_helper(final, "pipeline_full_processing")
