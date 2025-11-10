"""Tests for conversion nodes (StripMeshAdjacencyNode, MeshToPointCloudNode)."""

import pytest
from nodes.conversion import StripMeshAdjacencyNode, MeshToPointCloudNode


@pytest.mark.unit
def test_strip_mesh_adjacency(cube_mesh):
    """Test converting mesh to point cloud without sampling."""
    node = StripMeshAdjacencyNode()
    pc = node.strip_adjacency(mesh=cube_mesh)[0]

    assert pc is not None
    assert "points" in pc
    assert pc["points"].shape[0] == cube_mesh.vertices.shape[0]


@pytest.mark.unit
def test_strip_mesh_with_normals(cube_mesh):
    """Test converting mesh to point cloud with normals."""
    node = StripMeshAdjacencyNode()
    pc = node.strip_adjacency(mesh=cube_mesh, include_normals="true")[0]

    assert "normals" in pc
    assert pc["normals"].shape[0] == pc["points"].shape[0]


@pytest.mark.unit
@pytest.mark.parametrize("sample_count", [100, 500, 1000])
def test_mesh_to_point_cloud_sampling(sphere_mesh, sample_count):
    """Test sampling point cloud from mesh."""
    node = MeshToPointCloudNode()
    pc = node.mesh_to_pointcloud(
        mesh=sphere_mesh,
        sample_count=sample_count,
        sampling_method="uniform"
    )[0]

    assert pc["points"].shape[0] == sample_count


@pytest.mark.unit
@pytest.mark.parametrize("method", ["uniform", "face_weighted"])
def test_mesh_to_point_cloud_methods(sphere_mesh, method):
    """Test different sampling methods."""
    node = MeshToPointCloudNode()
    pc = node.mesh_to_pointcloud(
        mesh=sphere_mesh,
        sample_count=500,
        sampling_method=method
    )[0]

    assert pc is not None
    # "face_weighted" might return different count
    assert pc["points"].shape[0] > 0


@pytest.mark.unit
def test_point_cloud_with_normals(sphere_mesh):
    """Test point cloud generation with normals."""
    node = MeshToPointCloudNode()
    pc = node.mesh_to_pointcloud(
        mesh=sphere_mesh,
        sample_count=300,
        sampling_method="uniform",
        include_normals="true"
    )[0]

    assert "normals" in pc
    assert pc["normals"].shape == (300, 3)
