"""Tests for distance metric nodes."""

import pytest
from nodes.distance import HausdorffDistanceNode, ChamferDistanceNode, ComputeSDFNode


@pytest.mark.optional
def test_hausdorff_distance_identical(sphere_mesh):
    """Test Hausdorff distance between identical meshes."""
    pytest.importorskip("point_cloud_utils")

    node = HausdorffDistanceNode()
    distance, details = node.compute_hausdorff(
        mesh_a=sphere_mesh,
        mesh_b=sphere_mesh,
        sample_count=1000
    )

    assert distance < 0.01  # Should be near zero for identical meshes
    assert "hausdorff" in details.lower()


@pytest.mark.optional
def test_hausdorff_distance_different(cube_mesh, sphere_mesh):
    """Test Hausdorff distance between different meshes."""
    pytest.importorskip("point_cloud_utils")

    node = HausdorffDistanceNode()
    distance, details = node.compute_hausdorff(
        mesh_a=cube_mesh,
        mesh_b=sphere_mesh,
        sample_count=1000
    )

    assert distance > 0
    assert "hausdorff" in details.lower()


@pytest.mark.optional
def test_chamfer_distance_identical(sphere_mesh):
    """Test Chamfer distance between identical meshes."""
    pytest.importorskip("point_cloud_utils")

    node = ChamferDistanceNode()
    distance, info = node.compute_chamfer(
        mesh_a=sphere_mesh,
        mesh_b=sphere_mesh,
        sample_count=1000
    )

    assert distance < 0.01
    assert "chamfer" in info.lower()


@pytest.mark.optional
def test_chamfer_distance_different(cube_mesh, sphere_mesh):
    """Test Chamfer distance between different meshes."""
    pytest.importorskip("point_cloud_utils")

    node = ChamferDistanceNode()
    distance, info = node.compute_chamfer(
        mesh_a=cube_mesh,
        mesh_b=sphere_mesh,
        sample_count=1000
    )

    assert distance > 0


@pytest.mark.optional
@pytest.mark.slow
@pytest.mark.parametrize("resolution", [32, 64])
def test_compute_sdf(sphere_mesh, resolution):
    """Test SDF computation."""
    pytest.importorskip("mesh_to_sdf")

    node = ComputeSDFNode()
    sdf_volume, info = node.compute_sdf(
        mesh=sphere_mesh,
        resolution=resolution
    )

    assert sdf_volume is not None
    assert "voxels" in sdf_volume
    assert "resolution" in sdf_volume
    assert sdf_volume["resolution"] == resolution
    assert "sdf" in info.lower()
