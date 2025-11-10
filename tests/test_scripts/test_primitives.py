"""Tests for primitive generation (CreatePrimitive)."""

import pytest
from nodes.primitives import CreatePrimitive


@pytest.mark.unit
@pytest.mark.parametrize("shape,expected_verts", [
    ("cube", 8),
    ("plane", 4),
])
def test_create_primitive_shapes(shape, expected_verts, render_helper, save_mesh_helper):
    """Test creating different primitive shapes."""
    node = CreatePrimitive()
    mesh = node.create_primitive(shape=shape, size=1.0)[0]

    assert mesh is not None
    assert mesh.vertices.shape[0] >= expected_verts
    assert mesh.faces.shape[0] > 0
    assert mesh.is_watertight or shape == "plane"

    save_mesh_helper(mesh, f"primitive_{shape}", "obj")
    render_helper(mesh, f"primitive_{shape}")


@pytest.mark.unit
def test_create_sphere(render_helper, save_mesh_helper):
    """Test creating sphere primitive."""
    node = CreatePrimitive()
    mesh = node.create_primitive(shape="sphere", size=1.0)[0]

    assert mesh is not None
    assert mesh.vertices.shape[0] > 10
    assert mesh.is_watertight
    # Size parameter is diameter, so radius = size/2
    assert abs(mesh.bounding_sphere.primitive.radius - 0.5) < 0.1

    save_mesh_helper(mesh, "primitive_sphere", "obj")
    render_helper(mesh, "primitive_sphere")


@pytest.mark.unit
def test_create_primitive_with_subdivisions(render_helper, save_mesh_helper):
    """Test creating primitive with subdivisions."""
    node = CreatePrimitive()
    mesh_sub0 = node.create_primitive(shape="sphere", size=1.0, subdivisions=0)[0]
    mesh_sub2 = node.create_primitive(shape="sphere", size=1.0, subdivisions=2)[0]

    assert mesh_sub2.vertices.shape[0] > mesh_sub0.vertices.shape[0]

    save_mesh_helper(mesh_sub2, "primitive_sphere_subdivided", "obj")
    render_helper(mesh_sub2, "primitive_sphere_subdivided")


@pytest.mark.unit
@pytest.mark.parametrize("size", [0.5, 1.0, 2.0])
def test_create_primitive_sizes(size):
    """Test creating primitives with different sizes."""
    node = CreatePrimitive()
    mesh = node.create_primitive(shape="cube", size=size)[0]

    bounds = mesh.bounds
    extent = bounds[1] - bounds[0]
    assert all(extent >= size * 0.9)
