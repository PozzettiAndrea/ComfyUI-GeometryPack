"""
Test script to verify the GeomPack nodes work correctly
Now testing with trimesh and CGAL functionality
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import trimesh
from nodes import CreatePrimitive, SaveMesh, LoadMesh, MeshInfoNode, CGALRemeshNode


def test_create_primitive():
    """Test creating primitive shapes"""
    print("=" * 60)
    print("TEST 1: Create Primitive")
    print("=" * 60)

    node = CreatePrimitive()

    # Test cube
    print("\n[1/3] Testing cube creation...")
    mesh, = node.create_primitive("cube", 2.0, subdivisions=0)
    assert isinstance(mesh, trimesh.Trimesh), "Mesh should be a trimesh.Trimesh object"
    print(f"✓ Cube created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    assert len(mesh.vertices) == 8, "Cube should have 8 vertices"
    assert len(mesh.faces) == 12, "Cube should have 12 faces"

    # Test sphere
    print("\n[2/3] Testing sphere creation...")
    mesh, = node.create_primitive("sphere", 1.0, subdivisions=2)
    assert isinstance(mesh, trimesh.Trimesh), "Mesh should be a trimesh.Trimesh object"
    print(f"✓ Sphere created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    assert len(mesh.vertices) > 0, "Sphere should have vertices"
    assert len(mesh.faces) > 0, "Sphere should have faces"
    assert mesh.is_watertight, "Sphere should be watertight"

    # Test plane
    print("\n[3/3] Testing plane creation...")
    mesh, = node.create_primitive("plane", 1.0, subdivisions=3)
    assert isinstance(mesh, trimesh.Trimesh), "Mesh should be a trimesh.Trimesh object"
    print(f"✓ Plane created: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    assert len(mesh.vertices) == 16, "Plane with 3 subdivisions should have 16 vertices"

    print("\n✅ All primitive creation tests passed!")
    return mesh


def test_save_and_load(mesh):
    """Test saving and loading mesh"""
    print("\n" + "=" * 60)
    print("TEST 2: Save and Load Mesh")
    print("=" * 60)

    # Save mesh
    print("\n[1/2] Testing mesh save...")
    save_node = SaveMesh()
    test_file = "/tmp/test_mesh.obj"
    status, = save_node.save_mesh(mesh, test_file)
    print(f"✓ Mesh saved to {test_file}")
    print(f"  Status: {status}")

    # Load mesh
    print("\n[2/2] Testing mesh load...")
    load_node = LoadMesh()
    loaded_mesh, = load_node.load_mesh(test_file)
    assert isinstance(loaded_mesh, trimesh.Trimesh), "Loaded mesh should be a trimesh.Trimesh object"
    print(f"✓ Mesh loaded from {test_file}")
    print(f"  Vertices: {len(loaded_mesh.vertices)}")
    print(f"  Faces: {len(loaded_mesh.faces)}")

    # Verify
    assert len(loaded_mesh.vertices) == len(mesh.vertices), "Vertex count should match"
    assert len(loaded_mesh.faces) == len(mesh.faces), "Face count should match"

    print("\n✅ Save/Load test passed!")
    return loaded_mesh


def test_mesh_info(mesh):
    """Test mesh info node"""
    print("\n" + "=" * 60)
    print("TEST 3: Mesh Info")
    print("=" * 60)

    info_node = MeshInfoNode()
    info, = info_node.get_mesh_info(mesh)

    print("\n" + info)
    assert "Vertices:" in info, "Info should contain vertex count"
    assert "Faces:" in info, "Info should contain face count"
    assert "Volume:" in info, "Info should contain volume"
    assert "Is Watertight:" in info, "Info should contain watertight status"

    print("\n✅ Mesh info test passed!")


def test_cgal_remeshing():
    """Test CGAL remeshing node"""
    print("\n" + "=" * 60)
    print("TEST 4: CGAL Isotropic Remeshing")
    print("=" * 60)

    # Create a test mesh (cube)
    print("\n[1/3] Creating test cube...")
    create_node = CreatePrimitive()
    mesh, = create_node.create_primitive("cube", 1.0, subdivisions=0)
    original_vertices = len(mesh.vertices)
    original_faces = len(mesh.faces)
    print(f"✓ Original mesh: {original_vertices} vertices, {original_faces} faces")

    # Apply remeshing
    print("\n[2/3] Applying CGAL remeshing...")
    remesh_node = CGALRemeshNode()
    remeshed_mesh, = remesh_node.remesh(mesh, target_edge_length=0.15, iterations=3)
    assert isinstance(remeshed_mesh, trimesh.Trimesh), "Remeshed mesh should be a trimesh.Trimesh object"
    print(f"✓ Remeshed mesh: {len(remeshed_mesh.vertices)} vertices, {len(remeshed_mesh.faces)} faces")

    # Verify remeshing worked
    print("\n[3/3] Verifying remeshing results...")
    assert len(remeshed_mesh.vertices) > original_vertices, "Remeshing should increase vertex count"
    assert len(remeshed_mesh.faces) > original_faces, "Remeshing should increase face count"
    assert remeshed_mesh.is_watertight, "Remeshed mesh should be watertight"

    # Check that volume is approximately preserved
    volume_diff = abs(remeshed_mesh.volume - mesh.volume)
    volume_error = volume_diff / mesh.volume
    print(f"  Original volume: {mesh.volume:.6f}")
    print(f"  Remeshed volume: {remeshed_mesh.volume:.6f}")
    print(f"  Volume error: {volume_error*100:.2f}%")
    assert volume_error < 0.05, f"Volume should be approximately preserved (error: {volume_error*100:.2f}%)"

    print("\n✅ CGAL remeshing test passed!")
    return remeshed_mesh


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ComfyUI GeomPack Node Tests")
    print("=" * 60)

    try:
        # Test 1: Create primitives
        mesh = test_create_primitive()

        # Test 2: Save and load
        loaded_mesh = test_save_and_load(mesh)

        # Test 3: Mesh info
        test_mesh_info(loaded_mesh)

        # Test 4: CGAL remeshing
        remeshed_mesh = test_cgal_remeshing()

        # Summary
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe following nodes are working correctly:")
        print("  ✓ CreatePrimitive (cube, sphere, plane)")
        print("  ✓ SaveMesh (OBJ format)")
        print("  ✓ LoadMesh (OBJ format)")
        print("  ✓ MeshInfoNode (with trimesh enhancements)")
        print("  ✓ CGALRemeshNode (isotropic remeshing)")
        print("\nYou can now use these nodes in ComfyUI!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
