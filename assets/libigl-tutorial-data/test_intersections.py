#!/usr/bin/env python3
"""
Test all meshes in this directory for self-intersections using libigl CGAL.
Reports which meshes have self-intersections and their statistics.
"""

import os
import sys
import numpy as np
import trimesh

def test_self_intersections(filepath):
    """
    Test a mesh file for self-intersections using libigl CGAL.

    Returns:
        tuple: (has_intersections, num_intersection_pairs, num_intersecting_faces, error_msg)
    """
    try:
        # Load mesh
        mesh = trimesh.load(filepath, force='mesh', process=False)

        if not isinstance(mesh, trimesh.Trimesh):
            return (False, 0, 0, "Not a single mesh object")

        # Try to use libigl with CGAL
        try:
            import igl
            import igl.copyleft.cgal as cgal

            if not hasattr(cgal, 'remesh_self_intersections'):
                return (False, 0, 0, "CGAL remesh_self_intersections not available")

        except (ImportError, AttributeError) as e:
            return (False, 0, 0, f"libigl CGAL not available: {e}")

        # Convert to numpy arrays (note: libigl requires int64 for faces)
        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int64)

        # Run detection with keyword arguments
        try:
            # Call remesh_self_intersections with detect_only=True
            VV, FF, IF, J, IM = cgal.remesh_self_intersections(
                V, F,
                detect_only=True,
                first_only=False,
                stitch_all=False
            )

            # IF contains pairs of intersecting faces [n x 2]
            num_intersection_pairs = IF.shape[0] if IF is not None and hasattr(IF, 'shape') else 0

            if num_intersection_pairs > 0:
                # Get unique face indices
                intersecting_faces = np.unique(IF.flatten())
                num_intersecting_faces = len(intersecting_faces)
                return (True, num_intersection_pairs, num_intersecting_faces, "")
            else:
                return (False, 0, 0, "")

        except Exception as e:
            return (False, 0, 0, f"Detection error: {e}")

    except Exception as e:
        return (False, 0, 0, f"Load error: {e}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 80)
    print("Testing Meshes for Self-Intersections")
    print("=" * 80)
    print()

    # Find all mesh files
    mesh_extensions = ['.obj', '.off', '.mesh', '.stl', '.ply']
    mesh_files = []

    for filename in sorted(os.listdir(script_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in mesh_extensions:
            mesh_files.append(filename)

    if not mesh_files:
        print("No mesh files found!")
        return

    print(f"Found {len(mesh_files)} mesh files to test:")
    for f in mesh_files:
        print(f"  - {f}")
    print()

    # Test each mesh
    results = []

    print("Testing...")
    print("-" * 80)

    for filename in mesh_files:
        filepath = os.path.join(script_dir, filename)
        print(f"Testing {filename}...", end=" ")
        sys.stdout.flush()

        has_intersections, num_pairs, num_faces, error = test_self_intersections(filepath)

        results.append({
            'filename': filename,
            'has_intersections': has_intersections,
            'num_pairs': num_pairs,
            'num_faces': num_faces,
            'error': error
        })

        if error:
            print(f"ERROR: {error}")
        elif has_intersections:
            print(f"✓ SELF-INTERSECTING ({num_pairs} pairs, {num_faces} faces)")
        else:
            print("✓ Clean (no intersections)")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    # Separate into categories
    self_intersecting = [r for r in results if r['has_intersections']]
    clean = [r for r in results if not r['has_intersections'] and not r['error']]
    errors = [r for r in results if r['error']]

    print(f"Total meshes tested: {len(results)}")
    print(f"Self-intersecting: {len(self_intersecting)}")
    print(f"Clean: {len(clean)}")
    print(f"Errors: {len(errors)}")
    print()

    if self_intersecting:
        print("Self-Intersecting Meshes (KEEP THESE):")
        print("-" * 80)
        for r in self_intersecting:
            print(f"  ✓ {r['filename']}")
            print(f"      Intersection pairs: {r['num_pairs']}")
            print(f"      Intersecting faces: {r['num_faces']}")
        print()

    if clean:
        print("Clean Meshes (DELETE THESE):")
        print("-" * 80)
        for r in clean:
            print(f"  ✗ {r['filename']}")
        print()

    if errors:
        print("Errors:")
        print("-" * 80)
        for r in errors:
            print(f"  ? {r['filename']}: {r['error']}")
        print()

    # Save results to file
    results_file = os.path.join(script_dir, "intersection_test_results.txt")
    with open(results_file, 'w') as f:
        f.write("Self-Intersection Test Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("KEEP (Self-Intersecting):\n")
        for r in self_intersecting:
            f.write(f"  {r['filename']} - {r['num_pairs']} pairs, {r['num_faces']} faces\n")
        f.write("\n")

        f.write("DELETE (Clean):\n")
        for r in clean:
            f.write(f"  {r['filename']}\n")
        f.write("\n")

        if errors:
            f.write("ERRORS:\n")
            for r in errors:
                f.write(f"  {r['filename']}: {r['error']}\n")

    print(f"Results saved to: {results_file}")
    print()

    return results


if __name__ == "__main__":
    results = main()
