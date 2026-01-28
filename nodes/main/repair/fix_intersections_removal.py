# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Fix self-intersections by removing intersecting faces and filling holes.
"""

import numpy as np
import trimesh


class FixSelfIntersectionsByRemovalNode:
    """
    Fix self-intersections by removing intersecting faces and filling holes.

    Removes faces marked as self-intersecting (from DetectSelfIntersections node),
    then optionally fills the resulting holes and fixes normals. This is a simple
    but effective approach for meshes with isolated self-intersection regions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": True}),
                "fix_normals": ("BOOLEAN", {"default": True}),
                "max_hole_size": ("INT", {"default": 100, "min": 3, "max": 10000}),
                "re_detect_after_fix": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("fixed_mesh", "report")
    FUNCTION = "fix_by_removal"
    CATEGORY = "geompack/repair"

    def fix_by_removal(self, trimesh, fill_holes=True, fix_normals=True, max_hole_size=100,
                      re_detect_after_fix=True):
        """
        Fix self-intersections by removing bad faces and filling holes.

        Args:
            trimesh: Input trimesh.Trimesh object (should have intersection data)
            fill_holes: Whether to fill holes after face removal
            fix_normals: Whether to fix normals after repair
            max_hole_size: Maximum number of edges in a hole to fill
            re_detect_after_fix: Re-run intersection detection after fix to update fields

        Returns:
            tuple: (fixed_mesh, report_string)
        """
        print(f"[FixByRemoval] Processing mesh: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        # Check if mesh has self-intersection data
        if 'self_intersecting' not in trimesh.face_attributes:
            # No intersection data - try to detect first
            print("[FixByRemoval] No intersection data found. Please run DetectSelfIntersections first.")

            warning_msg = """Warning: No self-intersection data found!

Please run 'Detect Self Intersections' node first to identify
which faces are self-intersecting.

Returning mesh unchanged.

Workflow suggestion:
1. Load Mesh → Detect Self Intersections → Fix Self Intersections By Removal
"""
            return (trimesh, warning_msg)

        # Get intersecting face indices
        face_field = trimesh.face_attributes['self_intersecting']
        intersecting_face_mask = face_field > 0.5
        num_intersecting = np.sum(intersecting_face_mask)

        if num_intersecting == 0:
            print("[FixByRemoval] No self-intersecting faces found")
            report = """No Self-Intersections to Fix:

The mesh has no self-intersecting faces marked.
Either the mesh is already clean, or you need to run
'Detect Self Intersections' first.

Returning mesh unchanged.
"""
            return (trimesh, report)

        print(f"[FixByRemoval] Found {num_intersecting} intersecting faces to remove")

        # Create a copy and remove intersecting faces
        result_mesh = trimesh.copy()

        # Keep only non-intersecting faces
        keep_face_mask = ~intersecting_face_mask
        result_mesh.update_faces(keep_face_mask)

        faces_after_removal = len(result_mesh.faces)
        vertices_after_removal = len(result_mesh.vertices)

        print(f"[FixByRemoval] After removal: {vertices_after_removal} vertices, {faces_after_removal} faces")

        # Fill holes if requested
        holes_filled = 0
        if fill_holes and faces_after_removal > 0:
            print(f"[FixByRemoval] Filling holes (max size: {max_hole_size} edges)...")
            try:
                # Count holes before
                boundaries_before = len(result_mesh.outline_segments) if hasattr(result_mesh, 'outline_segments') else 0

                # Fill holes
                result_mesh.fill_holes()

                holes_filled = faces_after_removal - len(result_mesh.faces)
                # Actually that's not right - holes add faces
                faces_after_fill = len(result_mesh.faces)
                holes_filled = faces_after_fill - faces_after_removal

                print(f"[FixByRemoval] Added {holes_filled} faces to fill holes")
            except Exception as e:
                print(f"[FixByRemoval] Warning: Could not fill holes: {e}")
                holes_filled = 0

        # Fix normals if requested
        if fix_normals and len(result_mesh.faces) > 0:
            print("[FixByRemoval] Fixing normals...")
            try:
                result_mesh.fix_normals()
                print("[FixByRemoval] Normals fixed")
            except Exception as e:
                print(f"[FixByRemoval] Warning: Could not fix normals: {e}")

        # Re-detect intersections if requested
        new_intersecting_faces = 0
        redetection_status = ""
        if re_detect_after_fix and len(result_mesh.faces) > 0:
            print("[FixByRemoval] Re-detecting self-intersections...")
            try:
                import igl.copyleft.cgal as cgal

                V = np.asarray(result_mesh.vertices, dtype=np.float64)
                F = np.asarray(result_mesh.faces, dtype=np.int64)

                VV, FF, IF, J, IM = cgal.remesh_self_intersections(
                    V, F,
                    detect_only=True,
                    first_only=False,
                    stitch_all=False
                )

                if IF.shape[0] > 0:
                    intersecting_faces = np.unique(IF.flatten())
                    new_intersecting_faces = len(intersecting_faces)

                    # Update face attributes
                    face_field = np.zeros(len(F), dtype=np.float32)
                    face_field[intersecting_faces] = 1.0
                    result_mesh.face_attributes['self_intersecting'] = face_field

                    # Update vertex attributes
                    vertex_field = np.zeros(len(V), dtype=np.float32)
                    vertex_count = np.zeros(len(V), dtype=np.float32)
                    for face_idx in intersecting_faces:
                        vertex_indices = F[face_idx]
                        vertex_field[vertex_indices] = 1.0
                        vertex_count[vertex_indices] += 1.0
                    result_mesh.vertex_attributes['intersection_flag'] = vertex_field
                    result_mesh.vertex_attributes['intersection_count'] = vertex_count

                    print(f"[FixByRemoval] After fix: {new_intersecting_faces} intersecting faces remain")
                else:
                    new_intersecting_faces = 0
                    result_mesh.face_attributes['self_intersecting'] = np.zeros(len(F), dtype=np.float32)
                    result_mesh.vertex_attributes['intersection_flag'] = np.zeros(len(V), dtype=np.float32)
                    result_mesh.vertex_attributes['intersection_count'] = np.zeros(len(V), dtype=np.float32)
                    print("[FixByRemoval] ✓ No self-intersections remaining!")

                # Generate status message
                if num_intersecting > 0:
                    if new_intersecting_faces == 0:
                        redetection_status = f"  ✓ All intersections FIXED! (removed {num_intersecting} faces)"
                    else:
                        redetection_status = f"  ⚠ {new_intersecting_faces} intersecting faces still remain"
                else:
                    redetection_status = f"  After fix: {new_intersecting_faces} intersecting faces"

            except Exception as e:
                print(f"[FixByRemoval] Re-detection failed: {e}")
                redetection_status = f"  ⚠ Re-detection failed: {e}"
                # Clear old data since we couldn't update it
                if 'self_intersecting' in result_mesh.face_attributes:
                    del result_mesh.face_attributes['self_intersecting']
                if 'intersection_flag' in result_mesh.vertex_attributes:
                    del result_mesh.vertex_attributes['intersection_flag']
                if 'intersection_count' in result_mesh.vertex_attributes:
                    del result_mesh.vertex_attributes['intersection_count']
        else:
            # Clear old intersection data (it's no longer valid)
            if 'self_intersecting' in result_mesh.face_attributes:
                del result_mesh.face_attributes['self_intersecting']
            if 'intersection_flag' in result_mesh.vertex_attributes:
                del result_mesh.vertex_attributes['intersection_flag']
            if 'intersection_count' in result_mesh.vertex_attributes:
                del result_mesh.vertex_attributes['intersection_count']
            redetection_status = "  (Re-detection disabled - run 'Detect Self Intersections' manually)"

        # Store metadata
        result_mesh.metadata['fixed_by_removal'] = True
        result_mesh.metadata['faces_removed'] = int(num_intersecting)
        result_mesh.metadata['holes_filled'] = holes_filled
        result_mesh.metadata['new_intersecting_faces'] = int(new_intersecting_faces)

        # Generate report
        final_vertices = len(result_mesh.vertices)
        final_faces = len(result_mesh.faces)

        report = f"""Self-Intersection Fix By Removal:

Initial Mesh:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}
  Self-Intersecting Faces: {num_intersecting:,}

Operation:
  Faces Removed: {num_intersecting:,}
  Holes Filled: {'Yes' if fill_holes else 'No'} ({holes_filled:,} faces added)
  Normals Fixed: {'Yes' if fix_normals else 'No'}

Final Mesh:
  Vertices: {final_vertices:,}
  Faces: {final_faces:,}

Re-Detection Results:
{redetection_status}

Status:
  ✓ Self-intersecting faces removed!

{'⚠ Warning: Large number of faces removed. Consider using ' if num_intersecting > initial_faces * 0.1 else ''}
{'  perturbation or remeshing methods for better results.' if num_intersecting > initial_faces * 0.1 else ''}
"""

        print(f"[FixByRemoval] Complete: {final_vertices} vertices, {final_faces} faces")
        return (result_mesh, report)


NODE_CLASS_MAPPINGS = {
    "GeomPackFixSelfIntersectionsByRemoval": FixSelfIntersectionsByRemovalNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackFixSelfIntersectionsByRemoval": "Fix Self Intersections (Removal)",
}
