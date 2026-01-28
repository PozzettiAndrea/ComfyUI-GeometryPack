# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Fix self-intersections by slightly moving vertices apart.
"""

import numpy as np
import trimesh


class FixSelfIntersectionsByPerturbationNode:
    """
    Fix self-intersections by slightly moving vertices apart.

    Perturbs vertices adjacent to self-intersecting faces by moving them
    along their normals. This is a non-destructive approach that preserves
    mesh topology but may not resolve all intersection types.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "epsilon": ("FLOAT", {"default": 0.001, "min": 1e-8, "max": 1.0, "step": 0.0001}),
                "max_iterations": ("INT", {"default": 10, "min": 1, "max": 100}),
                "direction": (["outward", "inward", "adaptive"], {"default": "outward"}),
                "scale_by_intersection_count": ("BOOLEAN", {"default": True}),
                "re_detect_after_fix": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("fixed_mesh", "report")
    FUNCTION = "fix_by_perturbation"
    CATEGORY = "geompack/repair"

    def fix_by_perturbation(self, trimesh, epsilon=0.001, max_iterations=10,
                            direction="outward", scale_by_intersection_count=True,
                            re_detect_after_fix=True):
        """
        Fix self-intersections by perturbing vertices along normals.

        Args:
            trimesh: Input trimesh.Trimesh object (should have intersection data)
            epsilon: Base distance to move vertices
            max_iterations: Maximum number of perturbation iterations
            direction: Direction to move ("outward", "inward", or "adaptive")
            scale_by_intersection_count: Scale displacement by number of intersections
            re_detect_after_fix: Re-run intersection detection after fix to update fields

        Returns:
            tuple: (fixed_mesh, report_string)
        """
        print(f"[FixByPerturbation] Processing mesh: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[FixByPerturbation] Params: epsilon={epsilon}, max_iter={max_iterations}, direction={direction}, re_detect={re_detect_after_fix}")

        # Check if mesh has self-intersection data
        if 'intersection_flag' not in trimesh.vertex_attributes:
            print("[FixByPerturbation] No intersection data found. Please run DetectSelfIntersections first.")

            warning_msg = """Warning: No self-intersection data found!

Please run 'Detect Self Intersections' node first to identify
which vertices are adjacent to self-intersecting faces.

Returning mesh unchanged.

Workflow suggestion:
1. Load Mesh → Detect Self Intersections → Fix Self Intersections By Perturbation
"""
            return (trimesh, warning_msg)

        # Get vertices to perturb
        vertex_flags = trimesh.vertex_attributes['intersection_flag']
        affected_vertices = np.where(vertex_flags > 0.5)[0]
        num_affected = len(affected_vertices)

        if num_affected == 0:
            print("[FixByPerturbation] No affected vertices found")
            report = """No Vertices to Perturb:

The mesh has no vertices marked as adjacent to self-intersections.
Either the mesh is already clean, or you need to run
'Detect Self Intersections' first.

Returning mesh unchanged.
"""
            return (trimesh, report)

        print(f"[FixByPerturbation] Found {num_affected} vertices to perturb")

        # Get intersection counts if available (for scaling)
        if scale_by_intersection_count and 'intersection_count' in trimesh.vertex_attributes:
            intersection_counts = trimesh.vertex_attributes['intersection_count']
            # Normalize to [0, 1] range for scaling
            max_count = np.max(intersection_counts)
            if max_count > 0:
                scale_factors = intersection_counts / max_count
            else:
                scale_factors = np.ones(len(trimesh.vertices))
        else:
            scale_factors = np.ones(len(trimesh.vertices))

        # Create result mesh
        result_mesh = trimesh.copy()

        # Compute vertex normals
        vertex_normals = result_mesh.vertex_normals

        # Determine displacement direction
        if direction == "outward":
            dir_multiplier = 1.0
        elif direction == "inward":
            dir_multiplier = -1.0
        else:  # adaptive
            # For adaptive, we'll try outward first
            dir_multiplier = 1.0

        # Perform perturbation
        total_displacement = 0.0
        iterations_used = 0

        for iteration in range(max_iterations):
            # Calculate displacement for this iteration
            iter_epsilon = epsilon * (iteration + 1) / max_iterations

            # Apply displacement to affected vertices
            displacement = np.zeros_like(result_mesh.vertices)

            for vid in affected_vertices:
                # Base displacement along normal
                vertex_disp = vertex_normals[vid] * iter_epsilon * dir_multiplier

                # Scale by intersection count if requested
                if scale_by_intersection_count:
                    vertex_disp *= scale_factors[vid]

                displacement[vid] = vertex_disp

            # Apply displacement
            result_mesh.vertices += displacement
            total_displacement += np.linalg.norm(displacement[affected_vertices], axis=1).mean()
            iterations_used = iteration + 1

            # Recompute normals after displacement
            result_mesh.vertex_normals  # Force recomputation
            vertex_normals = result_mesh.vertex_normals

            print(f"[FixByPerturbation] Iteration {iteration + 1}: avg displacement = {np.linalg.norm(displacement[affected_vertices], axis=1).mean():.6f}")

        # Get original intersection count for comparison
        original_intersecting_faces = np.sum(trimesh.face_attributes.get('self_intersecting', np.array([])) > 0.5)

        # Re-detect intersections if requested
        new_intersecting_faces = 0
        redetection_status = ""
        if re_detect_after_fix:
            print("[FixByPerturbation] Re-detecting self-intersections...")
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

                    print(f"[FixByPerturbation] After fix: {new_intersecting_faces} intersecting faces remain")
                else:
                    new_intersecting_faces = 0
                    result_mesh.face_attributes['self_intersecting'] = np.zeros(len(F), dtype=np.float32)
                    result_mesh.vertex_attributes['intersection_flag'] = np.zeros(len(V), dtype=np.float32)
                    result_mesh.vertex_attributes['intersection_count'] = np.zeros(len(V), dtype=np.float32)
                    print("[FixByPerturbation] ✓ No self-intersections remaining!")

                # Generate status message
                if original_intersecting_faces > 0:
                    reduction = original_intersecting_faces - new_intersecting_faces
                    reduction_pct = 100.0 * reduction / original_intersecting_faces
                    if new_intersecting_faces == 0:
                        redetection_status = f"  ✓ All {original_intersecting_faces} intersections FIXED!"
                    elif reduction > 0:
                        redetection_status = f"  ⚠ Reduced from {original_intersecting_faces} to {new_intersecting_faces} ({reduction_pct:.1f}% reduction)"
                    else:
                        redetection_status = f"  ✗ No improvement: still {new_intersecting_faces} intersecting faces"
                else:
                    redetection_status = f"  After fix: {new_intersecting_faces} intersecting faces"

            except Exception as e:
                print(f"[FixByPerturbation] Re-detection failed: {e}")
                redetection_status = f"  ⚠ Re-detection failed: {e}"
                # Clear old data since we couldn't update it
                if 'self_intersecting' in result_mesh.face_attributes:
                    del result_mesh.face_attributes['self_intersecting']
                if 'intersection_flag' in result_mesh.vertex_attributes:
                    del result_mesh.vertex_attributes['intersection_flag']
                if 'intersection_count' in result_mesh.vertex_attributes:
                    del result_mesh.vertex_attributes['intersection_count']
        else:
            # Clear old intersection data (needs manual re-detection)
            if 'self_intersecting' in result_mesh.face_attributes:
                del result_mesh.face_attributes['self_intersecting']
            if 'intersection_flag' in result_mesh.vertex_attributes:
                del result_mesh.vertex_attributes['intersection_flag']
            if 'intersection_count' in result_mesh.vertex_attributes:
                del result_mesh.vertex_attributes['intersection_count']
            redetection_status = "  (Re-detection disabled - run 'Detect Self Intersections' manually)"

        # Store metadata
        result_mesh.metadata['fixed_by_perturbation'] = True
        result_mesh.metadata['vertices_perturbed'] = num_affected
        result_mesh.metadata['total_displacement'] = float(total_displacement)
        result_mesh.metadata['iterations_used'] = iterations_used
        result_mesh.metadata['original_intersecting_faces'] = int(original_intersecting_faces)
        result_mesh.metadata['new_intersecting_faces'] = int(new_intersecting_faces)

        # Calculate mesh bounds change
        original_bounds = np.max(trimesh.vertices, axis=0) - np.min(trimesh.vertices, axis=0)
        new_bounds = np.max(result_mesh.vertices, axis=0) - np.min(result_mesh.vertices, axis=0)
        bounds_change = new_bounds - original_bounds

        # Generate report
        report = f"""Self-Intersection Fix By Perturbation:

Mesh Statistics:
  Vertices: {len(result_mesh.vertices):,}
  Faces: {len(result_mesh.faces):,}

Perturbation Applied:
  Vertices Affected: {num_affected:,} ({100.0 * num_affected / len(result_mesh.vertices):.1f}%)
  Direction: {direction}
  Epsilon: {epsilon:.6f}
  Iterations: {iterations_used}
  Total Average Displacement: {total_displacement:.6f}

Bounding Box Change:
  X: {bounds_change[0]:+.6f}
  Y: {bounds_change[1]:+.6f}
  Z: {bounds_change[2]:+.6f}

Re-Detection Results:
{redetection_status}

Status:
  ✓ Vertices perturbed along normals!

Important Notes:
  • Mesh topology preserved (same faces and connectivity)
  • Vertex positions modified to separate intersecting regions
  • If intersections persist, try increasing epsilon or iterations
  • For severe intersections, consider 'Fix By Removal' instead
"""

        print(f"[FixByPerturbation] Complete: perturbed {num_affected} vertices over {iterations_used} iterations")
        return (result_mesh, report)


NODE_CLASS_MAPPINGS = {
    "GeomPackFixSelfIntersectionsByPerturbation": FixSelfIntersectionsByPerturbationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackFixSelfIntersectionsByPerturbation": "Fix Self Intersections (Perturbation)",
}
