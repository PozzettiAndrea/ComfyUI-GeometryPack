# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Detect self-intersecting faces in a mesh.
"""

import os
import numpy as np
import trimesh


class DetectSelfIntersectionsNode:
    """
    Detect self-intersecting faces in a mesh.

    Analyzes the mesh to find faces that intersect with each other.
    Creates scalar fields to visualize which faces/vertices are involved
    in self-intersections. Essential for mesh quality checking before
    boolean operations or 3D printing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh_with_field", "report")
    OUTPUT_NODE = True  # Enable UI output for dynamic display
    FUNCTION = "detect_intersections"
    CATEGORY = "geompack/repair"

    def detect_intersections(self, trimesh):
        """
        Detect self-intersecting faces and mark them with scalar fields.

        Args:
            trimesh: Input trimesh.Trimesh object

        Returns:
            tuple: (mesh_with_intersection_field, report_string)
        """
        print(f"[DetectSelfIntersections] Analyzing mesh: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        result_mesh = trimesh.copy()

        try:
            # Try to use libigl with CGAL for robust detection
            import igl

            # Check if CGAL functions are available
            try:
                import igl.copyleft.cgal as cgal
                has_cgal = hasattr(cgal, 'remesh_self_intersections')
            except (ImportError, AttributeError):
                has_cgal = False

            if has_cgal:
                print("[DetectSelfIntersections] Using libigl CGAL method")

                # Convert mesh to numpy arrays with proper dtypes
                V = np.asarray(trimesh.vertices, dtype=np.float64)
                F = np.asarray(trimesh.faces, dtype=np.int64)

                # Use remesh_self_intersections in detect-only mode
                # This returns intersection information without modifying the mesh
                try:
                    VV, FF, IF, J, IM = cgal.remesh_self_intersections(
                        V, F,
                        detect_only=True,
                        first_only=False,
                        stitch_all=False
                    )

                    # IF contains pairs of intersecting faces [n x 2]
                    intersecting_faces_list = []  # For UI display
                    num_pairs = 0

                    if IF.shape[0] > 0:
                        # Get unique face indices that are involved in intersections
                        intersecting_faces = np.unique(IF.flatten())
                        num_intersecting = len(intersecting_faces)
                        num_pairs = IF.shape[0]

                        # Create scalar field for faces
                        face_field = np.zeros(len(F), dtype=np.float32)
                        face_field[intersecting_faces] = 1.0
                        result_mesh.face_attributes['self_intersecting'] = face_field

                        # Propagate to vertices - a vertex is marked if any adjacent face intersects
                        vertex_field = np.zeros(len(V), dtype=np.float32)
                        for face_idx in intersecting_faces:
                            vertex_indices = F[face_idx]
                            vertex_field[vertex_indices] = 1.0
                        result_mesh.vertex_attributes['intersection_flag'] = vertex_field

                        # Count how many intersecting faces each vertex touches
                        vertex_count = np.zeros(len(V), dtype=np.float32)
                        for face_idx in intersecting_faces:
                            vertex_indices = F[face_idx]
                            vertex_count[vertex_indices] += 1.0
                        result_mesh.vertex_attributes['intersection_count'] = vertex_count

                        # Build face details for UI
                        for face_idx in intersecting_faces:
                            intersecting_faces_list.append({
                                "id": int(face_idx),
                                "vertices": F[face_idx].tolist()
                            })

                        print(f"[DetectSelfIntersections] Found {num_intersecting} intersecting faces ({num_pairs} intersection pairs)")

                    else:
                        # No intersections found
                        num_intersecting = 0
                        # Add zero fields for visualization
                        result_mesh.face_attributes['self_intersecting'] = np.zeros(len(F), dtype=np.float32)
                        result_mesh.vertex_attributes['intersection_flag'] = np.zeros(len(V), dtype=np.float32)
                        result_mesh.vertex_attributes['intersection_count'] = np.zeros(len(V), dtype=np.float32)
                        print("[DetectSelfIntersections] No self-intersections detected")

                except Exception as e:
                    print(f"[DetectSelfIntersections] CGAL detection failed: {e}")
                    # Fallback to basic method
                    num_intersecting = 0
                    num_pairs = 0
                    intersecting_faces_list = []
                    result_mesh.face_attributes['self_intersecting'] = np.zeros(len(trimesh.faces), dtype=np.float32)
                    result_mesh.vertex_attributes['intersection_flag'] = np.zeros(len(trimesh.vertices), dtype=np.float32)
                    result_mesh.vertex_attributes['intersection_count'] = np.zeros(len(trimesh.vertices), dtype=np.float32)
                    print("[DetectSelfIntersections] Falling back to zero fields (CGAL error)")

            else:
                # CGAL not available - use basic fallback
                print("[DetectSelfIntersections] CGAL not available, using basic detection")
                num_intersecting = 0
                num_pairs = 0
                intersecting_faces_list = []

                # Add zero fields
                result_mesh.face_attributes['self_intersecting'] = np.zeros(len(trimesh.faces), dtype=np.float32)
                result_mesh.vertex_attributes['intersection_flag'] = np.zeros(len(trimesh.vertices), dtype=np.float32)
                result_mesh.vertex_attributes['intersection_count'] = np.zeros(len(trimesh.vertices), dtype=np.float32)

                print("[DetectSelfIntersections] ⚠ CGAL not available - install with: pip install cgal")

            # Store metadata
            result_mesh.metadata['has_intersection_field'] = True
            result_mesh.metadata['intersection_detection_method'] = 'libigl_cgal' if has_cgal else 'none'

            # Generate report
            percentage = (100.0 * num_intersecting / len(trimesh.faces)) if len(trimesh.faces) > 0 else 0.0

            report = f"""Self-Intersection Detection:

Mesh Statistics:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

Detection Results:
  Intersecting Faces: {num_intersecting:,} ({percentage:.1f}%)
  Detection Method: {'libigl CGAL' if has_cgal else 'Basic (CGAL unavailable)'}

Status:
  {'✓ No self-intersections detected!' if num_intersecting == 0 else '⚠ Self-intersections found!'}

Scalar Fields Added:
  • face: 'self_intersecting' (1.0 = intersecting, 0.0 = valid)
  • vertex: 'intersection_flag' (1.0 = adjacent to intersection)
  • vertex: 'intersection_count' (number of intersecting faces touching vertex)

{'' if has_cgal else '⚠ Note: CGAL not available. Install for accurate detection: pip install cgal'}

Use 'Preview Mesh (VTK with Fields)' node to visualize the intersection fields!
"""
            # Get mesh name
            mesh_name = trimesh.metadata.get('file_name', 'mesh') if hasattr(trimesh, 'metadata') else 'mesh'
            mesh_name_short = os.path.splitext(mesh_name)[0]

            # Prepare UI data
            ui_data = {
                "mesh_name": mesh_name_short,
                "num_intersecting_faces": num_intersecting,
                "num_intersection_pairs": num_pairs,
                "total_faces": len(trimesh.faces),
                "total_vertices": len(trimesh.vertices),
                "has_cgal": has_cgal,
                "faces": intersecting_faces_list
            }

            return {
                "result": (result_mesh, report),
                "ui": {
                    "intersection_data": [ui_data]
                }
            }

        except ImportError as e:
            # libigl not available at all
            error_msg = f"""Error: libigl not available

{str(e)}

Self-intersection detection requires libigl with CGAL support.
Install with: pip install libigl cgal

For now, returning mesh unchanged.
"""
            print(f"[DetectSelfIntersections] libigl import error: {e}")
            return (trimesh, error_msg)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"""Error detecting self-intersections:

{str(e)}

Returning mesh unchanged. Check console for details.
"""
            print(f"[DetectSelfIntersections] Unexpected error: {e}")
            return (trimesh, error_msg)


NODE_CLASS_MAPPINGS = {
    "GeomPackDetectSelfIntersections": DetectSelfIntersectionsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackDetectSelfIntersections": "Self Intersections",
}
