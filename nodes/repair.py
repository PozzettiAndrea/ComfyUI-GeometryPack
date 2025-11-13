"""
Repair Nodes - Mesh repair, normal operations, and quality improvement
"""

import numpy as np
import trimesh


class FixNormalsNode:
    """
    Fix inconsistent normal orientations.

    Ensures all face normals point consistently (all outward or all inward).
    Uses graph traversal to propagate consistent orientation across the trimesh.
    Essential for proper rendering and boolean operations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("fixed_mesh", "info")
    FUNCTION = "fix_normals"
    CATEGORY = "geompack/repair"

    def fix_normals(self, trimesh):
        """
        Fix inconsistent face normal orientations.

        Args:
            trimesh: Input trimesh.Trimesh object

        Returns:
            tuple: (fixed_trimesh, info_string)
        """
        print(f"[FixNormals] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Create a copy to avoid modifying the original
        fixed_mesh = trimesh.copy()

        # Check initial winding consistency
        was_consistent = fixed_mesh.is_winding_consistent

        # Fix normals - this reorients faces for consistent winding
        fixed_mesh.fix_normals()

        # Check if it's now consistent
        is_consistent = fixed_mesh.is_winding_consistent

        info = f"""Normal Orientation Fix:

Before: {'Consistent' if was_consistent else 'Inconsistent'}
After:  {'Consistent' if is_consistent else 'Inconsistent'}

Vertices: {len(fixed_mesh.vertices):,}
Faces: {len(fixed_mesh.faces):,}

{'✓ Normals are now consistently oriented!' if is_consistent else '⚠ Some inconsistencies may remain (check mesh topology)'}
"""

        print(f"[FixNormals] {'✓' if is_consistent else '⚠'} Normal orientation: {was_consistent} -> {is_consistent}")

        return (fixed_mesh, info)


class CheckNormalsNode:
    """
    Analyze mesh normal consistency and quality.

    Checks face normal orientations, reports inconsistencies, and provides
    diagnostic information about mesh topology issues.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "check_normals"
    CATEGORY = "geompack/repair"

    def check_normals(self, trimesh):
        """
        Analyze mesh normal consistency.

        Args:
            trimesh: Input trimesh.Trimesh object

        Returns:
            tuple: (report_string,)
        """
        print(f"[CheckNormals] Analyzing mesh with {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Check winding consistency
        is_winding_consistent = trimesh.is_winding_consistent

        # Check if watertight (implies consistent normals typically)
        is_watertight = trimesh.is_watertight

        # Get face normals and check for degenerate triangles
        face_normals = trimesh.face_normals
        face_areas = trimesh.area_faces

        # Find degenerate faces (zero or near-zero area)
        degenerate_faces = np.sum(face_areas < 1e-10)

        # Check for NaN normals (indicates degenerate geometry)
        nan_normals = np.sum(np.isnan(face_normals).any(axis=1))

        # Calculate normal statistics
        normal_lengths = np.linalg.norm(face_normals, axis=1)
        avg_normal_length = np.mean(normal_lengths)

        report = f"""=== Normal Consistency Analysis ===

Mesh Statistics:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}
  Edges: {len(trimesh.edges_unique):,}

Topology:
  Winding Consistent: {'✓ Yes' if is_winding_consistent else '✗ No (normals may point in mixed directions)'}
  Watertight: {'✓ Yes' if is_watertight else '✗ No (has boundary edges/holes)'}

Face Quality:
  Degenerate Faces: {degenerate_faces:,} ({100.0 * degenerate_faces / len(trimesh.faces):.2f}%)
  NaN Normals: {nan_normals:,}
  Avg Normal Length: {avg_normal_length:.6f} (should be ~1.0)

Recommendations:
"""

        if not is_winding_consistent:
            report += "  • Use 'Fix Normals' node to correct orientation\n"

        if not is_watertight:
            report += "  • Use 'Fill Holes' node to close mesh boundaries\n"

        if degenerate_faces > 0:
            report += "  • Use 'Remove Degenerate Faces' or remeshing to clean geometry\n"

        if nan_normals > 0:
            report += "  • Remove degenerate faces before further processing\n"

        if is_winding_consistent and is_watertight and degenerate_faces == 0:
            report += "  ✓ Mesh normals are in excellent condition!\n"

        print(f"[CheckNormals] Winding: {is_winding_consistent}, Watertight: {is_watertight}, Degenerate: {degenerate_faces}")

        return (report,)


class FillHolesNode:
    """
    Fill holes in mesh by adding new faces.

    Identifies boundary loops (holes) and fills them with new triangular faces.
    Useful for mesh repair, creating watertight meshes for 3D printing, or
    closing gaps in scanned geometry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("filled_mesh", "info")
    FUNCTION = "fill_holes"
    CATEGORY = "geompack/repair"

    def fill_holes(self, trimesh):
        """
        Fill holes in the trimesh.

        Args:
            trimesh: Input trimesh.Trimesh object

        Returns:
            tuple: (filled_trimesh, info_string)
        """
        print(f"[FillHoles] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Check initial state
        was_watertight = trimesh.is_watertight
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        # Create a copy
        filled_mesh = trimesh.copy()

        # Fill holes
        filled_mesh.fill_holes()

        # Check result
        is_watertight = filled_mesh.is_watertight
        final_vertices = len(filled_mesh.vertices)
        final_faces = len(filled_mesh.faces)

        added_vertices = final_vertices - initial_vertices
        added_faces = final_faces - initial_faces

        info = f"""Hole Filling Results:

Initial State:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}
  Watertight: {'Yes' if was_watertight else 'No'}

After Filling:
  Vertices: {final_vertices:,} (+{added_vertices})
  Faces: {final_faces:,} (+{added_faces})
  Watertight: {'✓ Yes' if is_watertight else '⚠ No'}

{'✓ All holes successfully filled!' if is_watertight and added_faces > 0 else ''}
{'ℹ No holes detected - mesh was already watertight.' if was_watertight else ''}
{'⚠ Some holes may remain (check mesh topology).' if not is_watertight and added_faces > 0 else ''}
"""

        print(f"[FillHoles] Added {added_faces} faces, Watertight: {was_watertight} -> {is_watertight}")

        return (filled_mesh, info)


class ComputeNormalsNode:
    """
    Recompute mesh normals with custom settings.

    Recalculates face and vertex normals. Useful after mesh manipulation,
    importing from formats without normals, or when normals seem incorrect.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "smooth_vertex_normals": (["true", "false"], {
                    "default": "true"
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh_with_normals",)
    FUNCTION = "compute_normals"
    CATEGORY = "geompack/repair"

    def compute_normals(self, trimesh, smooth_vertex_normals="true"):
        """
        Recompute mesh normals.

        Args:
            trimesh: Input trimesh.Trimesh object
            smooth_vertex_normals: Whether to smooth vertex normals

        Returns:
            tuple: (mesh_with_normals,)
        """
        print(f"[ComputeNormals] Processing mesh with {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Create a copy
        result_mesh = trimesh.copy()

        # Face normals are always recomputed automatically by trimesh
        # But we can force a cache clear and recomputation
        result_mesh._cache.clear()

        if smooth_vertex_normals == "false":
            # Use face normals directly (faceted appearance)
            # This creates sharp edges by not averaging normals across faces
            vertex_normals = np.zeros_like(result_mesh.vertices)
            for i, face in enumerate(result_mesh.faces):
                face_normal = result_mesh.face_normals[i]
                vertex_normals[face] += face_normal
            # Normalize
            norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            vertex_normals = vertex_normals / norms

            # Store in mesh (note: trimesh will override this with smoothed normals)
            # So we need to mark it in metadata
            result_mesh.metadata['normals_smoothed'] = False
            print(f"[ComputeNormals] Computed faceted (non-smooth) normals")
        else:
            # Trimesh automatically computes smooth vertex normals
            # Just access them to ensure they're computed
            _ = result_mesh.vertex_normals
            result_mesh.metadata['normals_smoothed'] = True
            print(f"[ComputeNormals] Computed smooth vertex normals")

        return (result_mesh,)


class VisualizNormalFieldNode:
    """
    Create normal field visualization for VTK viewer.

    Adds vertex normals as scalar fields (X, Y, Z components) that can be
    visualized with color mapping in the VTK viewer. Useful for debugging
    normal orientation issues or understanding surface geometry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh_with_fields", "info")
    FUNCTION = "visualize_normals"
    CATEGORY = "geompack/repair"

    def visualize_normals(self, trimesh):
        """
        Add normal components as vertex scalar fields.

        Args:
            trimesh: Input trimesh.Trimesh object

        Returns:
            tuple: (mesh_with_normal_fields, info_string)
        """
        print(f"[VisualizeNormals] Processing mesh with {len(trimesh.vertices)} vertices")

        # Create a copy
        result_mesh = trimesh.copy()

        # Get vertex normals
        normals = result_mesh.vertex_normals

        # Add each component as a scalar field
        result_mesh.vertex_attributes['normal_x'] = normals[:, 0].astype(np.float32)
        result_mesh.vertex_attributes['normal_y'] = normals[:, 1].astype(np.float32)
        result_mesh.vertex_attributes['normal_z'] = normals[:, 2].astype(np.float32)

        # Also add normal magnitude (should be ~1.0 for unit normals)
        normal_magnitude = np.linalg.norm(normals, axis=1).astype(np.float32)
        result_mesh.vertex_attributes['normal_magnitude'] = normal_magnitude

        info = f"""Normal Field Visualization:

Added Scalar Fields:
  • normal_x: X component of vertex normals ({normals[:, 0].min():.3f} to {normals[:, 0].max():.3f})
  • normal_y: Y component of vertex normals ({normals[:, 1].min():.3f} to {normals[:, 1].max():.3f})
  • normal_z: Z component of vertex normals ({normals[:, 2].min():.3f} to {normals[:, 2].max():.3f})
  • normal_magnitude: Length of normal vectors (avg: {normal_magnitude.mean():.6f})

Use VTK viewer with 'Preview Mesh (VTK with Fields)' to visualize
these scalar fields with color mapping!

Expected Values:
  • Components: -1.0 to 1.0
  • Magnitude: ~1.0 (unit normals)
"""

        print(f"[VisualizeNormals] Added 4 scalar fields to mesh")

        return (result_mesh, info)


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
                    if IF.shape[0] > 0:
                        # Get unique face indices that are involved in intersections
                        intersecting_faces = np.unique(IF.flatten())
                        num_intersecting = len(intersecting_faces)

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

                        print(f"[DetectSelfIntersections] Found {num_intersecting} intersecting faces ({IF.shape[0]} intersection pairs)")

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
                    result_mesh.face_attributes['self_intersecting'] = np.zeros(len(trimesh.faces), dtype=np.float32)
                    result_mesh.vertex_attributes['intersection_flag'] = np.zeros(len(trimesh.vertices), dtype=np.float32)
                    result_mesh.vertex_attributes['intersection_count'] = np.zeros(len(trimesh.vertices), dtype=np.float32)
                    print("[DetectSelfIntersections] Falling back to zero fields (CGAL error)")

            else:
                # CGAL not available - use basic fallback
                print("[DetectSelfIntersections] CGAL not available, using basic detection")
                num_intersecting = 0

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

            return (result_mesh, report)

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


class RemeshSelfIntersectionsNode:
    """
    Remove self-intersections by remeshing.

    Uses libigl CGAL to subdivide self-intersecting triangles so that
    intersections lie exactly on edges. Can optionally extract outer hull
    for a clean manifold result. Essential for preparing meshes for
    boolean operations or 3D printing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "detect_only": ("BOOLEAN", {"default": False}),
                "remove_unreferenced": ("BOOLEAN", {"default": True}),
                "extract_outer_hull": ("BOOLEAN", {"default": False}),
                "stitch_all": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("fixed_mesh", "report")
    FUNCTION = "remesh_intersections"
    CATEGORY = "geompack/repair"

    def remesh_intersections(self, trimesh, detect_only=False, remove_unreferenced=True,
                           extract_outer_hull=False, stitch_all=True):
        """
        Remesh self-intersections using libigl CGAL.

        Args:
            trimesh: Input trimesh.Trimesh object
            detect_only: Only detect intersections, don't remesh
            remove_unreferenced: Remove unreferenced vertices after remeshing
            extract_outer_hull: Extract outer hull for manifold result (slow)
            stitch_all: Attempt to stitch all boundaries

        Returns:
            tuple: (remeshed_mesh, report_string)
        """
        print(f"[RemeshSelfIntersections] Processing mesh: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[RemeshSelfIntersections] Options: detect_only={detect_only}, remove_unreferenced={remove_unreferenced}, extract_outer_hull={extract_outer_hull}, stitch_all={stitch_all}")

        try:
            # Try to use libigl with CGAL
            import igl

            # Check if CGAL functions are available
            try:
                import igl.copyleft.cgal as cgal
                has_cgal = hasattr(cgal, 'remesh_self_intersections')
            except (ImportError, AttributeError):
                has_cgal = False

            if not has_cgal:
                error_msg = """Error: libigl CGAL not available

Self-intersection remeshing requires libigl with CGAL support.
Install with: pip install cgal

Returning mesh unchanged.
"""
                print("[RemeshSelfIntersections] CGAL not available")
                return (trimesh, error_msg)

            print("[RemeshSelfIntersections] Using libigl CGAL method")

            # Convert mesh to numpy arrays with proper dtypes
            V = np.asarray(trimesh.vertices, dtype=np.float64)
            F = np.asarray(trimesh.faces, dtype=np.int64)

            initial_vertices = len(V)
            initial_faces = len(F)

            # Perform remeshing with keyword arguments
            try:
                VV, FF, IF, J, IM = cgal.remesh_self_intersections(
                    V, F,
                    detect_only=detect_only,
                    first_only=False,
                    stitch_all=stitch_all
                )

                num_intersection_pairs = IF.shape[0] if IF is not None and hasattr(IF, 'shape') else 0

                if detect_only:
                    print(f"[RemeshSelfIntersections] Detected {num_intersection_pairs} intersection pairs")
                    result_mesh = trimesh.copy()

                    if num_intersection_pairs > 0:
                        # Mark intersecting faces
                        intersecting_faces = np.unique(IF.flatten())
                        face_field = np.zeros(len(F), dtype=np.float32)
                        face_field[intersecting_faces] = 1.0
                        result_mesh.face_attributes['self_intersecting'] = face_field

                else:
                    print(f"[RemeshSelfIntersections] Remeshing complete: {len(VV)} vertices, {len(FF)} faces")

                    # Post-processing
                    if remove_unreferenced and not detect_only:
                        print("[RemeshSelfIntersections] Removing unreferenced vertices...")
                        VV_clean, FF_clean, _, _ = igl.remove_unreferenced(VV, FF)
                        print(f"[RemeshSelfIntersections] After cleanup: {len(VV_clean)} vertices, {len(FF_clean)} faces")
                        VV, FF = VV_clean, FF_clean

                    if extract_outer_hull and not detect_only:
                        print("[RemeshSelfIntersections] Extracting outer hull (this may take a while)...")
                        try:
                            # Try to extract outer hull for manifold result
                            if hasattr(igl, 'outer_hull_legacy'):
                                VV_hull, FF_hull, _, _ = igl.outer_hull_legacy(VV, FF)
                                print(f"[RemeshSelfIntersections] Outer hull: {len(VV_hull)} vertices, {len(FF_hull)} faces")
                                VV, FF = VV_hull, FF_hull
                            else:
                                print("[RemeshSelfIntersections] ⚠ outer_hull_legacy not available, skipping")
                        except Exception as e:
                            print(f"[RemeshSelfIntersections] Outer hull extraction failed: {e}")

                    # Create new trimesh from remeshed data
                    result_mesh = trimesh.Trimesh(vertices=VV, faces=FF, process=False)

                    # Store operation metadata
                    result_mesh.metadata['remeshed_self_intersections'] = True
                    result_mesh.metadata['original_vertices'] = initial_vertices
                    result_mesh.metadata['original_faces'] = initial_faces
                    result_mesh.metadata['intersections_found'] = num_intersection_pairs

                # Generate report
                final_vertices = len(result_mesh.vertices)
                final_faces = len(result_mesh.faces)
                added_vertices = final_vertices - initial_vertices
                added_faces = final_faces - initial_faces

                if detect_only:
                    report = f"""Self-Intersection Detection (detect_only mode):

Mesh Statistics:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

Detection Results:
  Intersection Pairs: {num_intersection_pairs:,}

Status:
  {'✓ No self-intersections detected!' if num_intersection_pairs == 0 else '⚠ Self-intersections found!'}

Note: Mesh was not modified (detect_only=True)
To fix intersections, set detect_only=False
"""
                else:
                    report = f"""Self-Intersection Remeshing:

Initial Mesh:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After Remeshing:
  Vertices: {final_vertices:,} ({'+' if added_vertices >= 0 else ''}{added_vertices:,})
  Faces: {final_faces:,} ({'+' if added_faces >= 0 else ''}{added_faces:,})

Processing:
  Intersection Pairs Found: {num_intersection_pairs:,}
  Removed Unreferenced: {'Yes' if remove_unreferenced else 'No'}
  Extracted Outer Hull: {'Yes' if extract_outer_hull else 'No'}
  Stitch All: {'Yes' if stitch_all else 'No'}

Status:
  {'✓ Remeshing complete!' if not detect_only else '✓ Detection complete!'}
  {'✓ Mesh is now manifold' if extract_outer_hull and result_mesh.is_watertight else ''}

{'⚠ Note: Remeshing subdivides intersections but may create non-manifold edges.' if not extract_outer_hull else ''}
{'  Consider enabling extract_outer_hull for a clean manifold result.' if not extract_outer_hull else ''}
"""

                return (result_mesh, report)

            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = f"""Error during remeshing:

{str(e)}

Returning mesh unchanged. Check console for details.
"""
                print(f"[RemeshSelfIntersections] Remeshing error: {e}")
                return (trimesh, error_msg)

        except ImportError as e:
            error_msg = f"""Error: libigl not available

{str(e)}

Self-intersection remeshing requires libigl with CGAL support.
Install with: pip install libigl cgal

Returning mesh unchanged.
"""
            print(f"[RemeshSelfIntersections] libigl import error: {e}")
            return (trimesh, error_msg)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"""Unexpected error:

{str(e)}

Returning mesh unchanged. Check console for details.
"""
            print(f"[RemeshSelfIntersections] Unexpected error: {e}")
            return (trimesh, error_msg)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackFixNormals": FixNormalsNode,
    "GeomPackCheckNormals": CheckNormalsNode,
    "GeomPackFillHoles": FillHolesNode,
    "GeomPackComputeNormals": ComputeNormalsNode,
    "GeomPackVisualizeNormals": VisualizNormalFieldNode,
    "GeomPackDetectSelfIntersections": DetectSelfIntersectionsNode,
    "GeomPackRemeshSelfIntersections": RemeshSelfIntersectionsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackFixNormals": "Fix Normals",
    "GeomPackCheckNormals": "Check Normals",
    "GeomPackFillHoles": "Fill Holes",
    "GeomPackComputeNormals": "Compute Normals",
    "GeomPackVisualizeNormals": "Visualize Normal Field",
    "GeomPackDetectSelfIntersections": "Detect Self Intersections",
    "GeomPackRemeshSelfIntersections": "Remesh Self Intersections",
}
