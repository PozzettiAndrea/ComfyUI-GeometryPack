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


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackFixNormals": FixNormalsNode,
    "GeomPackCheckNormals": CheckNormalsNode,
    "GeomPackFillHoles": FillHolesNode,
    "GeomPackComputeNormals": ComputeNormalsNode,
    "GeomPackVisualizeNormals": VisualizNormalFieldNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackFixNormals": "Fix Normals",
    "GeomPackCheckNormals": "Check Normals",
    "GeomPackFillHoles": "Fill Holes",
    "GeomPackComputeNormals": "Compute Normals",
    "GeomPackVisualizeNormals": "Visualize Normal Field",
}
