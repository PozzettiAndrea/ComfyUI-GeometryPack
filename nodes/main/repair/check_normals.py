# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Analyze mesh normal consistency and quality.
"""

import numpy as np
import trimesh


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


NODE_CLASS_MAPPINGS = {
    "GeomPackCheckNormals": CheckNormalsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackCheckNormals": "Check Normals",
}
