# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remove degenerate faces (zero area or duplicate vertex indices) from mesh.

Degenerate faces can be created by:
- Vertex merging when two vertices of a triangle merge to the same index
- OCC meshing creating sliver triangles at CAD face boundaries
- Import from poorly-constructed mesh files
"""

import numpy as np


class RemoveDegenerateFacesNode:
    """
    Remove degenerate faces from a mesh.

    A face is considered degenerate if:
    - It has duplicate vertex indices (e.g., [0, 1, 1])
    - It has zero or near-zero area

    This is useful for cleaning meshes after vertex merging operations,
    or for fixing meshes imported from CAD systems that create sliver triangles.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
            },
            "optional": {
                "min_area": ("FLOAT", {
                    "default": 1e-10,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 1e-10,
                    "tooltip": "Minimum face area threshold (faces below this are removed)"
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("cleaned_mesh", "info")
    FUNCTION = "remove_degenerate"
    CATEGORY = "geompack/repair"

    def remove_degenerate(self, mesh, min_area=1e-10):
        """
        Remove degenerate faces from mesh.

        Args:
            mesh: Input trimesh.Trimesh object
            min_area: Minimum face area threshold

        Returns:
            tuple: (cleaned_mesh, info_string)
        """
        print(f"[RemoveDegenerateFaces] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        faces_before = len(mesh.faces)
        verts_before = len(mesh.vertices)

        # Create a copy to avoid modifying original
        cleaned_mesh = mesh.copy()

        # Method 1: Remove faces with duplicate vertex indices
        # (e.g., [0, 1, 1] where two vertices are the same)
        duplicate_mask = np.array([len(set(f)) == 3 for f in cleaned_mesh.faces])
        num_duplicate = np.sum(~duplicate_mask)

        if num_duplicate > 0:
            print(f"[RemoveDegenerateFaces] Found {num_duplicate} faces with duplicate vertex indices")
            cleaned_mesh.update_faces(duplicate_mask)

        # Method 2: Remove faces with zero/tiny area using trimesh's built-in
        if hasattr(cleaned_mesh, 'nondegenerate_faces'):
            area_mask = cleaned_mesh.nondegenerate_faces()
            num_zero_area = np.sum(~area_mask)
            if num_zero_area > 0:
                print(f"[RemoveDegenerateFaces] Found {num_zero_area} faces with zero area")
                cleaned_mesh.update_faces(area_mask)

        # Method 3: Remove faces below min_area threshold
        if min_area > 0:
            face_areas = cleaned_mesh.area_faces
            area_threshold_mask = face_areas >= min_area
            num_tiny = np.sum(~area_threshold_mask)
            if num_tiny > 0:
                print(f"[RemoveDegenerateFaces] Found {num_tiny} faces below area threshold {min_area:.2e}")
                cleaned_mesh.update_faces(area_threshold_mask)

        # Remove unreferenced vertices
        cleaned_mesh.remove_unreferenced_vertices()

        faces_after = len(cleaned_mesh.faces)
        verts_after = len(cleaned_mesh.vertices)
        faces_removed = faces_before - faces_after
        verts_removed = verts_before - verts_after

        # Build info string
        info = f"""Degenerate Face Removal Results:

Before:
  Vertices: {verts_before:,}
  Faces: {faces_before:,}

After:
  Vertices: {verts_after:,} ({-verts_removed:+,})
  Faces: {faces_after:,} ({-faces_removed:+,})

{'✓ Removed ' + str(faces_removed) + ' degenerate faces' if faces_removed > 0 else 'ℹ No degenerate faces found'}
"""

        print(f"[RemoveDegenerateFaces] Removed {faces_removed} degenerate faces, {verts_removed} unreferenced vertices")

        return (cleaned_mesh, info)


NODE_CLASS_MAPPINGS = {
    "GeomPackRemoveDegenerateFaces": RemoveDegenerateFacesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemoveDegenerateFaces": "Remove Degenerate Faces",
}
