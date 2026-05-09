# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remove degenerate faces (zero area or duplicate vertex indices) from mesh.

Degenerate faces can be created by:
- Vertex merging when two vertices of a triangle merge to the same index
- OCC meshing creating sliver triangles at CAD face boundaries
- Import from poorly-constructed mesh files
"""

import logging
import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemoveDegenerateFacesNode(io.ComfyNode):
    """
    Remove degenerate faces from a mesh.

    A face is considered degenerate if:
    - It has duplicate vertex indices (e.g., [0, 1, 1])
    - It has zero or near-zero area

    This is useful for cleaning meshes after vertex merging operations,
    or for fixing meshes imported from CAD systems that create sliver triangles.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemoveDegenerateFaces",
            display_name="Remove Degenerate Faces",
            category="geompack/repair",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.Float.Input("min_area", default=1e-10, min=0.0, max=1.0, step=1e-10, tooltip="Minimum face area threshold (faces below this are removed)", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="cleaned_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, min_area=1e-10):
        """
        Remove degenerate faces from mesh.

        Args:
            mesh: Input trimesh.Trimesh object
            min_area: Minimum face area threshold

        Returns:
            tuple: (cleaned_mesh, info_string)
        """
        log.info("Input: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))

        faces_before = len(mesh.faces)
        verts_before = len(mesh.vertices)

        # Create a copy to avoid modifying original
        cleaned_mesh = mesh.copy()

        # Method 1: Remove faces with duplicate vertex indices
        # (e.g., [0, 1, 1] where two vertices are the same)
        duplicate_mask = np.array([len(set(f)) == 3 for f in cleaned_mesh.faces])
        num_duplicate = np.sum(~duplicate_mask)

        if num_duplicate > 0:
            log.info("Found %d faces with duplicate vertex indices", num_duplicate)
            cleaned_mesh.update_faces(duplicate_mask)

        # Method 2: Remove faces with zero/tiny area using trimesh's built-in
        if hasattr(cleaned_mesh, 'nondegenerate_faces'):
            area_mask = cleaned_mesh.nondegenerate_faces()
            num_zero_area = np.sum(~area_mask)
            if num_zero_area > 0:
                log.info("Found %d faces with zero area", num_zero_area)
                cleaned_mesh.update_faces(area_mask)

        # Method 3: Remove faces below min_area threshold
        if min_area > 0:
            face_areas = cleaned_mesh.area_faces
            area_threshold_mask = face_areas >= min_area
            num_tiny = np.sum(~area_threshold_mask)
            if num_tiny > 0:
                log.info("Found %d faces below area threshold %.2e", num_tiny, min_area)
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

{'[OK] Removed ' + str(faces_removed) + ' degenerate faces' if faces_removed > 0 else '[INFO] No degenerate faces found'}
"""

        log.info("Removed %d degenerate faces, %d unreferenced vertices", faces_removed, verts_removed)

        return io.NodeOutput(cleaned_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {
    "GeomPackRemoveDegenerateFaces": RemoveDegenerateFacesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemoveDegenerateFaces": "Remove Degenerate Faces",
}
