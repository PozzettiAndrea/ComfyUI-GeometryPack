# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Get Mesh Filename Node - Extract filename from mesh metadata.

Similar to CADGetFilename in CADabra, this extracts the original filename
from meshes loaded via LoadMesh or LoadMeshBatch.
"""

import os


class GetMeshFilename:
    """
    Extract filename (without extension) from a mesh's metadata.

    Returns the original filename that was used when loading the mesh file.
    Useful for batch processing to preserve original names in output.

    Supports batch processing: input a list of meshes, get a list of filenames.
    """

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "get_filename"
    CATEGORY = "geompack/io"

    def get_filename(self, mesh):
        """
        Extract filename from mesh metadata.

        Args:
            mesh: Input trimesh object(s)

        Returns:
            tuple: (list of filenames without extension)
        """
        # Handle both single and batch inputs
        meshes = mesh if isinstance(mesh, list) else [mesh]

        filenames = []
        for m in meshes:
            # Get filename from metadata (set by load_mesh_file)
            name = m.metadata.get('file_name', '') if hasattr(m, 'metadata') else ''
            if name:
                # Remove extension for cleaner output
                name = os.path.splitext(name)[0]
            else:
                name = "unknown"
            filenames.append(name)

        print(f"[GetMeshFilename] Extracted {len(filenames)} filename(s)")
        return (filenames,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeomPackGetMeshFilename": GetMeshFilename,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackGetMeshFilename": "Get Mesh Filename",
}
