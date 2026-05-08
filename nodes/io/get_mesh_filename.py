# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Get Mesh Filename Node - Extract filename from mesh metadata.

Similar to CADGetFilename in CADabra, this extracts the original filename
from meshes loaded via LoadMesh or LoadMeshBatch.
"""

import logging
import os
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class GetMeshFilename(io.ComfyNode):
    """
    Extract filename (without extension) from a mesh's metadata.

    Returns the original filename that was used when loading the mesh file.
    Useful for batch processing to preserve original names in output.

    Supports batch processing: input a list of meshes, get a list of filenames.
    """

    INPUT_IS_LIST = True


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackGetMeshFilename",
            display_name="Get Mesh Filename",
            category="geompack/io",
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
            ],
            outputs=[
                io.String.Output(display_name="filename", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, mesh):
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

        log.info("Extracted %d filename(s)", len(filenames))
        return io.NodeOutput(filenames)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeomPackGetMeshFilename": GetMeshFilename,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackGetMeshFilename": "Get Mesh Filename",
}
