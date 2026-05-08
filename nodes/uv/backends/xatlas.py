# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""XAtlas UV unwrapping backend node."""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class UVXatlasNode(io.ComfyNode):
    """XAtlas automatic UV unwrapping backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackUV_Xatlas",
            display_name="UV Xatlas (backend)",
            category="geompack/uv",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="unwrapped_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh):
        """XAtlas automatic UV unwrapping."""
        try:
            import xatlas
        except ImportError:
            raise ImportError(
                "xatlas not installed. Install with: pip install xatlas\n"
                "Required for fast UV unwrapping without Blender."
            )

        log.info("Backend: xatlas")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))

        # Core unwrapping logic
        vmapping, indices, uvs = xatlas.parametrize(trimesh.vertices, trimesh.faces)
        new_vertices = trimesh.vertices[vmapping]

        unwrapped = trimesh_module.Trimesh(
            vertices=new_vertices,
            faces=indices,
            process=False
        )

        from trimesh.visual import TextureVisuals
        unwrapped.visual = TextureVisuals(uv=uvs)

        # Preserve metadata
        unwrapped.metadata = trimesh.metadata.copy()
        unwrapped.metadata['uv_unwrap'] = {
            'algorithm': 'xatlas',
            'original_vertices': len(trimesh.vertices),
            'unwrapped_vertices': len(new_vertices),
            'vertex_duplication_ratio': len(new_vertices) / len(trimesh.vertices)
        }

        log.info("Output: %d vertices, %d faces", len(unwrapped.vertices), len(unwrapped.faces))

        info = f"""UV Unwrap Results (XAtlas):

Algorithm: XAtlas automatic unwrapping
Optimized for: Lightmaps and texture atlasing

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(new_vertices):,}
  Faces: {len(unwrapped.faces):,}
  Vertex Duplication: {len(new_vertices)/len(trimesh.vertices):.2f}x

Fast automatic UV unwrapping with vertex splitting at seams.
"""
        return io.NodeOutput(unwrapped, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackUV_Xatlas": UVXatlasNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackUV_Xatlas": "UV Xatlas (backend)"}
