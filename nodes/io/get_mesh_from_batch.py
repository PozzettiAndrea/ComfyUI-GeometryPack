# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Get Mesh From Batch - extract a single mesh from a TRIMESH batch by index.

The list-handling counterpart of Load Mesh Batch: takes the batch (a list of
trimesh objects) and an index, returns just that mesh so it can feed any
single-mesh node. The index is clamped to the batch, so out-of-range values
select the last mesh instead of erroring.
"""

import logging

from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class GetMeshFromBatch(io.ComfyNode):
    """Extract one mesh from a TRIMESH batch by index."""

    INPUT_IS_LIST = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackGetMeshFromBatch",
            display_name="Get Mesh From Batch",
            category="geompack/io",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("index", default=0, min=0, max=10000),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, index):
        # Inputs arrive as lists (INPUT_IS_LIST=True); the batch itself is `trimesh`.
        index_val = index[0] if isinstance(index, list) else index
        if not trimesh:
            raise ValueError("Empty mesh batch provided")
        actual = max(0, min(index_val, len(trimesh) - 1))
        mesh = trimesh[actual]
        md = getattr(mesh, "metadata", None) or {}
        name = md.get("file_name") or ""
        log.info("Get Mesh From Batch: index %d/%d%s", actual + 1, len(trimesh),
                 f" ({name})" if name else "")
        return io.NodeOutput(mesh)


NODE_CLASS_MAPPINGS = {
    "GeomPackGetMeshFromBatch": GetMeshFromBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackGetMeshFromBatch": "Get Mesh From Batch",
}
