# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""IGL BFS-orient fix normals backend node."""

import logging
import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FixNormalsIglBfsNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFixNormals_IglBfs",
            display_name="Fix Normals IGL BFS (backend)",
            category="geompack/repair",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="fixed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh):
        import igl

        fixed_mesh = trimesh.copy()

        was_consistent = fixed_mesh.is_winding_consistent

        V = np.asarray(fixed_mesh.vertices, dtype=np.float64)
        F = np.asarray(fixed_mesh.faces, dtype=np.int64)

        FF, C = igl.bfs_orient(F)

        fixed_mesh.faces = FF

        num_components = len(np.unique(C))
        log.info("igl.bfs_orient: %d orientation components", num_components)

        is_consistent = fixed_mesh.is_winding_consistent

        info = (
            f"Normal Orientation Fix:\n"
            f"\n"
            f"Method: igl_bfs\n"
            f"Before: {'Consistent' if was_consistent else 'Inconsistent'}\n"
            f"After:  {'Consistent' if is_consistent else 'Inconsistent'}\n"
            f"Orientation Components: {num_components}\n"
            f"\n"
            f"Vertices: {len(fixed_mesh.vertices):,}\n"
            f"Faces: {len(fixed_mesh.faces):,}\n"
            f"\n"
            f"Note: BFS makes faces consistent but doesn't determine inside/outside\n"
            f"{'[OK] Normals are now consistently oriented!' if is_consistent else '[WARN] Some inconsistencies may remain (check mesh topology)'}"
        )

        log.info("Normal orientation: %s -> %s", was_consistent, is_consistent)

        return io.NodeOutput(fixed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFixNormals_IglBfs": FixNormalsIglBfsNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFixNormals_IglBfs": "Fix Normals IGL BFS (backend)"}
