# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""IGL winding number fix normals backend node."""

import logging
import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FixNormalsIglWindingNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFixNormals_IglWinding",
            display_name="Fix Normals IGL Winding (backend)",
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

        # Compute face normals
        face_normals = igl.per_face_normals(V, F, np.array([1., 1., 1.]))

        # Compute face centroids
        face_centroids = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0

        # Adaptive epsilon based on mesh scale
        bbox_diag = np.linalg.norm(V.max(axis=0) - V.min(axis=0))
        eps = 1e-4 * bbox_diag

        # Query points offset slightly inside (along -normal direction)
        query_inside = face_centroids - face_normals * eps

        # Compute winding numbers
        W = igl.fast_winding_number(V, F, query_inside)

        # If winding < 0.5, the query point is outside, meaning normal points inward
        flip_mask = W < 0.5

        # Flip faces by reversing vertex order
        F_out = F.copy()
        F_out[flip_mask] = F_out[flip_mask][:, [0, 2, 1]]

        num_flipped = int(np.sum(flip_mask))
        fixed_mesh.faces = F_out

        log.info("igl_winding: flipped %d/%d faces", num_flipped, len(F))

        is_consistent = fixed_mesh.is_winding_consistent

        info = (
            f"Normal Orientation Fix:\n"
            f"\n"
            f"Method: igl_winding\n"
            f"Before: {'Consistent' if was_consistent else 'Inconsistent'}\n"
            f"After:  {'Consistent' if is_consistent else 'Inconsistent'}\n"
            f"Faces Flipped: {num_flipped}\n"
            f"\n"
            f"Vertices: {len(fixed_mesh.vertices):,}\n"
            f"Faces: {len(fixed_mesh.faces):,}\n"
            f"\n"
            f"Note: Winding number works best on closed/watertight meshes\n"
            f"{'[OK] Normals are now consistently oriented!' if is_consistent else '[WARN] Some inconsistencies may remain (check mesh topology)'}"
        )

        log.info("Normal orientation: %s -> %s", was_consistent, is_consistent)

        return io.NodeOutput(fixed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFixNormals_IglWinding": FixNormalsIglWindingNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFixNormals_IglWinding": "Fix Normals IGL Winding (backend)"}
