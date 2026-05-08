# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""IGL raycast fix normals backend node."""

import logging
import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FixNormalsIglRaycastNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFixNormals_IglRaycast",
            display_name="Fix Normals IGL Raycast (backend)",
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

        # Small offset to avoid self-intersection
        eps = 1e-6

        flip_mask = np.zeros(len(F), dtype=bool)

        for i in range(len(F)):
            origin = face_centroids[i] + face_normals[i] * eps
            direction = face_normals[i]

            # Cast ray and get hits
            hits = igl.ray_mesh_intersect(
                np.ascontiguousarray(origin, dtype=np.float64),
                np.ascontiguousarray(direction, dtype=np.float64),
                V, F
            )

            # Odd number of hits = pointing inward
            if hits is not None and len(hits) % 2 == 1:
                flip_mask[i] = True

        # Flip faces by reversing vertex order
        F_out = F.copy()
        F_out[flip_mask] = F_out[flip_mask][:, [0, 2, 1]]

        num_flipped = int(np.sum(flip_mask))
        fixed_mesh.faces = F_out

        log.info("igl_raycast: flipped %d/%d faces", num_flipped, len(F))

        is_consistent = fixed_mesh.is_winding_consistent

        info = (
            f"Normal Orientation Fix:\n"
            f"\n"
            f"Method: igl_raycast\n"
            f"Before: {'Consistent' if was_consistent else 'Inconsistent'}\n"
            f"After:  {'Consistent' if is_consistent else 'Inconsistent'}\n"
            f"Faces Flipped: {num_flipped}\n"
            f"\n"
            f"Vertices: {len(fixed_mesh.vertices):,}\n"
            f"Faces: {len(fixed_mesh.faces):,}\n"
            f"\n"
            f"Note: Raycasting works best on closed meshes without self-intersections\n"
            f"{'[OK] Normals are now consistently oriented!' if is_consistent else '[WARN] Some inconsistencies may remain (check mesh topology)'}"
        )

        log.info("Normal orientation: %s -> %s", was_consistent, is_consistent)

        return io.NodeOutput(fixed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFixNormals_IglRaycast": FixNormalsIglRaycastNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFixNormals_IglRaycast": "Fix Normals IGL Raycast (backend)"}
