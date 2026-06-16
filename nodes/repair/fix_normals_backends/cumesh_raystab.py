# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""CuMesh GPU ray-stabbing fix-normals backend node.

Uses cuBVH.signed_distance(mode='raystab') as the inside/outside sign oracle.
raystab parity is winding-INDEPENDENT, so it gives the correct geometric sign even
on meshes whose input winding is broken -- unlike mode='watertight', which reads its
sign from the existing (possibly wrong) face orientation and is therefore useless
for fixing normals. CUDA-only; fast on large meshes.
"""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FixNormalsCuMeshRaystabNode(io.ComfyNode):
    """Fix face orientation on the GPU via cuBVH ray-stabbing signed distance."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFixNormals_CuMeshRaystab",
            display_name="Fix Normals CuMesh Raystab (backend)",
            category="geompack/repair",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("rs_eps", default=1e-4, min=1e-7, max=1e-1, step=1e-5, display_mode="number", tooltip="Probe offset as a fraction of the bbox diagonal. Each face centroid is pushed this far along its normal before the inside/outside test. Too small = the probe lands on the surface (ambiguous); too large = it can cross into a neighbouring shell."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="fixed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, rs_eps=1e-4):
        import torch
        from cumesh.bvh import cuBVH
        import comfy.model_management

        device = comfy.model_management.get_torch_device()
        assert device.type == "cuda", (
            f"CuMesh raystab requires CUDA but got device '{device}' -- cumesh is GPU-only, no CPU fallback")

        fixed_mesh = trimesh.copy()
        was_consistent = fixed_mesh.is_winding_consistent

        V = np.ascontiguousarray(fixed_mesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(fixed_mesh.faces, dtype=np.int64)

        # cuBVH requires > 8 triangles; bail gracefully on tiny meshes.
        if len(F) <= 8:
            info = (f"Normal Orientation Fix:\n\nMethod: cumesh_raystab (GPU)\n"
                    f"SKIPPED: mesh has {len(F)} faces (cuBVH needs > 8). Returned unchanged.\n")
            log.warning("cumesh_raystab: %d faces <= 8, cuBVH unavailable; returning unchanged", len(F))
            return io.NodeOutput(fixed_mesh, info, ui={"text": [info]})

        # Per-face normals (numpy; direction only, then normalized for a clean offset).
        tri = V[F]
        face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nlen = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / np.maximum(nlen, 1e-20)
        face_centroids = tri.mean(axis=1)

        bbox_diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
        eps = float(rs_eps) * bbox_diag

        # Probe just outside each face along its normal. If the normal is correct
        # (outward) the probe is OUTSIDE -> signed distance > 0; if it points inward
        # the probe lands inside -> distance < 0 -> flip the face.
        query = (face_centroids + face_normals * eps).astype(np.float32)

        bvh = cuBVH(V.astype(np.float32), F.astype(np.int32))
        S, _fid, _ = bvh.signed_distance(torch.from_numpy(query), mode='raystab')
        S = S.detach().cpu().numpy()

        flip_mask = S < 0.0
        F_out = F.copy()
        F_out[flip_mask] = F_out[flip_mask][:, [0, 2, 1]]
        num_flipped = int(np.sum(flip_mask))
        fixed_mesh.faces = F_out

        del bvh
        comfy.model_management.soft_empty_cache()

        is_consistent = fixed_mesh.is_winding_consistent
        info = (
            f"Normal Orientation Fix:\n"
            f"\n"
            f"Method: cumesh_raystab (GPU, ray-stabbing signed distance)\n"
            f"Before: {'Consistent' if was_consistent else 'Inconsistent'}\n"
            f"After:  {'Consistent' if is_consistent else 'Inconsistent'}\n"
            f"Faces Flipped: {num_flipped}\n"
            f"\n"
            f"Vertices: {len(fixed_mesh.vertices):,}\n"
            f"Faces: {len(fixed_mesh.faces):,}\n"
            f"\n"
            f"Note: raystab parity is winding-independent (robust on broken input)\n"
            f"{'[OK] Normals are now consistently oriented!' if is_consistent else '[WARN] Some inconsistencies may remain (check mesh topology)'}"
        )
        log.info("cumesh_raystab: flipped %d/%d faces, %s -> %s",
                 num_flipped, len(F), was_consistent, is_consistent)

        return io.NodeOutput(fixed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFixNormals_CuMeshRaystab": FixNormalsCuMeshRaystabNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFixNormals_CuMeshRaystab": "Fix Normals CuMesh Raystab (backend)"}
