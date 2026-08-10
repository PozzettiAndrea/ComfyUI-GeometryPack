# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""CuMesh GPU face-winding unification (fix normals) backend node.

Uses CuMesh.unify_face_orientations() -- the GPU winding/orientation fix -- on its
own. Fast on large meshes; CUDA-only.
"""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FixNormalsCuMeshNode(io.ComfyNode):
    """Fix winding/normal orientation on the GPU via CuMesh.unify_face_orientations()."""

    ACCELERATOR = "cuda"  # comfy-env: this node requires CUDA at execution

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFixNormals_CuMesh",
            display_name="Fix Normals CuMesh Unify (backend)",
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
        import torch
        import cumesh as CuMesh
        import comfy.model_management

        was_consistent = trimesh.is_winding_consistent

        device = comfy.model_management.get_torch_device()
        assert device.type == "cuda", (
            f"CuMesh requires CUDA but got device '{device}' -- cumesh is GPU-only, no CPU fallback")

        V = torch.tensor(np.asarray(trimesh.vertices), dtype=torch.float32).to(device)
        F = torch.tensor(np.asarray(trimesh.faces), dtype=torch.int32).to(device)

        cm = CuMesh.CuMesh()
        cm.init(V, F)
        cm.unify_face_orientations()
        out_v, out_f = cm.read()

        fixed_mesh = trimesh_module.Trimesh(
            vertices=out_v.cpu().numpy(),
            faces=out_f.cpu().numpy(),
            process=False,
        )
        fixed_mesh.metadata = trimesh.metadata.copy()

        del cm, V, F, out_v, out_f
        comfy.model_management.soft_empty_cache()

        is_consistent = fixed_mesh.is_winding_consistent
        info = (
            f"Normal Orientation Fix:\n"
            f"\n"
            f"Method: cumesh unify_face_orientations (GPU)\n"
            f"Before: {'Consistent' if was_consistent else 'Inconsistent'}\n"
            f"After:  {'Consistent' if is_consistent else 'Inconsistent'}\n"
            f"\n"
            f"Vertices: {len(fixed_mesh.vertices):,}\n"
            f"Faces: {len(fixed_mesh.faces):,}\n"
            f"\n"
            f"{'[OK] Normals are now consistently oriented!' if is_consistent else '[WARN] Some inconsistencies may remain (check mesh topology)'}"
        )
        log.info("cumesh unify: %s -> %s", was_consistent, is_consistent)

        return io.NodeOutput(fixed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFixNormals_CuMesh": FixNormalsCuMeshNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFixNormals_CuMesh": "Fix Normals CuMesh Unify (backend)"}
