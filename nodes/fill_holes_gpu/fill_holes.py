# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""CuMesh GPU fill holes backend node."""

import logging
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FillHolesGPUNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFillHoles_GPU",
            display_name="Fill Holes GPU (backend)",
            category="geompack/repair",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.Float.Input("perimeter", default=0.03, min=0.001, max=100.0, step=0.001,
                               tooltip="Maximum hole perimeter to fill. Relative to mesh scale.",
                               optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="filled_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, perimeter=0.03):
        import torch
        import cumesh as CuMesh
        import comfy.model_management

        initial_faces = len(mesh.faces)
        was_watertight = mesh.is_watertight

        device = comfy.model_management.get_torch_device()
        assert device.type == "cuda", f"CuMesh requires CUDA but got device '{device}'"

        vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)
        faces = torch.tensor(mesh.faces, dtype=torch.int32).to(device)

        cumesh_obj = CuMesh.CuMesh()
        cumesh_obj.init(vertices, faces)
        cumesh_obj.fill_holes(max_hole_perimeter=perimeter)

        final_verts, final_faces = cumesh_obj.read()
        filled = trimesh_module.Trimesh(
            vertices=final_verts.cpu().numpy(),
            faces=final_faces.cpu().numpy(),
            process=False,
        )

        added = len(filled.faces) - initial_faces
        is_watertight = filled.is_watertight

        log.info("CuMesh GPU: perimeter=%s, +%d faces, watertight: %s -> %s",
                 perimeter, added, was_watertight, is_watertight)

        info = (f"Method: cumesh (GPU)\n"
                f"Max hole perimeter: {perimeter}\n"
                f"Faces added: {added}\n"
                f"Watertight: {was_watertight} -> {is_watertight}")

        return io.NodeOutput(filled, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFillHoles_GPU": FillHolesGPUNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFillHoles_GPU": "Fill Holes GPU (backend)"}
