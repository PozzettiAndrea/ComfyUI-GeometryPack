# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Geogram curvature-adapted CVT remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshGeogramAnisoNode(io.ComfyNode):
    """Geogram curvature-adapted CVT remeshing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_GeogramAniso",
            display_name="Remesh Geogram Anisotropic (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("nb_points_aniso", default=5000, min=0, max=1000000, step=100, tooltip="Target number of output vertices."),
                io.Float.Input("anisotropy", default=0.04, min=0.005, max=0.5, step=0.005, display_mode="number", tooltip="Anisotropy factor. Lower = more anisotropic. Typical: 0.02-0.1."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, nb_points_aniso=5000, anisotropy=0.04):
        import pygeogram

        log.info("Backend: geogram_anisotropic")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: nb_points=%s, anisotropy=%s", f"{nb_points_aniso:,}", anisotropy)

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        V_out, F_out = pygeogram.remesh_anisotropic(
            V, F, nb_points_aniso, anisotropy=anisotropy,
        )

        remeshed_mesh = trimesh_module.Trimesh(vertices=V_out, faces=F_out, process=False)
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'geogram_anisotropic',
            'nb_points': nb_points_aniso,
            'anisotropy': anisotropy,
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (Geogram CVT Anisotropic): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"nb_points={nb_points_aniso:,}, anisotropy={anisotropy}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_GeogramAniso": RemeshGeogramAnisoNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_GeogramAniso": "Remesh Geogram Anisotropic (backend)"}
