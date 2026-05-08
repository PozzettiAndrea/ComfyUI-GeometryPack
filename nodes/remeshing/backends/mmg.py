# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""MMG adaptive surface remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshMMGNode(io.ComfyNode):
    """MMG curvature-adaptive surface remeshing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_MMG",
            display_name="Remesh MMG (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("hausd", default=0.01, min=0.0001, max=10.0, step=0.001, display_mode="number", tooltip="Hausdorff distance: maximum geometric deviation from original surface."),
                io.Float.Input("hmin", default=0.0, min=0.0, max=10.0, step=0.001, display_mode="number", tooltip="Minimum edge length. 0 = auto."),
                io.Float.Input("hmax", default=0.0, min=0.0, max=100.0, step=0.01, display_mode="number", tooltip="Maximum edge length. 0 = auto."),
                io.Float.Input("hgrad", default=1.3, min=1.0, max=5.0, step=0.1, display_mode="number", tooltip="Gradation: controls how fast element sizes change. 1.3 = smooth transitions."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, hausd=0.01, hmin=0.0, hmax=0.0, hgrad=1.3):
        try:
            import mmgpy
        except ImportError:
            log.warning("mmgpy not available on Windows — returning input mesh unchanged")
            info = "Remesh (MMG Adaptive): skipped — mmgpy not available on this platform"
            return io.NodeOutput(trimesh, info, ui={"text": [info]})

        log.info("Backend: mmg_adaptive")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: hausd=%s, hmin=%s, hmax=%s, hgrad=%s", hausd, hmin, hmax, hgrad)

        vertices = np.array(trimesh.vertices, dtype=np.float64)
        faces = np.array(trimesh.faces, dtype=np.int32)
        mmg_mesh = mmgpy.Mesh(vertices, faces)

        opts_kwargs = {"hausd": hausd, "hgrad": hgrad, "verbose": -1}
        if hmin > 0:
            opts_kwargs["hmin"] = hmin
        if hmax > 0:
            opts_kwargs["hmax"] = hmax

        opts = mmgpy.MmgSOptions(**opts_kwargs)

        log.info("Running mmgs surface remeshing...")
        result = mmg_mesh.remesh(opts)

        if not result.success:
            raise ValueError(f"MMG remeshing failed (return code {result.return_code})")

        out_vertices = mmg_mesh.get_vertices()
        out_faces = mmg_mesh.get_triangles()

        remeshed_mesh = trimesh_module.Trimesh(vertices=out_vertices, faces=out_faces, process=False)
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'mmg_adaptive',
            'hausd': hausd, 'hmin': hmin, 'hmax': hmax, 'hgrad': hgrad,
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (MMG Adaptive): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"hausd={hausd}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_MMG": RemeshMMGNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_MMG": "Remesh MMG (backend)"}
