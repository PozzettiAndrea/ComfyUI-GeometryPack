# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""QuadWild BiMDF tri-to-quad remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshQuadWildNode(io.ComfyNode):
    """QuadWild BiMDF tri-to-quad remeshing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_QuadWild",
            display_name="Remesh QuadWild (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("qw_sharp_angle", default=35.0, min=0.0, max=180.0, step=1.0, tooltip="Dihedral angle threshold for sharp feature detection."),
                io.Float.Input("qw_alpha", default=0.02, min=0.005, max=0.1, step=0.005, display_mode="number", tooltip="Balance between regularity and isometry. Lower = more regular quads."),
                io.Float.Input("qw_scale_factor", default=1.0, min=0.1, max=10.0, step=0.1, tooltip="Quad size multiplier. Larger = bigger quads, fewer faces."),
                io.Combo.Input("qw_remesh", options=["true", "false"], default="true", tooltip="Pre-remesh input for better triangle quality."),
                io.Combo.Input("qw_smooth", options=["true", "false"], default="true", tooltip="Smooth output mesh topology after quadrangulation."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, qw_sharp_angle=35.0, qw_alpha=0.02, qw_scale_factor=1.0,
                qw_remesh="true", qw_smooth="true"):
        import pyquadwild

        log.info("Backend: quadwild")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: sharp_angle=%s, alpha=%s, scale=%s, remesh=%s, smooth=%s",
                 qw_sharp_angle, qw_alpha, qw_scale_factor, qw_remesh, qw_smooth)

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        V_out, F_out = pyquadwild.quadwild_remesh(
            V, F,
            remesh=(qw_remesh == "true"),
            sharp_angle=qw_sharp_angle,
            alpha=qw_alpha,
            scale_factor=qw_scale_factor,
            smooth=(qw_smooth == "true"),
        )

        remeshed_mesh = trimesh_module.Trimesh(vertices=V_out, faces=F_out, process=False)
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'quadwild',
            'sharp_angle': qw_sharp_angle,
            'alpha': qw_alpha,
            'scale_factor': qw_scale_factor,
            'remesh': qw_remesh == "true",
            'smooth': qw_smooth == "true",
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (QuadWild BiMDF): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"alpha={qw_alpha}, scale={qw_scale_factor}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_QuadWild": RemeshQuadWildNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_QuadWild": "Remesh QuadWild (backend)"}
