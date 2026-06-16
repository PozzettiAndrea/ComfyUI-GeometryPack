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
                io.Float.Input("qw_alpha", default=0.02, min=0.005, max=0.1, step=0.005, display_mode="number", tooltip="Balances quad-grid REGULARITY vs feature ALIGNMENT (QuadWild's BiMDF alpha weight). Lower (~0.005) = a more uniform, regular quad grid with fewer singularities (irregular vertices), but the quad edges follow surface features/curvature less. Higher (~0.1) = allows more singularities so the quad flow bends to align with curvature and sharp edges, at the cost of grid uniformity. Typical 0.005-0.1; default 0.02."),
                io.Float.Input("qw_scale_factor", default=1.0, min=0.1, max=10.0, step=0.1, tooltip="Quad size multiplier. Larger = bigger quads, fewer faces. Used only when target_vertices and target_edge_length are both 0."),
                io.Int.Input("qw_target_vertices", default=0, min=0, max=20000000, step=100, tooltip="Target output vertices (0 = off, use scale_factor). Back-solves scale_factor from input mesh area. Overrides scale_factor."),
                io.Float.Input("qw_target_edge_length", default=0.0, min=0.0, max=1000.0, step=0.001, display_mode="number", tooltip="Target quad edge length (0 = off, use scale_factor). Overrides BOTH scale_factor and target_vertices."),
                io.Combo.Input("qw_remesh", options=["true", "false"], default="true", tooltip="Pre-remesh input for better triangle quality."),
                io.Combo.Input("qw_smooth", options=["true", "false"], default="true", tooltip="Smooth output mesh topology after quadrangulation."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @staticmethod
    def _resolve_scale(trimesh, scale_factor, target_vertices, target_edge_length):
        """Resolve QuadWild's scale_factor from an optional target edge length or
        vertex count (0 = unset). QuadWild's output edge ~= mean input edge *
        scale_factor; for a quad mesh of area A with edge e, #verts ~= #quads ~= A/e^2,
        so e = sqrt(A / N). Priority: edge_length > vertices > scale_factor.
        Returns (scale_factor, human-readable note)."""
        try:
            base_edge = float(np.mean(trimesh.edges_unique_length))
        except Exception:
            base_edge = 0.0
        try:
            area = float(trimesh.area)
        except Exception:
            area = 0.0

        tgt_edge = float(target_edge_length or 0.0)
        tgt_verts = int(target_vertices or 0)

        # vertex count -> edge length (via area), if no explicit edge length given
        if tgt_edge <= 0.0 and tgt_verts > 0 and area > 0.0:
            tgt_edge = (area / tgt_verts) ** 0.5

        if tgt_edge > 0.0 and base_edge > 0.0:
            scale = max(0.02, min(50.0, tgt_edge / base_edge))
            implied_v = int(area / (tgt_edge ** 2)) if (tgt_edge > 0 and area > 0) else 0
            note = (f"target_edge={tgt_edge:.4g} (~{implied_v:,} verts) | "
                    f"area={area:.4g}, base_edge={base_edge:.4g} -> scale_factor={scale:.4g}")
            return scale, note
        return float(scale_factor), f"scale_factor={scale_factor} (no target set)"

    @classmethod
    def execute(cls, trimesh, qw_sharp_angle=35.0, qw_alpha=0.02, qw_scale_factor=1.0,
                qw_remesh="true", qw_smooth="true", qw_target_vertices=0, qw_target_edge_length=0.0):
        import pyquadwild

        scale, note = cls._resolve_scale(trimesh, qw_scale_factor, qw_target_vertices, qw_target_edge_length)

        log.info("Backend: quadwild")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Density: %s", note)
        log.info("Parameters: sharp_angle=%s, alpha=%s, scale=%s, remesh=%s, smooth=%s",
                 qw_sharp_angle, qw_alpha, scale, qw_remesh, qw_smooth)

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        V_out, F_out = pyquadwild.quadwild_remesh(
            V, F,
            remesh=(qw_remesh == "true"),
            sharp_angle=qw_sharp_angle,
            alpha=qw_alpha,
            scale_factor=scale,
            smooth=(qw_smooth == "true"),
        )

        remeshed_mesh = trimesh_module.Trimesh(vertices=V_out, faces=F_out, process=False)
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'quadwild',
            'sharp_angle': qw_sharp_angle,
            'alpha': qw_alpha,
            'scale_factor': scale,
            'target_vertices': int(qw_target_vertices or 0),
            'target_edge_length': float(qw_target_edge_length or 0.0),
            'remesh': qw_remesh == "true",
            'smooth': qw_smooth == "true",
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (QuadWild BiMDF): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"alpha={qw_alpha}, {note}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_QuadWild": RemeshQuadWildNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_QuadWild": "Remesh QuadWild (backend)"}
