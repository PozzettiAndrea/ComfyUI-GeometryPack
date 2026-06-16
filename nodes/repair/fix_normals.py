# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Fix Normals Node - Single frontend with backend selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes, each running in its own isolation env.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FixNormalsNode(io.ComfyNode):
    """
    Fix Normals - Unified normal fixing with backend selection.

    Dispatches to hidden backend nodes via node expansion.
    """

    BACKEND_MAP = {
        "trimesh":         "GeomPackFixNormals_Trimesh",
        "igl_bfs":         "GeomPackFixNormals_IglBfs",
        "igl_winding":     "GeomPackFixNormals_IglWinding",
        "igl_raycast":     "GeomPackFixNormals_IglRaycast",
        "igl_signed_dist": "GeomPackFixNormals_IglSignedDist",
        "cumesh":          "GeomPackFixNormals_CuMesh",
        "cumesh_raystab":  "GeomPackFixNormals_CuMeshRaystab",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFixNormals",
            display_name="Fix Normals",
            category="geompack/repair",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.DynamicCombo.Input("backend", tooltip="Normal fixing algorithm and backend", options=[
                    io.DynamicCombo.Option("trimesh", []),
                    io.DynamicCombo.Option("igl_bfs", []),
                    io.DynamicCombo.Option("igl_winding", []),
                    io.DynamicCombo.Option("igl_raycast", [
                        io.Int.Input("rays_minimum", default=10, min=1, max=1000, step=1, tooltip="Minimum rays cast per face for the inside/outside vote. More rays = more robust on noisy / self-intersecting meshes, slower."),
                        io.Combo.Input("use_parity", options=["true", "false"], default="false", tooltip="Decide orientation by ray-hit parity (odd/even) instead of front/back hit counts. Parity suits watertight meshes; front/back voting is more robust on open meshes."),
                        io.Combo.Input("facet_wise", options=["false", "true"], default="false", tooltip="Orient each facet independently instead of per connected component. Usually leave off (per-component is more coherent)."),
                    ]),
                    io.DynamicCombo.Option("igl_signed_dist", [
                        io.Combo.Input("sd_sign_type", options=["fast_winding_number", "winding_number", "pseudonormal"], default="fast_winding_number", tooltip="Sign oracle for the inside/outside test. fast_winding_number (default, recommended) reliably re-orients inverted meshes. winding_number / pseudonormal derive their near-surface sign from the existing face orientation, so they mainly tidy mostly-correct watertight meshes and may NOT fix a wholly-inverted mesh."),
                    ]),
                    io.DynamicCombo.Option("cumesh", []),
                    io.DynamicCombo.Option("cumesh_raystab", [
                        io.Float.Input("rs_eps", default=1e-4, min=1e-7, max=1e-1, step=1e-5, display_mode="number", tooltip="Probe offset as a fraction of the bbox diagonal. Too small = probe on surface (ambiguous); too large = crosses into a neighbouring shell."),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="fixed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, backend):
        from comfy_execution.graph_utils import GraphBuilder

        if cls.SCHEMA is None:
            cls.GET_SCHEMA()

        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]

        log.info("Fix Normals dispatch: %s -> %s", selected, node_id)

        kwargs = {"trimesh": trimesh}
        for k, v in backend.items():
            if k == "backend":
                continue
            kwargs[k] = v

        graph = GraphBuilder()
        backend_node = graph.node(node_id, **kwargs)

        return {
            "result": (backend_node.out(0), backend_node.out(1)),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "GeomPackFixNormals": FixNormalsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackFixNormals": "Fix Normals",
}
