# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified UV Unwrap Node - Single frontend with backend selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes, each running in its own isolation env.

Supports:
- xatlas: Fast automatic UV unwrapping (vertex splitting)
- cumesh: GPU-accelerated UV unwrapping
- libigl_lscm: Least Squares Conformal Maps (angle-preserving)
- libigl_harmonic: Harmonic mapping (requires boundary)
- libigl_arap: As-Rigid-As-Possible (iterative, high quality)
- blender_smart: Smart UV Project (automatic seams)
- blender_cube: Cube projection (6 faces)
- blender_cylinder: Cylindrical projection
- blender_sphere: Spherical projection
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class UVUnwrapNode(io.ComfyNode):
    """
    Universal UV Unwrap - Unified UV unwrapping operations.

    Dispatches to hidden backend nodes via node expansion.
    Backends span multiple isolation envs (main, blender, gpu).
    """

    # Map DynamicCombo option key -> hidden backend node_id
    BACKEND_MAP = {
        "xatlas": "GeomPackUV_Xatlas",
        "cumesh": "GeomPackUV_CuMesh",
        "libigl_lscm": "GeomPackUV_LibiglLSCM",
        "libigl_harmonic": "GeomPackUV_LibiglHarmonic",
        "libigl_arap": "GeomPackUV_LibiglARAP",
        "blender_smart": "GeomPackUV_BlenderSmart",
        "blender_cube": "GeomPackUV_BlenderCube",
        "blender_cylinder": "GeomPackUV_BlenderCylinder",
        "blender_sphere": "GeomPackUV_BlenderSphere",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackUVUnwrap",
            display_name="UV Unwrap",
            category="geompack/uv",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.DynamicCombo.Input("backend", options=[
                    io.DynamicCombo.Option("xatlas", []),
                    io.DynamicCombo.Option("cumesh", [
                        io.Float.Input("chart_cone_angle", default=90.0, min=0.0, max=359.9, step=1.0),
                        io.Int.Input("chart_refine_iterations", default=0, min=0, max=10, step=1),
                        io.Int.Input("chart_global_iterations", default=1, min=0, max=10, step=1),
                        io.Int.Input("chart_smooth_strength", default=1, min=0, max=10, step=1),
                    ]),
                    io.DynamicCombo.Option("libigl_lscm", []),
                    io.DynamicCombo.Option("libigl_harmonic", []),
                    io.DynamicCombo.Option("libigl_arap", [
                        io.Int.Input("iterations", default=10, min=1, max=100, step=1),
                    ]),
                    io.DynamicCombo.Option("blender_smart", [
                        io.Float.Input("angle_limit", default=66.0, min=1.0, max=89.0, step=1.0),
                        io.Float.Input("island_margin", default=0.02, min=0.0, max=1.0, step=0.01),
                        io.Combo.Input("scale_to_bounds", options=["true", "false"], default="true"),
                    ]),
                    io.DynamicCombo.Option("blender_cube", [
                        io.Float.Input("cube_size", default=1.0, min=0.1, max=10.0, step=0.1),
                        io.Combo.Input("scale_to_bounds", options=["true", "false"], default="true"),
                    ]),
                    io.DynamicCombo.Option("blender_cylinder", [
                        io.Float.Input("cylinder_radius", default=1.0, min=0.1, max=10.0, step=0.1),
                        io.Combo.Input("scale_to_bounds", options=["true", "false"], default="true"),
                    ]),
                    io.DynamicCombo.Option("blender_sphere", [
                        io.Combo.Input("scale_to_bounds", options=["true", "false"], default="true"),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="unwrapped_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, backend):
        from comfy_execution.graph_utils import GraphBuilder

        # Ensure SCHEMA is initialized (worker subprocess doesn't call GET_SCHEMA)
        if cls.SCHEMA is None:
            cls.GET_SCHEMA()

        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]

        log.info("UV Unwrap dispatch: %s -> %s", selected, node_id)

        # Build kwargs for the backend node: mesh + backend-specific params
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
    "GeomPackUVUnwrap": UVUnwrapNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackUVUnwrap": "UV Unwrap",
}
