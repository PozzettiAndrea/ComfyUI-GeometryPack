# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Remesh Node - Single frontend with backend selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes, each running in its own isolation env.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshNode(io.ComfyNode):
    """
    Remesh - Unified remeshing with backend selection.

    Dispatches to hidden backend nodes via node expansion.
    Backends span multiple isolation envs (main, blender, gpu).
    """

    # Map DynamicCombo option key -> hidden backend node_id
    BACKEND_MAP = {
        "pymeshlab_isotropic": "GeomPackRemesh_PyMeshLab",
        "instant_meshes":      "GeomPackRemesh_InstantMeshes",
        "quadriflow":          "GeomPackRemesh_QuadriFlow",
        "mmg_adaptive":        "GeomPackRemesh_MMG",
        "geogram_smooth":      "GeomPackRemesh_GeogramSmooth",
        "geogram_anisotropic": "GeomPackRemesh_GeogramAniso",
        "pmp_uniform":         "GeomPackRemesh_PMPUniform",
        "pmp_adaptive":        "GeomPackRemesh_PMPAdaptive",
        "quadwild":            "GeomPackRemesh_QuadWild",
        "cgal_isotropic":      "GeomPackRemesh_CGAL",
        "blender_voxel":       "GeomPackRemesh_BlenderVoxel",
        "blender_sharp":       "GeomPackRemesh_BlenderSharp",
        "blender_blocks":      "GeomPackRemesh_BlenderBlocks",
        "gpu_cumesh":          "GeomPackRemesh_GPU",
    }

    # Some frontend param names differ from backend param names
    # (to avoid conflicts between DynamicCombo options).
    # Map: frontend_param_name -> backend_param_name
    PARAM_REMAP = {
        "cgal_edge_length": "target_edge_length",
        "cgal_iterations": "iterations",
        "gpu_target_face_count": "target_face_count",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh",
            display_name="Remesh",
            category="geompack/remeshing",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.DynamicCombo.Input("backend", tooltip="Remeshing algorithm and backend", options=[
                    # ---- Main env backends ----
                    io.DynamicCombo.Option("pymeshlab_isotropic", [
                        io.Float.Input("target_edge_length", default=1.00, min=0.001, max=10.0, step=0.01, display_mode="number", tooltip="Target edge length for output triangles. Value is relative to mesh scale."),
                        io.Int.Input("iterations", default=3, min=1, max=20, step=1, tooltip="Number of remeshing passes. More iterations = smoother result, slower processing."),
                        io.Float.Input("feature_angle", default=30.0, min=0.0, max=180.0, step=1.0, tooltip="Angle threshold (degrees) for feature edge detection."),
                        io.Combo.Input("adaptive", options=["true", "false"], default="false", tooltip="Use curvature-adaptive edge lengths."),
                    ]),
                    io.DynamicCombo.Option("instant_meshes", [
                        io.Int.Input("target_vertex_count", default=5000, min=100, max=1000000, step=100, tooltip="Target vertex count. Creates field-aligned quad-dominant mesh."),
                        io.Combo.Input("deterministic", options=["true", "false"], default="true", tooltip="Use deterministic algorithm for reproducible results."),
                        io.Float.Input("crease_angle", default=0.0, min=0.0, max=180.0, step=1.0, tooltip="Angle threshold for preserving sharp edges. 0 = no preservation."),
                    ]),
                    io.DynamicCombo.Option("quadriflow", [
                        io.Int.Input("target_face_count", default=5000, min=100, max=5000000, step=100, tooltip="Target output faces. Creates quad-dominant mesh with good topology."),
                        io.Combo.Input("preserve_sharp", options=["true", "false"], default="false", tooltip="Preserve sharp edges during remeshing."),
                        io.Combo.Input("preserve_boundary", options=["true", "false"], default="true", tooltip="Preserve mesh boundary edges during remeshing."),
                    ]),
                    io.DynamicCombo.Option("mmg_adaptive", [
                        io.Float.Input("hausd", default=0.01, min=0.0001, max=10.0, step=0.001, display_mode="number", tooltip="Hausdorff distance: max deviation from original surface."),
                        io.Float.Input("hmin", default=0.0, min=0.0, max=10.0, step=0.001, display_mode="number", tooltip="Minimum edge length. 0 = auto."),
                        io.Float.Input("hmax", default=0.0, min=0.0, max=100.0, step=0.01, display_mode="number", tooltip="Maximum edge length. 0 = auto."),
                        io.Float.Input("hgrad", default=1.3, min=1.0, max=5.0, step=0.1, display_mode="number", tooltip="Gradation: controls size change rate. 1.3 = smooth transitions."),
                    ]),
                    io.DynamicCombo.Option("geogram_smooth", [
                        io.Int.Input("nb_points", default=5000, min=0, max=1000000, step=100, tooltip="Target output vertices. 0 = same count as input."),
                        io.Int.Input("nb_lloyd_iter", default=5, min=1, max=50, step=1, tooltip="Lloyd relaxation iterations."),
                        io.Int.Input("nb_newton_iter", default=30, min=1, max=100, step=1, tooltip="Newton optimization iterations."),
                        io.Int.Input("newton_m", default=7, min=1, max=20, step=1, tooltip="L-BFGS Hessian evaluations."),
                    ]),
                    io.DynamicCombo.Option("geogram_anisotropic", [
                        io.Int.Input("nb_points_aniso", default=5000, min=0, max=1000000, step=100, tooltip="Target output vertices."),
                        io.Float.Input("anisotropy", default=0.04, min=0.005, max=0.5, step=0.005, display_mode="number", tooltip="Anisotropy factor. Lower = more anisotropic. Typical: 0.02-0.1."),
                    ]),
                    io.DynamicCombo.Option("pmp_uniform", [
                        io.Float.Input("pmp_edge_length", default=1.0, min=0.001, max=100.0, step=0.01, display_mode="number", tooltip="Target edge length for uniform remeshing."),
                        io.Int.Input("pmp_iterations", default=10, min=1, max=100, step=1, tooltip="Number of remeshing iterations."),
                        io.Combo.Input("pmp_use_projection", options=["true", "false"], default="true", tooltip="Project vertices back onto input surface."),
                    ]),
                    io.DynamicCombo.Option("pmp_adaptive", [
                        io.Float.Input("pmp_min_edge", default=0.1, min=0.001, max=100.0, step=0.01, display_mode="number", tooltip="Minimum edge length (high-curvature areas)."),
                        io.Float.Input("pmp_max_edge", default=2.0, min=0.01, max=100.0, step=0.01, display_mode="number", tooltip="Maximum edge length (flat areas)."),
                        io.Float.Input("pmp_approx_error", default=0.1, min=0.001, max=10.0, step=0.01, display_mode="number", tooltip="Maximum geometric approximation error."),
                        io.Int.Input("pmp_adaptive_iterations", default=10, min=1, max=100, step=1, tooltip="Number of remeshing iterations."),
                        io.Combo.Input("pmp_adaptive_projection", options=["true", "false"], default="true", tooltip="Project vertices back onto input surface."),
                    ]),
                    io.DynamicCombo.Option("quadwild", [
                        io.Float.Input("qw_sharp_angle", default=35.0, min=0.0, max=180.0, step=1.0, tooltip="Dihedral angle threshold for sharp feature detection."),
                        io.Float.Input("qw_alpha", default=0.02, min=0.005, max=0.1, step=0.005, display_mode="number", tooltip="Balance regularity vs isometry. Lower = more regular quads."),
                        io.Float.Input("qw_scale_factor", default=1.0, min=0.1, max=10.0, step=0.1, tooltip="Quad size multiplier. Larger = bigger quads, fewer faces."),
                        io.Combo.Input("qw_remesh", options=["true", "false"], default="true", tooltip="Pre-remesh input for better triangle quality."),
                        io.Combo.Input("qw_smooth", options=["true", "false"], default="true", tooltip="Smooth output mesh topology after quadrangulation."),
                    ]),
                    # ---- CGAL backend ----
                    io.DynamicCombo.Option("cgal_isotropic", [
                        io.Float.Input("cgal_edge_length", default=1.00, min=0.001, max=10.0, step=0.01, display_mode="number", tooltip="Target edge length for CGAL isotropic remeshing."),
                        io.Int.Input("cgal_iterations", default=3, min=1, max=20, step=1, tooltip="Number of remeshing passes."),
                        io.Combo.Input("protect_boundaries", options=["true", "false"], default="true", tooltip="Lock boundary/open edges in place during remeshing."),
                    ]),
                    # ---- Blender backends ----
                    io.DynamicCombo.Option("blender_voxel", [
                        io.Float.Input("voxel_size", default=1, min=0.001, max=1.0, step=0.01, display_mode="number", tooltip="Voxel size. Smaller = more detail. Output is always watertight."),
                    ]),
                    io.DynamicCombo.Option("blender_sharp", [
                        io.Int.Input("octree_depth", default=6, min=1, max=10, step=1, tooltip="Resolution. Higher = more detail, more faces."),
                        io.Float.Input("scale", default=0.9, min=0.0, max=1.0, step=0.05, display_mode="number", tooltip="Ratio of output size to input bounding box."),
                        io.Float.Input("sharpness", default=1.0, min=0.0, max=5.0, step=0.1, display_mode="number", tooltip="Edge sharpness."),
                    ]),
                    io.DynamicCombo.Option("blender_blocks", [
                        io.Int.Input("octree_depth", default=6, min=1, max=10, step=1, tooltip="Resolution. Higher = more detail, more faces."),
                        io.Float.Input("scale", default=0.9, min=0.0, max=1.0, step=0.05, display_mode="number", tooltip="Ratio of output size to input bounding box."),
                    ]),
                    # ---- GPU backend ----
                    io.DynamicCombo.Option("gpu_cumesh", [
                        io.Int.Input("gpu_target_face_count", default=500000, min=1000, max=5000000, step=1000, tooltip="Target faces after simplification."),
                        io.Float.Input("remesh_band", default=1.0, min=0.1, max=5.0, step=0.1, tooltip="Band width for dual-contouring. Higher = smoother."),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
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

        log.info("Remesh dispatch: %s -> %s", selected, node_id)

        # Build kwargs for the backend node: mesh + backend-specific params
        kwargs = {"trimesh": trimesh}
        for k, v in backend.items():
            if k == "backend":
                continue
            # Remap param names if needed (frontend name -> backend name)
            backend_key = cls.PARAM_REMAP.get(k, k)
            kwargs[backend_key] = v

        graph = GraphBuilder()
        backend_node = graph.node(node_id, **kwargs)

        return {
            "result": (backend_node.out(0), backend_node.out(1)),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "GeomPackRemesh": RemeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemesh": "Remesh",
}
