# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Sharpen Mesh Node - Single frontend with backend selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes.

Available backends:
- two_step: Two-phase bilateral normal filtering via pymeshlab. Smooths face
  normals (respecting dihedral angle thresholds), then repositions vertices to
  match. Sharpens creases while keeping faces flat. Best for CAD-like geometry
  from marching cubes, scanning, or neural SDF extraction.
- unsharp_mask: Geometric unsharp masking via pymeshlab. Subtracts a smoothed
  version from the original to amplify ridges and valleys.
- libigl_unsharp: Cotangent-weighted geometric unsharp mask via libigl.
  Geometrically superior to uniform-weight unsharp masking because cotangent
  Laplacian respects mesh geometry (triangle shape/area).
- l0_minimize: L0 normal minimization (He & Schaefer 2013). Minimizes the
  number of distinct face normal orientations, forcing the mesh into
  piecewise-flat regions with sharp edges at boundaries. Best for aggressive
  CAD-like sharpening.
- guided_normal: Guided mesh normal filtering (Zhang et al. 2015). Uses a
  min-range-metric guidance signal to drive bilateral normal filtering while
  preserving sharp edges. Interleaves vertex updates within normal iterations.
- fast_effective: Fast and Effective Feature-Preserving Mesh Denoising
  (Sun et al. TVCG 2007). Uses thresholded cosine-similarity weights for
  normal filtering: w = max(0, dot(ni,nj) - T)^2. Simple and fast.
- non_iterative: Non-Iterative Feature-Preserving Mesh Smoothing
  (Jones et al. SIGGRAPH 2003). Mollifies normals on a smoothed mesh copy,
  then does a single-pass bilateral vertex update using spatial and influence
  Gaussian weights with BFS face neighbor search.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class SharpenMeshNode(io.ComfyNode):
    """
    Sharpen Mesh - Unified sharpening with backend selection.

    Dispatches to hidden backend nodes via node expansion.
    """

    BACKEND_MAP = {
        "two_step":       "GeomPackSharpen_TwoStep",
        "unsharp_mask":   "GeomPackSharpen_UnsharpMask",
        "libigl_unsharp": "GeomPackSharpen_LibiglUnsharp",
        "l0_minimize":    "GeomPackSharpen_L0Minimize",
        "guided_normal":  "GeomPackSharpen_GuidedNormal",
        "fast_effective":  "GeomPackSharpen_FastEffective",
        "non_iterative":  "GeomPackSharpen_NonIterative",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpenMesh",
            display_name="Sharpen Mesh",
            category="geompack/smoothing",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.DynamicCombo.Input("backend", tooltip=(
                        "Sharpening algorithm. "
                        "two_step=bilateral normal filtering (recommended for CAD-like edges), "
                        "unsharp_mask=geometric unsharp masking (pymeshlab), "
                        "libigl_unsharp=cotangent-weighted unsharp (geometry-aware), "
                        "l0_minimize=piecewise-flat L0 optimization (aggressive CAD prep), "
                        "guided_normal=guided normal filtering with min-range-metric (controllable), "
                        "fast_effective=thresholded cosine weight normal filtering (fast), "
                        "non_iterative=mollified normal single-pass bilateral (non-iterative)"
                    ), options=[
                    io.DynamicCombo.Option("two_step", [
                        io.Int.Input("smooth_steps", default=3, min=1, max=50, step=1, tooltip=(
                            "Number of two-step smoothing passes. "
                            "More steps = stronger sharpening effect."
                        )),
                        io.Float.Input("normal_threshold", default=60.0, min=0.0, max=180.0, step=0.5, tooltip=(
                            "Dihedral angle threshold in degrees. "
                            "Edges sharper than this angle are preserved as features. "
                            "Lower = more aggressive (more edges treated as creases). "
                            "60 is a good default for most CAD models."
                        )),
                    ]),
                    io.DynamicCombo.Option("unsharp_mask", [
                        io.Float.Input("weight", default=0.3, min=0.0, max=3.0, step=0.01, tooltip=(
                            "Unsharp mask weight controlling sharpening strength. "
                            "Higher = more pronounced sharpening."
                        )),
                        io.Int.Input("iterations", default=5, min=1, max=50, step=1, tooltip=(
                            "Smoothing iterations for the reference smooth mesh. "
                            "More iterations = larger-scale sharpening."
                        )),
                    ]),
                    io.DynamicCombo.Option("libigl_unsharp", [
                        io.Float.Input("weight", default=0.5, min=0.01, max=5.0, step=0.01, tooltip=(
                            "How much detail to add back. 0.5 = subtle sharpening, "
                            "1.0 = double the detail, 2.0+ = aggressive."
                        )),
                        io.Int.Input("iterations", default=3, min=1, max=50, step=1, tooltip=(
                            "Smoothing iterations for the reference mesh. "
                            "More iterations = smoother reference = sharpens broader features. "
                            "Fewer iterations = sharpens fine detail."
                        )),
                    ]),
                    io.DynamicCombo.Option("l0_minimize", [
                        io.Float.Input("alpha", default=0.001, min=0.0001, max=0.1, step=0.0001, tooltip=(
                            "Initial regularization weight for L0 minimization. "
                            "Controls the threshold below which normal differences "
                            "are snapped to zero. Smaller = gentler start, "
                            "larger = more aggressive initial flattening."
                        )),
                        io.Float.Input("beta", default=2.0, min=1.1, max=10.0, step=0.1, tooltip=(
                            "Growth rate for alpha each iteration. Alpha is multiplied "
                            "by beta each step. 2.0 doubles per iteration. "
                            "Higher = faster convergence to piecewise-flat."
                        )),
                        io.Int.Input("iterations", default=10, min=1, max=50, step=1, tooltip=(
                            "Number of L0 optimization iterations. The algorithm "
                            "gradually increases the threshold, snapping more normals "
                            "flat each step."
                        )),
                    ]),
                    io.DynamicCombo.Option("guided_normal", [
                        io.Int.Input("normal_iterations", default=5, min=1, max=50, step=1, tooltip=(
                            "Iterations for guided bilateral normal filtering. "
                            "More iterations produce smoother/flatter regions while "
                            "preserving sharp edges."
                        )),
                        io.Int.Input("vertex_iterations", default=10, min=1, max=100, step=1, tooltip=(
                            "Iterations for updating vertex positions to match filtered "
                            "normals. More iterations give better convergence."
                        )),
                        io.Float.Input("sigma_s", default=1.0, min=0.1, max=10.0, step=0.1, tooltip=(
                            "Spatial weight sigma as a multiple of average edge length. "
                            "Controls the neighborhood size for normal filtering. "
                            "Larger = smoother but may blur sharp features."
                        )),
                        io.Float.Input("sigma_r", default=0.35, min=0.01, max=1.0, step=0.01, tooltip=(
                            "Normal similarity threshold. Controls which normals are "
                            "averaged together. Smaller = more aggressive edge "
                            "preservation. 0.35 corresponds to roughly 40 degree "
                            "dihedral angle threshold."
                        )),
                    ]),
                    io.DynamicCombo.Option("fast_effective", [
                        io.Float.Input("threshold_T", default=0.5, min=1e-10, max=1.0, step=0.01, tooltip=(
                            "Cosine similarity threshold (Sun et al. TVCG 2007). "
                            "Normals with dot(ni,nj) > T contribute with weight "
                            "(dot-T)^2; below T they contribute nothing. "
                            "Lower = more normals averaged (smoother), "
                            "higher = only very similar normals averaged (sharper). "
                            "0.5 is a good default."
                        )),
                        io.Int.Input("normal_iterations", default=20, min=1, max=500, step=1, tooltip=(
                            "Iterations for normal filtering. More iterations "
                            "produce stronger flattening of near-flat regions."
                        )),
                        io.Int.Input("vertex_iterations", default=50, min=1, max=500, step=1, tooltip=(
                            "Iterations for vertex position update from filtered "
                            "normals. Boundary vertices are kept fixed."
                        )),
                    ]),
                    io.DynamicCombo.Option("non_iterative", [
                        io.Float.Input("sigma_f", default=1.0, min=0.001, max=10.0, step=0.1, tooltip=(
                            "Spatial sigma as multiple of average edge length "
                            "(Jones et al. SIGGRAPH 2003). Controls spatial extent "
                            "of the bilateral filter. Face neighbors are searched "
                            "within radius 2*sigma_f. Larger = smoother."
                        )),
                        io.Float.Input("sigma_g", default=1.0, min=0.001, max=10.0, step=0.1, tooltip=(
                            "Influence sigma as multiple of average edge length. "
                            "Controls sensitivity to projection distance (how far "
                            "the vertex moves toward each face plane). Smaller = "
                            "more feature-preserving."
                        )),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
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

        log.info("Sharpen dispatch: %s -> %s", selected, node_id)

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
    "GeomPackSharpenMesh": SharpenMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackSharpenMesh": "Sharpen Mesh",
}
