# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Smooth Mesh Node - Single frontend with backend selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class SmoothMeshNode(io.ComfyNode):
    """
    Smooth Mesh - Unified smoothing with backend selection.

    Dispatches to hidden backend nodes via node expansion.
    """

    BACKEND_MAP = {
        "taubin":            "GeomPackSmooth_Taubin",
        "laplacian":         "GeomPackSmooth_Laplacian",
        "hc_laplacian":      "GeomPackSmooth_HCLaplacian",
        "trimesh_laplacian": "GeomPackSmooth_TrimeshLaplacian",
        "trimesh_taubin":    "GeomPackSmooth_TrimeshTaubin",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSmoothMesh",
            display_name="Smooth Mesh",
            category="geompack/smoothing",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.DynamicCombo.Input("backend", tooltip=(
                        "Smoothing algorithm. "
                        "taubin=shrinkage-free (recommended), "
                        "laplacian=fast but shrinks, "
                        "hc_laplacian=low shrinkage, "
                        "trimesh_*=lightweight alternatives"
                    ), options=[
                    io.DynamicCombo.Option("taubin", [
                        io.Int.Input("iterations", default=5, min=1, max=200, step=1, tooltip=(
                            "Number of Taubin passes (PyMeshLab 'stepsmoothnum'). Each pass is internally TWO "
                            "Laplacian steps: a positive-lambda SHRINK step then a negative-mu UN-SHRINK step, so "
                            "one iteration is a full shrink-free smoothing cycle. More iterations = sharper "
                            "frequency roll-off = high-frequency noise / tessellation artifacts are attenuated MORE "
                            "COMPLETELY -- but it does NOT change WHICH feature scale is removed (lambda/mu set "
                            "that). Think 'how thoroughly', not 'how large-scale'. Cost is linear; shape is "
                            "preserved so over-iterating mostly wastes time rather than collapsing the model. "
                            "Typical 5-30; noisy image-derived meshes 20-60. (CADFit uses 20.)")),
                        io.Float.Input("lambda_", default=0.5, min=0.01, max=1.0, step=0.01, tooltip=(
                            "Positive diffusion (smoothing) step size, 0<lambda<1. This is the size of the "
                            "Laplacian step that pulls each vertex toward the average of its 1-ring neighbours. "
                            "Larger lambda smooths more per pass AND lowers the filter cutoff, so it removes "
                            "LARGER-scale features (not just fine noise); smaller lambda touches only the finest "
                            "detail. So lambda picks the feature SCALE you attenuate, iterations pick how "
                            "completely. 0.5 is the classic default. Pushing toward 1.0 is aggressive and, with "
                            "too-small |mu|, can go unstable -- if you raise lambda, make mu more negative to "
                            "keep it shrink-free (see mu). (CADFit / trimesh default: 0.5.)")),
                        io.Float.Input("mu", default=-0.53, min=-1.0, max=-0.01, step=0.01, tooltip=(
                            "Negative inflation (un-shrink) step. After the +lambda step shrinks the mesh, the -mu "
                            "step pushes vertices back outward along the same Laplacian, cancelling Taubin's volume "
                            "loss -- THIS is what makes Taubin shrink-free, unlike plain Laplacian smoothing which "
                            "melts the model. Rule: |mu| must be >= lambda. The pair sets the passband frequency "
                            "k_PB = 1/lambda + 1/mu (a small positive number): features below k_PB are kept, above "
                            "it are smoothed. Standard recipe = mu just slightly more negative than -lambda "
                            "(lambda=0.5 -> mu=-0.53 -> k_PB~=0.11). If |mu| ~= lambda it barely smooths and sits "
                            "at the shrink-free boundary (k_PB~=0, what trimesh/CADFit do with mu=-0.50); if "
                            "|mu| < lambda it is NOT shrink-free and the mesh inflates/diverges. Keep |mu| a hair "
                            "above lambda whenever you change lambda.")),
                    ]),
                    io.DynamicCombo.Option("laplacian", [
                        io.Int.Input("iterations", default=5, min=1, max=200, step=1, tooltip="Number of smoothing passes. More = smoother but slower."),
                        io.Combo.Input("cotangent_weight", options=["true", "false"], default="true", tooltip="Use cotangent weights instead of uniform weights. Cotangent weights respect mesh geometry better but may be unstable on degenerate meshes."),
                    ]),
                    io.DynamicCombo.Option("hc_laplacian", []),
                    io.DynamicCombo.Option("trimesh_laplacian", [
                        io.Int.Input("iterations", default=5, min=1, max=200, step=1, tooltip="Number of smoothing passes. More = smoother but slower."),
                        io.Float.Input("lambda_", default=0.5, min=0.01, max=1.0, step=0.01, tooltip="Smoothing strength per step. Higher = more aggressive smoothing per iteration."),
                    ]),
                    io.DynamicCombo.Option("trimesh_taubin", [
                        io.Int.Input("iterations", default=5, min=1, max=200, step=1, tooltip=(
                            "Number of Taubin passes (pure-Python trimesh backend; uniform Laplacian). Each pass = "
                            "a +lambda smooth step then a -mu un-shrink step (shrink-free). More iterations = "
                            "high-frequency noise attenuated more completely, same feature scale (set by "
                            "lambda/mu). This is the EXACT backend CADFit's preprocess uses, with iterations=20. "
                            "Typical 5-30; noisy image-derived meshes 20-60.")),
                        io.Float.Input("lambda_", default=0.5, min=0.01, max=1.0, step=0.01, tooltip=(
                            "Positive diffusion (smoothing) step, 0<lambda<1 -- step toward the 1-ring average. "
                            "Larger lambda = more smoothing per pass AND lower cutoff = removes larger features; "
                            "smaller = only the finest detail. Picks the SCALE; iterations pick how completely. "
                            "trimesh maps this to 'lamb'. CADFit/trimesh default: 0.5.")),
                        io.Float.Input("mu", default=-0.53, min=-1.0, max=-0.01, step=0.01, tooltip=(
                            "Negative un-shrink step; cancels the volume loss of the +lambda step (shrink-free). "
                            "trimesh maps this to 'nu' with a sign flip: mu = -nu. Constraint (from trimesh): "
                            "0 < 1/lambda - 1/nu < 0.1, i.e. keep |mu| >= lambda. Standard mu=-0.53 for "
                            "lambda=0.5 (passband ~0.11). NOTE: trimesh's own default is nu=0.5 -> mu=-0.50, "
                            "exactly the shrink-free boundary (passband 0) -- that's what CADFit runs. |mu| < "
                            "lambda is NOT shrink-free and will inflate the mesh.")),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="smoothed_mesh"),
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

        log.info("Smooth dispatch: %s -> %s", selected, node_id)

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
    "GeomPackSmoothMesh": SmoothMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackSmoothMesh": "Smooth Mesh",
}
