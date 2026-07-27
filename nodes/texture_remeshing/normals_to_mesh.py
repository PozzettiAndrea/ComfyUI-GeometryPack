# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Normals to Mesh - unified frontend with backend selection.

Reconstructs a 3D height surface from an RGB normal map, each backend a separate
hidden node dispatched through GraphBuilder:
- poisson_cpu: mask-aware sparse Poisson integration, exact SuperLU direct solve
- poisson_gpu: same system, Jacobi-preconditioned CG on the GPU (fast on big masks)
- frankot_chellappa: full-frame FFT least-squares integration (fastest; mask only
  crops the output, it does not confine the integration domain)
"""

from comfy_api.latest import io


class NormalsToMeshNode(io.ComfyNode):
    """Reconstruct a height surface from an RGB normal map -- pick a backend."""

    BACKEND_MAP = {
        "poisson_cpu": "GeomPackNormalsToMesh_PoissonCPU",
        # "poisson_gpu": "GeomPackNormalsToMesh_PoissonGPU",  # disabled for now (backend code kept)
        "frankot_chellappa": "GeomPackNormalsToMesh_FrankotChellappa",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackNormalsToMesh",
            display_name="Normals to Mesh",
            category="geompack/texture_remeshing",
            enable_expand=True,
            is_output_node=True,
            description=(
                "Reconstruct a 3D height surface from an RGB NORMAL map via normal "
                "integration. A heightfield has only 2 gradient DOF (gx=-nx/nz, "
                "gy=ny/nz), so the B channel is redundant -- by default it is ignored "
                "entirely (normal_z='recompute'). Outputs a TRIMESH surface, a "
                "normalised height IMAGE, and info."
            ),
            inputs=[
                io.DynamicCombo.Input("backend",
                    tooltip="Which integration algorithm reconstructs height from the normals:\n"
                            "- poisson_cpu: mask-aware sparse Poisson (graph Laplacian over mask "
                            "pixels only, Neumann boundary), solved exactly with scipy SuperLU. "
                            "Robust and deterministic; no gradients leak in from outside the "
                            "silhouette.\n"
                            "- frankot_chellappa: full-frame FFT least-squares integration. "
                            "Fastest by far, but always integrates the WHOLE frame: a mask only "
                            "crops the output mesh, so gradients outside a silhouette still "
                            "bleed into the boundary. Best for full-frame normal maps "
                            "(terrain / texture style) with no silhouette.",
                    options=[
                        io.DynamicCombo.Option("poisson_cpu", []),
                        # poisson_gpu disabled for now (Jacobi-PCG on GPU; backend code kept in
                        # backends/normals_to_mesh_poisson_gpu.py -- re-enable by uncommenting
                        # here, in BACKEND_MAP, and in texture_remeshing/__init__.py):
                        # io.DynamicCombo.Option("poisson_gpu", [
                        #     io.Int.Input("cg_iters", default=2000, min=10, max=100000, step=10,
                        #         tooltip="Max conjugate-gradient iterations. Big masks need more; "
                        #                 "stops early once cg_tol is reached."),
                        #     io.Float.Input("cg_tol", default=1e-5, min=1e-8, max=1e-2, step=1e-6,
                        #         display_mode="number",
                        #         tooltip="Relative residual tolerance for early stop. Lower = more "
                        #                 "accurate, slower."),
                        # ]),
                        io.DynamicCombo.Option("frankot_chellappa", []),
                    ]),
                io.Image.Input("normals",
                    tooltip="RGB normal map: R=nx, G=ny, B=nz encoded in [0,1] (=> [-1,1]). "
                            "Predicted (e.g. Lotus) normals work -- they're re-normalised."),
                io.Mask.Input("mask", optional=True,
                    tooltip="Which pixels to reconstruct. Optional: when omitted, the whole "
                            "frame is reconstructed (mask = image boundaries). When connected, "
                            "the poisson backends integrate ONLY over mask>0.5 (the silhouette "
                            "is the surface boundary, nothing leaks in from outside); for "
                            "frankot_chellappa the mask only crops the output mesh -- it always "
                            "integrates the full frame."),
                io.Float.Input("height_scale", default=1.0, min=0.001, max=100.0, step=0.01,
                    display_mode="number",
                    tooltip="Z multiplier. 1.0 keeps Z metrically proportional to X/Y (the "
                            "integration is already scale-correct); raise/lower to exaggerate "
                            "or flatten relief."),
                io.Combo.Input("normal_z", options=["recompute", "use"], default="recompute", optional=True,
                    tooltip="recompute (default): ignore the B channel entirely and derive "
                            "nz=sqrt(1-nx^2-ny^2) -- a heightfield normal has only 2 degrees of "
                            "freedom, so B adds no information and its encoding error only "
                            "corrupts the gradients (max_slope_deg guards the nz~0 edge case). "
                            "use: read B and normalise the full (nx,ny,nz) vector -- opt in only "
                            "for maps whose three channels carry independent noise, where "
                            "full-vector normalisation can average it down."),
                io.Combo.Input("flip_y", options=["false", "true"], default="false", optional=True,
                    tooltip="Flip the green (ny) channel: OpenGL vs DirectX normal-map "
                            "convention. If the surface comes out inverted top-to-bottom, "
                            "toggle this."),
                io.Float.Input("max_slope_deg", default=87.0, min=1.0, max=89.9, step=0.5,
                    display_mode="number", optional=True,
                    tooltip="Maximum surface slope angle, in degrees (90 = vertical). The "
                            "per-pixel gradients derived from the normals are clamped to "
                            "tan(max_slope_deg). A grazing normal (nz ~ 0, i.e. a near-vertical "
                            "surface) otherwise produces a gradient in the thousands and the "
                            "integrator shoots that pixel towards infinity -- a heightfield "
                            "cannot represent vertical cliffs anyway. Raise towards 89.9 only "
                            "if your normal map is clean and you want steeper walls."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="surface"),
                io.Image.Output(display_name="height_map"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, backend=None, normals=None, mask=None, height_scale=1.0,
                normal_z="recompute", flip_y="false", max_slope_deg=87.0):
        from comfy_execution.graph_utils import GraphBuilder
        if cls.SCHEMA is None:
            cls.GET_SCHEMA()
        if backend is None:
            backend = {"backend": "poisson_cpu"}
        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]
        kwargs = {
            "normals": normals,
            "height_scale": height_scale,
            "normal_z": normal_z,
            "flip_y": flip_y,
            "max_slope_deg": max_slope_deg,
        }
        if mask is not None:
            kwargs["mask"] = mask
        for k, v in backend.items():
            if k == "backend":
                continue
            kwargs[k] = v
        graph = GraphBuilder()
        backend_node = graph.node(node_id, **kwargs)
        return {
            "result": (backend_node.out(0), backend_node.out(1), backend_node.out(2)),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {"GeomPackNormalsToMesh": NormalsToMeshNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackNormalsToMesh": "Normals to Mesh"}
