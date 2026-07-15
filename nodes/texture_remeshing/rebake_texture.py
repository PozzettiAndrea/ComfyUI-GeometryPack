# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Rebake Texture - unified frontend with backend selection.

Bakes original_mesh's texture onto uv_mesh's OWN UV layout (produced by whichever UV
Unwrap node you already ran -- Xatlas, ARAP, LSCM, Harmonic, Geogram ABF, or anything
else that leaves visual.uv populated). Deliberately does NOT do UV unwrapping itself --
compose it with an upstream UV Unwrap node and (if uv_mesh is a different topology) a
Remesh node, same as every other backend-selecting node in this pack.

cpu: numpy-vectorized per-texel rasterization + closest-point projection via trimesh's
     spatial-index search.
gpu: hardware (OpenGL/EGL) rasterization + closest-point search via cumesh's cuBVH --
     both real GPU acceleration, not from-scratch approximations. Falls back to the CPU
     rasterizer automatically if EGL/moderngl isn't available.
"""

from comfy_api.latest import io


class RebakeTextureNode(io.ComfyNode):
    """Bake the original mesh's texture onto a re-UV'd mesh -- pick a backend."""

    BACKEND_MAP = {
        "cpu": "GeomPackRebakeTexture_CPU",
        "gpu": "GeomPackRebakeTexture_GPU",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRebakeTexture",
            display_name="Rebake Texture",
            category="geompack/texture_remeshing",
            description=(
                "Bakes original_mesh's texture onto uv_mesh's own UV layout via per-texel "
                "closest-point projection -- a real UV-mapped texture bake, not a per-vertex "
                "color hack. uv_mesh must already be UV-unwrapped (Xatlas / ARAP / LSCM / "
                "Harmonic / Geogram ABF / etc. -- run a UV Unwrap node first)."
            ),
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("original_mesh", tooltip="Source mesh: needs its own UVs + material/texture. Sampled FROM."),
                io.Custom("TRIMESH").Input("uv_mesh", tooltip="Target mesh: needs its own UV layout already (any UV Unwrap node). Baked ONTO."),
                io.DynamicCombo.Input("backend", tooltip="Bake backend", options=[
                    io.DynamicCombo.Option("cpu", [
                        io.Int.Input("texture_size", default=1024, min=64, max=8192, step=64,
                                     tooltip="Output texture resolution (square)."),
                        io.Int.Input("bake_margin", default=8, min=0, max=64,
                                     tooltip="Pixels to pad/dilate baked color into empty texels around UV "
                                             "island borders, to avoid black seam bleeding when mipmapped."),
                    ]),
                    io.DynamicCombo.Option("gpu", [
                        io.Int.Input("texture_size", default=1024, min=64, max=8192, step=64,
                                     tooltip="Output texture resolution (square)."),
                        io.Int.Input("bake_margin", default=8, min=0, max=64,
                                     tooltip="Pixels to pad/dilate baked color into empty texels around UV "
                                             "island borders, to avoid black seam bleeding when mipmapped."),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="textured_mesh"),
                io.Image.Output(display_name="texture"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, original_mesh, uv_mesh, backend):
        from comfy_execution.graph_utils import GraphBuilder
        if cls.SCHEMA is None:
            cls.GET_SCHEMA()
        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]
        kwargs = {"original_mesh": original_mesh, "uv_mesh": uv_mesh}
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


NODE_CLASS_MAPPINGS = {"GeomPackRebakeTexture": RebakeTextureNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRebakeTexture": "Rebake Texture"}
