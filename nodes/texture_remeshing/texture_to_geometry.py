# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Depth Map to Mesh - unified frontend with backend selection.

Converts a heightmap texture (IMAGE or MASK) into 3D geometry via the selected
backend, each a separate hidden node dispatched through GraphBuilder:
- grid: fast grid-based displacement (may have stair-step artifacts); the only
  backend that carries the optional 'field' input onto the mesh (1:1 pixel->vertex)
- poisson_pymeshlab: PyMeshLab Screened Poisson reconstruction (smooth)
- delaunay_2d: 2D Delaunay triangulation

Background removal is done via the 'mask' input (the old skip_black/black_threshold
depth-value heuristic was removed in favor of explicit masks).
"""

from comfy_api.latest import io


class TextureToGeometryNode(io.ComfyNode):
    """Convert a heightmap texture to 3D mesh geometry -- pick a backend."""

    BACKEND_MAP = {
        "grid": "GeomPackTextureToGeometry_Grid",
        "poisson_pymeshlab": "GeomPackTextureToGeometry_PoissonPyMeshLab",
        "delaunay_2d": "GeomPackTextureToGeometry_Delaunay2D",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackTextureToGeometry",
            display_name="Depth Map to Mesh",
            category="geompack/texture_remeshing",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.DynamicCombo.Input("backend",
                    tooltip="Which reconstruction algorithm turns the depth pixels into a mesh:\n"
                            "- grid: fastest. One vertex per pixel, displaced in Z. Preserves the "
                            "pixel grid exactly but can show stair-step artifacts on slopes. The "
                            "only backend that can carry the optional 'field' map onto the mesh.\n"
                            "- poisson_pymeshlab: smooth watertight surface via PyMeshLab Screened "
                            "Poisson reconstruction. Slower; output topology is unrelated to the pixel grid.\n"
                            "- delaunay_2d: triangulates the pixels in the XY plane (scipy Delaunay), "
                            "keeping their Z displacement. No stair-steps, but no smoothing either.",
                    options=[
                        io.DynamicCombo.Option("grid", [
                            io.Combo.Input("smooth_normals", options=["true", "false"], default="true",
                                tooltip="Recompute consistent smooth vertex normals on the finished grid "
                                        "mesh (trimesh fix_normals). Disable to keep raw face normals."),
                        ]),
                        io.DynamicCombo.Option("poisson_pymeshlab", [
                            io.Int.Input("poisson_depth", default=8, min=4, max=12, step=1,
                                tooltip="Octree depth of the Poisson solver. Higher = finer detail and "
                                        "more triangles (roughly 4x work per +1); 8 is a good default."),
                        ]),
                        io.DynamicCombo.Option("delaunay_2d", []),
                    ]),
                io.MultiType.Input("depth", [io.Image, io.Mask],
                    tooltip="The depth / height map to convert. Each pixel becomes a point at "
                            "(x, y) in [-1, 1] with Z = pixel value x height_scale. Accepts an "
                            "IMAGE (RGB averaged to grayscale) or a MASK (single channel). "
                            "Values are expected in [0, 1] (0-255 inputs are auto-rescaled)."),
                io.Float.Input("height_scale", default=1.0, min=0.01, max=10000000.0, step=0.1,
                    display_mode="number",
                    tooltip="Multiplier for the Z displacement. The XY footprint of the mesh is "
                            "always [-1, 1], so height_scale=1.0 means a full-white pixel sits at "
                            "Z=1.0 (half the mesh width). Increase to exaggerate relief, decrease "
                            "to flatten it."),
                io.MultiType.Input("field", [io.Image, io.Mask], optional=True,
                    tooltip="Optional extra per-pixel map stored on the mesh as a vertex attribute "
                            "(named by field_name) -- e.g. a segmentation / part-ID / curvature map "
                            "you want carried onto the geometry for later Split-by-Field or field "
                            "coloring in viewers. Does NOT affect the shape. Must match the depth "
                            "map's resolution. Only the grid backend can carry it (its vertices map "
                            "1:1 to pixels); the other backends ignore it."),
                io.String.Input("field_name", default="field", optional=True,
                    tooltip="Name of the vertex attribute that the optional 'field' map is stored "
                            "under on the output mesh (e.g. 'part_id', 'curvature'). This is the "
                            "name you'll select in field-aware nodes/viewers downstream. Ignored "
                            "when 'field' is not connected."),
                io.Mask.Input("mask", optional=True,
                    tooltip="Optional MASK selecting which pixels become geometry: only pixels "
                            "where the mask is > 0.5 are used; masked-out pixels are skipped "
                            "entirely (leaving holes). Use this to cut away background around an "
                            "object. Independent of the depth values. If its resolution differs "
                            "from the depth map it is nearest-neighbour resized to match."),
                io.Combo.Input("invert_height", options=["false", "true"], default="false", optional=True,
                    tooltip="Flip the depth values (value -> 1 - value) before displacement. Use "
                            "when your depth map's convention is inverted (e.g. white = far "
                            "instead of white = near)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, backend=None, depth=None, height_scale=1.0,
                field=None, field_name="field", mask=None,
                invert_height="false"):
        from comfy_execution.graph_utils import GraphBuilder
        if cls.SCHEMA is None:
            cls.GET_SCHEMA()
        if backend is None:
            backend = {"backend": "grid"}
        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]
        kwargs = {
            "height_scale": height_scale,
            "depth": depth,
            "field_name": field_name,
            "invert_height": invert_height,
        }
        # optional link inputs: only pass when connected (None would fail validation)
        if field is not None:
            kwargs["field"] = field
        if mask is not None:
            kwargs["mask"] = mask
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


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackTextureToGeometry": TextureToGeometryNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackTextureToGeometry": "Depth Map to Mesh",
}
