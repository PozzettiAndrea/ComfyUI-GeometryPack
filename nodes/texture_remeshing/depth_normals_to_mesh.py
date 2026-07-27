# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Depth + Normals to Mesh - unified frontend with backend selection.

Builds an oriented point cloud from a depth map (positions) + a normal map
(orientations) and surface-reconstructs it via the selected backend, each a
separate hidden node dispatched through GraphBuilder:
- poisson: Screened Poisson reconstruction (smooth, watertight; Open3D if
  installed, else PyMeshLab)
- ball_pivoting: PyMeshLab ball pivoting (interpolating; may leave holes)

Using real normals (instead of normals estimated from depth) eliminates the
stair-step artifacts of grid displacement and the estimation noise of
depth-only reconstruction -- built for CAD raytracing / renderer outputs where
both passes are available.
"""

from comfy_api.latest import io


class DepthNormalsToMeshNode(io.ComfyNode):
    """Depth map + normal map -> smooth reconstructed mesh -- pick a backend."""

    BACKEND_MAP = {
        "poisson": "GeomPackDepthNormalsToMesh_Poisson",
        "ball_pivoting": "GeomPackDepthNormalsToMesh_BallPivoting",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackDepthNormalsToMesh",
            display_name="Depth + Normals to Mesh",
            category="geompack/texture_remeshing",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.DynamicCombo.Input("backend",
                    tooltip="Which surface-reconstruction algorithm turns the oriented point "
                            "cloud (depth = positions, normals = orientations) into a mesh:\n"
                            "- poisson: Screened Poisson reconstruction -- smooth watertight "
                            "surface, robust to noise. Uses Open3D when installed (with "
                            "low-density boundary trimming), otherwise PyMeshLab.\n"
                            "- ball_pivoting: PyMeshLab ball pivoting with auto radius -- "
                            "interpolates the actual points (no smoothing), but may leave holes "
                            "where points are sparse.",
                    options=[
                        io.DynamicCombo.Option("poisson", [
                            io.Int.Input("poisson_depth", default=8, min=4, max=12, step=1,
                                tooltip="Octree depth of the Poisson solver. Higher = finer detail "
                                        "and more triangles (roughly 4x work per +1); 8 is a good "
                                        "default."),
                            io.Float.Input("poisson_scale", default=1.1, min=1.0, max=2.0, step=0.1,
                                tooltip="Scale factor of the Poisson reconstruction bounding box "
                                        "relative to the point cloud's."),
                            io.Float.Input("trim_factor", default=2.5, min=0.0, max=100.0, step=0.5,
                                tooltip="Poisson solves over a CUBIC domain and extrapolates a "
                                        "membrane into regions with no sample points -- a "
                                        "rectangular image comes back as a square surface. Faces "
                                        "farther than trim_factor x the point spacing from the "
                                        "input cloud are removed, restoring the true footprint. "
                                        "0 disables trimming (keep the full extrapolated surface)."),
                        ]),
                        io.DynamicCombo.Option("ball_pivoting", []),
                    ]),
                io.Image.Input("normal_map",
                    tooltip="RGB normal map with Nx in R and Ny in G (Nz is derived from the "
                            "unit-normal constraint). Gives each point its orientation for the "
                            "surface reconstruction."),
                io.MultiType.Input("depth", [io.Image, io.Mask],
                    tooltip="The depth / height map. Accepts an IMAGE (RGB averaged to "
                            "grayscale) OR a MASK (single channel). Normalised to [0,1], then "
                            "each pixel becomes a 3D point at Z = value x depth_scale."),
                io.Int.Input("resolution", default=512, min=64, max=2048, step=64,
                    tooltip="Point-sampling density: the depth and normal maps are resampled so "
                            "their LONGEST side is at most this many pixels (aspect ratio "
                            "preserved, never upsampled past the input's native size), and one "
                            "oriented 3D point is emitted per sampled pixel. Higher = more "
                            "detail but slower reconstruction (512 -> up to ~260k points)."),
                io.Float.Input("depth_scale", default=1.0, min=0.01, max=10.0, step=0.1,
                    tooltip="Scale factor for depth values"),
                io.Mask.Input("mask", optional=True,
                    tooltip="Optional MASK selecting which pixels become points: only pixels "
                            "where the mask is > 0.5 are used (e.g. cut away background around "
                            "an object). Nearest-neighbour resized to the sampling resolution "
                            "if it differs."),
                io.Combo.Input("invert_depth", options=["false", "true"], default="false",
                    optional=True,
                    tooltip="Invert depth values (for depth maps where white = far)"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, backend=None, normal_map=None, depth=None, resolution=512,
                depth_scale=1.0, mask=None, invert_depth="false"):
        from comfy_execution.graph_utils import GraphBuilder
        if cls.SCHEMA is None:
            cls.GET_SCHEMA()
        if backend is None:
            backend = {"backend": "poisson"}
        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]
        kwargs = {
            "normal_map": normal_map,
            "depth": depth,
            "resolution": resolution,
            "depth_scale": depth_scale,
            "invert_depth": invert_depth,
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
            "result": (backend_node.out(0), backend_node.out(1)),
            "expand": graph.finalize(),
        }


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackDepthNormalsToMesh": DepthNormalsToMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackDepthNormalsToMesh": "Depth + Normals to Mesh",
}
