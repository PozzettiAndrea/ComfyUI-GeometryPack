# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Depth Map to Mesh - 2D Delaunay backend. Triangulates the heightmap points in the
XY plane (scipy Delaunay), keeping original Z displacement. The 'field' input is
accepted (uniform dispatcher signature) but ignored."""

import logging

import trimesh
from comfy_api.latest import io

from ._texture_to_geometry_common import (
    extract_heightmap, pixel_keep_mask, heightmap_to_points, build_info,
)

log = logging.getLogger("geometrypack")


class TextureToGeometryDelaunay2DNode(io.ComfyNode):
    """2D Delaunay triangulation backend for Depth Map to Mesh."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackTextureToGeometry_Delaunay2D",
            display_name="Depth Map to Mesh Delaunay 2D (backend)",
            category="geompack/texture_remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Float.Input("height_scale", default=1.0, min=0.01, max=10000000.0, step=0.1, display_mode="number"),
                io.MultiType.Input("depth", [io.Image, io.Mask]),
                io.MultiType.Input("field", [io.Image, io.Mask], optional=True),
                io.String.Input("field_name", default="field", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Combo.Input("invert_height", options=["false", "true"], default="false", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, height_scale, depth=None, field=None, field_name="field", mask=None,
                invert_height="false"):
        heightmap = extract_heightmap(depth, invert_height)
        height, width = heightmap.shape

        keep_mask = pixel_keep_mask(mask, height, width)
        if keep_mask is not None:
            log.info("mask input: keeping %d/%d pixels (%.1f%%)",
                     int(keep_mask.sum()), keep_mask.size, 100.0 * keep_mask.mean())

        points, _valid = heightmap_to_points(heightmap, height_scale, keep_mask)
        log.info("Generated %d points", len(points))

        mesh = cls._build_delaunay_2d(points)

        log.info("Created mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
        info = build_info(width, height, height_scale, invert_height,
                          keep_mask, "delaunay_2d",
                          "2D Delaunay triangulation", mesh)
        return io.NodeOutput(mesh, info, ui={"text": [info]})

    @staticmethod
    def _build_delaunay_2d(points):
        """Build mesh using 2D Delaunay triangulation."""
        try:
            from scipy.spatial import Delaunay
        except ImportError:
            raise ImportError(
                "scipy is required for delaunay_2d backend.\n"
                "Install with: pip install scipy"
            )

        log.info("Using 2D Delaunay triangulation...")

        # Project to XY plane for triangulation
        points_2d = points[:, :2]
        tri = Delaunay(points_2d)

        # Create mesh with original 3D coordinates
        mesh = trimesh.Trimesh(
            vertices=points,
            faces=tri.simplices,
            process=False
        )

        return mesh


NODE_CLASS_MAPPINGS = {"GeomPackTextureToGeometry_Delaunay2D": TextureToGeometryDelaunay2DNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackTextureToGeometry_Delaunay2D": "Depth Map to Mesh Delaunay 2D (backend)"}
