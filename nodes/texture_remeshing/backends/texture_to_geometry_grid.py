# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Depth Map to Mesh - grid backend. Fast grid-based displacement: one vertex per
pixel, displaced in Z. May show stair-step artifacts on slopes. The only backend
that carries the optional 'field' input onto the mesh (vertices map 1:1 to pixels)."""

import logging

import numpy as np
import trimesh
from comfy_api.latest import io

from ._texture_to_geometry_common import (
    _to_numpy, extract_heightmap, pixel_keep_mask, build_info,
)

log = logging.getLogger("geometrypack")


class TextureToGeometryGridNode(io.ComfyNode):
    """Grid-based displacement backend for Depth Map to Mesh."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackTextureToGeometry_Grid",
            display_name="Depth Map to Mesh Grid (backend)",
            category="geompack/texture_remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Float.Input("height_scale", default=1.0, min=0.01, max=10000000.0, step=0.1, display_mode="number"),
                io.MultiType.Input("depth", [io.Image, io.Mask]),
                io.MultiType.Input("field", [io.Image, io.Mask], optional=True),
                io.String.Input("field_name", default="field", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Combo.Input("smooth_normals", options=["true", "false"], default="true", optional=True),
                io.Combo.Input("invert_height", options=["false", "true"], default="false", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, height_scale, depth=None, field=None, field_name="field", mask=None,
                smooth_normals="true", invert_height="false"):
        heightmap = extract_heightmap(depth, invert_height)
        height, width = heightmap.shape

        keep_mask = pixel_keep_mask(mask, height, width)
        if keep_mask is not None:
            log.info("mask input: keeping %d/%d pixels (%.1f%%)",
                     int(keep_mask.sum()), keep_mask.size, 100.0 * keep_mask.mean())

        mesh = cls._build_grid_mesh(
            heightmap, height_scale, width, height,
            smooth_normals == "true",
            field, field_name, keep_mask
        )

        log.info("Created mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
        info = build_info(width, height, height_scale, invert_height,
                          keep_mask, "grid",
                          "Grid-based displacement mesh", mesh)
        return io.NodeOutput(mesh, info, ui={"text": [info]})

    @staticmethod
    def _build_grid_mesh(heightmap, height_scale, width, height,
                         smooth_normals,
                         field=None, field_name="field", keep_mask=None):
        """Build mesh using grid-based displacement (original algorithm)."""
        # Generate vertices
        vertices = []
        for y in range(height):
            for x in range(width):
                nx = (x / (width - 1)) * 2.0 - 1.0
                ny = (y / (height - 1)) * 2.0 - 1.0
                h = heightmap[y, x] * height_scale
                vertices.append([nx, ny, h])

        vertices = np.array(vertices, dtype=np.float32)

        # Per-pixel keep test: a triangle is emitted only if all 3 corner pixels are
        # kept (from the explicit mask input).
        pix_ok = np.ones((height, width), dtype=bool)
        if keep_mask is not None:
            pix_ok &= keep_mask
        drop_any = keep_mask is not None

        # Generate faces (triangles)
        faces = []
        for y in range(height - 1):
            for x in range(width - 1):
                i = y * width + x
                a = pix_ok[y, x]; b = pix_ok[y, x + 1]
                c = pix_ok[y + 1, x]; d = pix_ok[y + 1, x + 1]
                if a and b and c:
                    faces.append([i, i + 1, i + width])
                if b and d and c:
                    faces.append([i + 1, i + width + 1, i + width])

        faces = np.array(faces, dtype=np.int32)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Add field as vertex attribute if provided (IMAGE or MASK -> (H,W) scalar)
        if field is not None:
            field_arr = _to_numpy(field)
            if field_arr.ndim == 4:                 # IMAGE (B,H,W,C)
                field_arr = field_arr[0]
            if field_arr.ndim == 3:
                if field_arr.shape[2] in (3, 4):    # (H,W,C) -> grayscale
                    field_arr = np.mean(field_arr[:, :, :3], axis=2)
                elif field_arr.shape[2] == 1:
                    field_arr = field_arr[:, :, 0]
                else:                                # (B,H,W) mask
                    field_arr = field_arr[0]
            if field_arr.ndim > 2:
                field_arr = field_arr.squeeze()

            # Check resolution matches
            if field_arr.shape == (height, width):
                # Sample field at each vertex position (same order as vertices)
                field_values = field_arr.flatten()
                mesh.vertex_attributes[field_name] = field_values.astype(np.float32)
                log.info("Added vertex attribute '%s' with %d values, range: [%.3f, %.3f]",
                         field_name, len(field_values), field_values.min(), field_values.max())
            else:
                log.warning("Warning: field shape %s doesn't match heightmap (%d, %d), skipping",
                            field_arr.shape, height, width)

        # drop orphan vertices left by masked-out / skipped pixels (reindexes
        # vertex_attributes like the field too)
        if drop_any:
            try:
                mesh.remove_unreferenced_vertices()
            except Exception as e:
                log.debug("vertex cleanup skipped: %s", e)

        if smooth_normals:
            mesh.fix_normals()

        return mesh


NODE_CLASS_MAPPINGS = {"GeomPackTextureToGeometry_Grid": TextureToGeometryGridNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackTextureToGeometry_Grid": "Depth Map to Mesh Grid (backend)"}
