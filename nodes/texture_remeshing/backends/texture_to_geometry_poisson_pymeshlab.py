# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Depth Map to Mesh - PyMeshLab Screened Poisson backend. Smooth surface
reconstruction from the heightmap point cloud; no stair-step artifacts.
The 'field' input is accepted (uniform dispatcher signature) but ignored --
reconstruction produces a new topology unrelated to the input pixels."""

import logging

import numpy as np
import trimesh
from comfy_api.latest import io

from ._texture_to_geometry_common import (
    extract_heightmap, pixel_keep_mask, heightmap_to_points, build_info,
)

log = logging.getLogger("geometrypack")


class TextureToGeometryPoissonPyMeshLabNode(io.ComfyNode):
    """PyMeshLab Screened Poisson backend for Depth Map to Mesh."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackTextureToGeometry_PoissonPyMeshLab",
            display_name="Depth Map to Mesh Poisson PyMeshLab (backend)",
            category="geompack/texture_remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Float.Input("height_scale", default=1.0, min=0.01, max=10000000.0, step=0.1, display_mode="number"),
                io.MultiType.Input("depth", [io.Image, io.Mask]),
                io.MultiType.Input("field", [io.Image, io.Mask], optional=True),
                io.String.Input("field_name", default="field", optional=True),
                io.Mask.Input("mask", optional=True),
                io.Int.Input("poisson_depth", default=8, min=4, max=12, step=1, optional=True),
                io.Combo.Input("invert_height", options=["false", "true"], default="false", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, height_scale, depth=None, field=None, field_name="field", mask=None,
                poisson_depth=8, invert_height="false"):
        heightmap = extract_heightmap(depth, invert_height)
        height, width = heightmap.shape

        keep_mask = pixel_keep_mask(mask, height, width)
        if keep_mask is not None:
            log.info("mask input: keeping %d/%d pixels (%.1f%%)",
                     int(keep_mask.sum()), keep_mask.size, 100.0 * keep_mask.mean())

        points, _valid = heightmap_to_points(heightmap, height_scale, keep_mask)
        log.info("Generated %d points", len(points))

        mesh = cls._build_poisson_pymeshlab(points, poisson_depth)

        log.info("Created mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
        info = build_info(width, height, height_scale, invert_height,
                          keep_mask, "poisson_pymeshlab",
                          f"PyMeshLab Screened Poisson reconstruction (depth={poisson_depth})", mesh)
        return io.NodeOutput(mesh, info, ui={"text": [info]})

    @staticmethod
    def _build_poisson_pymeshlab(points, depth):
        """Build mesh using PyMeshLab Screened Poisson reconstruction."""
        try:
            import pymeshlab
        except ImportError:
            raise ImportError(
                "PyMeshLab is required for poisson_pymeshlab backend.\n"
                "Install with: pip install pymeshlab"
            )

        log.info("Using PyMeshLab Screened Poisson reconstruction...")

        # Create MeshSet and add point cloud
        ms = pymeshlab.MeshSet()
        pml_mesh = pymeshlab.Mesh(vertex_matrix=points)
        ms.add_mesh(pml_mesh)

        # Estimate normals for point cloud
        log.info("Estimating normals...")
        ms.compute_normal_for_point_clouds(k=10)

        # For depth maps, normals should point "up" (positive Z)
        # Check and flip if needed
        current_mesh = ms.current_mesh()
        normals = current_mesh.vertex_normal_matrix()
        if np.mean(normals[:, 2]) < 0:
            ms.meshing_invert_face_orientation()

        # Screened Poisson reconstruction
        log.info("Running Screened Poisson reconstruction (depth=%d)...", depth)
        ms.generate_surface_reconstruction_screened_poisson(
            depth=depth,
            scale=1.1
        )

        # Get result mesh
        result_mesh = ms.current_mesh()
        vertices = result_mesh.vertex_matrix()
        faces = result_mesh.face_matrix()

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False
        )

        return mesh


NODE_CLASS_MAPPINGS = {"GeomPackTextureToGeometry_PoissonPyMeshLab": TextureToGeometryPoissonPyMeshLabNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackTextureToGeometry_PoissonPyMeshLab": "Depth Map to Mesh Poisson PyMeshLab (backend)"}
