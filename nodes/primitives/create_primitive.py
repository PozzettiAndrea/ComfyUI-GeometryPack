# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Create Primitive Node - Create basic geometric shapes
"""

import logging
import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _create_cube(size: float = 1.0) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=[size, size, size])
    mesh.metadata['primitive_type'] = 'cube'
    mesh.metadata['size'] = size
    return mesh


def _create_sphere(radius: float = 1.0, subdivisions: int = 2) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.metadata['primitive_type'] = 'sphere'
    mesh.metadata['radius'] = radius
    mesh.metadata['subdivisions'] = subdivisions
    return mesh


def _create_plane(size: float = 1.0, subdivisions: int = 1) -> trimesh.Trimesh:
    n = subdivisions + 1
    s = size / 2.0

    x = np.linspace(-s, s, n)
    y = np.linspace(-s, s, n)
    xx, yy = np.meshgrid(x, y)

    vertices = np.stack([
        xx.flatten(),
        yy.flatten(),
        np.zeros(n * n)
    ], axis=1).astype(np.float64)

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            idx = i * n + j
            faces.append([idx, idx + n, idx + n + 1])
            faces.append([idx, idx + n + 1, idx + 1])

    faces = np.array(faces, dtype=np.int32)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.metadata['primitive_type'] = 'plane'
    mesh.metadata['size'] = size
    mesh.metadata['subdivisions'] = subdivisions
    return mesh


class CreatePrimitive(io.ComfyNode):
    """
    Create primitive geometry (cube, sphere, plane)
    Uses trimesh creation functions for high-quality primitives.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackCreatePrimitive",
            display_name="Create Primitive",
            category="geompack/primitives",
            inputs=[
                io.Combo.Input("shape", options=["cube", "sphere", "plane"], default="cube"),
                io.Float.Input("size", default=1.0, min=0.01, max=100.0, step=0.1),
                io.Int.Input("subdivisions", default=2, min=0, max=5, step=1, optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
            ],
        )

    @classmethod
    def execute(cls, shape, size, subdivisions=2):
        """
        Create a primitive mesh.

        Args:
            shape: Type of primitive (cube, sphere, plane)
            size: Size of the primitive
            subdivisions: Number of subdivisions (for sphere and plane)

        Returns:
            tuple: (trimesh.Trimesh,)
        """
        if shape == "cube":
            mesh = _create_cube(size)
        elif shape == "sphere":
            mesh = _create_sphere(radius=size/2.0, subdivisions=subdivisions)
        elif shape == "plane":
            mesh = _create_plane(size=size, subdivisions=subdivisions)
        else:
            raise ValueError(f"Unknown shape: {shape}")

        log.info("Created %s: %d vertices, %d faces", shape, len(mesh.vertices), len(mesh.faces))

        return io.NodeOutput(mesh)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackCreatePrimitive": CreatePrimitive,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackCreatePrimitive": "Create Primitive",
}
