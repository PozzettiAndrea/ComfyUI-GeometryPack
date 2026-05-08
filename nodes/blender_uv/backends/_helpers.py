# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Shared helpers for Blender UV unwrapping backends."""

import numpy as np


def _extract_uvs_from_blender_mesh(mesh, vertices_np):
    """Extract UVs from a blender mesh, handling vertex splitting at seams."""
    uv_layer = mesh.uv_layers.active
    if uv_layer:
        loop_uvs = np.array([uv_layer.data[i].uv[:] for i in range(len(uv_layer.data))], dtype=np.float32)
        new_vertices = []
        new_faces = []
        uvs_per_loop = []
        vertex_map = {}

        for poly in mesh.polygons:
            new_face = []
            for loop_idx in poly.loop_indices:
                orig_vert_idx = mesh.loops[loop_idx].vertex_index
                uv = tuple(loop_uvs[loop_idx])
                key = (orig_vert_idx, uv)
                if key not in vertex_map:
                    vertex_map[key] = len(new_vertices)
                    new_vertices.append(vertices_np[orig_vert_idx].tolist())
                    uvs_per_loop.append(list(uv))
                new_face.append(vertex_map[key])
            new_faces.append(new_face)

        return new_vertices, new_faces, uvs_per_loop
    else:
        return vertices_np.tolist(), [list(p.vertices) for p in mesh.polygons], [[0.0, 0.0]] * len(vertices_np)
