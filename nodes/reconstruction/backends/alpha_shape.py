# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Alpha shape surface reconstruction backend node."""

import logging
from collections import Counter

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class ReconstructAlphaShapeNode(io.ComfyNode):
    """Alpha shape reconstruction using scipy Delaunay."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackReconstruct_AlphaShape",
            display_name="Reconstruct Alpha Shape (backend)",
            category="geompack/reconstruction",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("points"),
                io.Float.Input("alpha", default=0.0, min=0.0, max=100.0, step=0.01, tooltip="Radius threshold controlling which Delaunay tetrahedra are kept. Only tetrahedra with longest edge < 2*alpha are included. Larger = coarser shape with more fill, smaller = tighter fit with more holes. 0 = auto (10% of bounding box diagonal)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="reconstructed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, points, alpha=0.0):
        log.info("Backend: alpha_shape")
        vertices = np.asarray(points.vertices, dtype=np.float64)
        alpha_value = alpha

        log.info("Computing alpha shape...")

        if alpha_value == 0.0:
            bbox_diag = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
            alpha_value = bbox_diag * 0.1
            log.info("Auto alpha: %.4f", alpha_value)

        try:
            from scipy.spatial import Delaunay

            tri = Delaunay(vertices)

            valid_faces = []
            for simplex in tri.simplices:
                tet_verts = vertices[simplex]

                edges = []
                for i in range(4):
                    for j in range(i + 1, 4):
                        edges.append(np.linalg.norm(tet_verts[i] - tet_verts[j]))
                max_edge = max(edges)

                if max_edge < alpha_value * 2:
                    for i in range(4):
                        face = tuple(sorted([simplex[j] for j in range(4) if j != i]))
                        valid_faces.append(face)

            face_counts = Counter(valid_faces)
            boundary_faces = [list(f) for f, count in face_counts.items() if count == 1]

            if len(boundary_faces) == 0:
                raise ValueError("Alpha value too small, no faces generated")

            result = trimesh_module.Trimesh(
                vertices=vertices,
                faces=boundary_faces,
                process=True
            )

            info = f"""Reconstruct Surface Results (Alpha Shape):

Alpha Value: {alpha_value:.4f}

Input Points: {len(vertices):,}
Output Vertices: {len(result.vertices):,}
Output Faces: {len(result.faces):,}

Alpha shapes capture the overall shape with controllable detail level.
"""
            # Preserve metadata
            if hasattr(points, 'metadata') and points.metadata:
                result.metadata = points.metadata.copy()
            else:
                result.metadata = {}
            result.metadata['reconstruction'] = {
                'method': 'alpha_shape',
                'input_points': len(vertices),
                'output_vertices': len(result.vertices),
                'output_faces': len(result.faces),
            }

            return io.NodeOutput(result, info, ui={"text": [info]})

        except ImportError:
            raise ImportError("Alpha shape requires scipy. Install with: pip install scipy")


NODE_CLASS_MAPPINGS = {"GeomPackReconstruct_AlphaShape": ReconstructAlphaShapeNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackReconstruct_AlphaShape": "Reconstruct Alpha Shape (backend)"}
