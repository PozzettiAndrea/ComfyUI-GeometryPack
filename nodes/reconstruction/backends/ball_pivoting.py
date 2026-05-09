# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Ball pivoting surface reconstruction backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class ReconstructBallPivotingNode(io.ComfyNode):
    """Ball pivoting algorithm using PyMeshLab."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackReconstruct_BallPivoting",
            display_name="Reconstruct Ball Pivoting (backend)",
            category="geompack/reconstruction",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("points"),
                io.Float.Input("ball_radius", default=0.0, min=0.0, max=100.0, step=0.01, tooltip="Ball radius in absolute units (same as mesh coordinates). Smaller = finer detail but more holes, larger = smoother but loses detail. 0 = auto (PyMeshLab estimates from point spacing)."),
                io.Combo.Input("estimate_normals", options=["true", "false"], default="true", tooltip="Re-estimate point normals using k-nearest neighbors. Enable if the input has no normals or unreliable normals."),
                io.Float.Input("normal_radius", default=0.1, min=0.001, max=10.0, step=0.01, tooltip="Search radius for normal estimation via k-nearest neighbors. Should be 2-3x the average point spacing. Only used when estimate_normals is true."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="reconstructed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, points, ball_radius=0.0, estimate_normals="true", normal_radius=0.1):
        log.info("Backend: ball_pivoting")
        vertices = np.asarray(points.vertices, dtype=np.float64)
        normals = None
        if hasattr(points, 'vertex_normals') and len(points.vertex_normals) > 0:
            normals = np.asarray(points.vertex_normals, dtype=np.float64)

        do_estimate = estimate_normals == "true"

        try:
            import pymeshlab

            log.info("Using PyMeshLab Ball Pivoting...")

            ms = pymeshlab.MeshSet()

            if normals is not None and not do_estimate:
                pml_mesh = pymeshlab.Mesh(
                    vertex_matrix=vertices,
                    v_normals_matrix=normals
                )
            else:
                pml_mesh = pymeshlab.Mesh(vertex_matrix=vertices)

            ms.add_mesh(pml_mesh)

            if normals is None or do_estimate:
                ms.compute_normal_for_point_clouds(k=10)

            if ball_radius == 0.0:
                ms.generate_surface_reconstruction_ball_pivoting()
            else:
                ms.generate_surface_reconstruction_ball_pivoting(
                    ballradius=pymeshlab.PureValue(ball_radius)
                )

            result_mesh = ms.current_mesh()
            result = trimesh_module.Trimesh(
                vertices=result_mesh.vertex_matrix(),
                faces=result_mesh.face_matrix(),
                process=False
            )

            info = f"""Reconstruct Surface Results (Ball Pivoting):

Engine: PyMeshLab
Ball Radius: {'auto' if ball_radius == 0.0 else f'{ball_radius:.3f}'}

Input Points: {len(vertices):,}
Output Vertices: {len(result.vertices):,}
Output Faces: {len(result.faces):,}

Ball pivoting preserves fine details but may have holes.
"""
            # Preserve metadata
            if hasattr(points, 'metadata') and points.metadata:
                result.metadata = points.metadata.copy()
            else:
                result.metadata = {}
            result.metadata['reconstruction'] = {
                'method': 'ball_pivoting',
                'input_points': len(vertices),
                'output_vertices': len(result.vertices),
                'output_faces': len(result.faces),
            }

            return io.NodeOutput(result, info, ui={"text": [info]})

        except ImportError:
            raise ImportError(
                "Ball pivoting requires PyMeshLab.\n"
                "Install with: pip install pymeshlab"
            )


NODE_CLASS_MAPPINGS = {"GeomPackReconstruct_BallPivoting": ReconstructBallPivotingNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackReconstruct_BallPivoting": "Reconstruct Ball Pivoting (backend)"}
