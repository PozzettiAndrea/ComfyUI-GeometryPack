# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Depth + Normals to Mesh - Ball Pivoting backend. Ball-pivoting surface
reconstruction of the oriented point cloud (PyMeshLab, auto radius). May leave
holes in regions with sparse points."""

import logging

import trimesh
from comfy_api.latest import io

from ._depth_normals_to_mesh_common import build_oriented_point_cloud, build_info

log = logging.getLogger("geometrypack")


class DepthNormalsToMeshBallPivotingNode(io.ComfyNode):
    """Ball Pivoting backend for Depth + Normals to Mesh."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackDepthNormalsToMesh_BallPivoting",
            display_name="Depth + Normals to Mesh Ball Pivoting (backend)",
            category="geompack/texture_remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Image.Input("normal_map"),
                io.MultiType.Input("depth", [io.Image, io.Mask]),
                io.Int.Input("resolution", default=512, min=64, max=2048, step=64, optional=True),
                io.Float.Input("depth_scale", default=1.0, min=0.01, max=10.0, step=0.1, optional=True),
                io.Mask.Input("mask", optional=True),
                io.Combo.Input("invert_depth", options=["false", "true"], default="false", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, normal_map=None, depth=None, resolution=512, depth_scale=1.0,
                mask=None, invert_depth="false"):
        points, normals, keep_mask, depth_shape, normal_shape, grid_shape = build_oriented_point_cloud(
            normal_map, depth, int(resolution), depth_scale, mask, invert_depth)

        mesh, method_info = cls._ball_pivoting_reconstruct(points, normals)
        log.info("Output: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))

        info = build_info(depth_shape, normal_shape, grid_shape, depth_scale, "ball_pivoting",
                          points, keep_mask, mesh, method_info)
        return io.NodeOutput(mesh, info, ui={"text": [info]})

    @staticmethod
    def _ball_pivoting_reconstruct(points, normals):
        """Ball pivoting algorithm using PyMeshLab."""
        try:
            import pymeshlab

            log.info("Using PyMeshLab Ball Pivoting...")

            ms = pymeshlab.MeshSet()
            pml_mesh = pymeshlab.Mesh(
                vertex_matrix=points,
                v_normals_matrix=normals
            )
            ms.add_mesh(pml_mesh)

            # Ball pivoting with auto radius
            ms.generate_surface_reconstruction_ball_pivoting()

            result_mesh = ms.current_mesh()
            result = trimesh.Trimesh(
                vertices=result_mesh.vertex_matrix(),
                faces=result_mesh.face_matrix(),
                process=False
            )

            method_info = """Ball Pivoting Reconstruction (PyMeshLab):
  Radius: auto
  Note: May have holes in regions with sparse points"""

            return result, method_info

        except ImportError:
            raise ImportError(
                "Ball pivoting requires PyMeshLab.\n"
                "Install with: pip install pymeshlab"
            )


NODE_CLASS_MAPPINGS = {"GeomPackDepthNormalsToMesh_BallPivoting": DepthNormalsToMeshBallPivotingNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackDepthNormalsToMesh_BallPivoting": "Depth + Normals to Mesh Ball Pivoting (backend)"}
