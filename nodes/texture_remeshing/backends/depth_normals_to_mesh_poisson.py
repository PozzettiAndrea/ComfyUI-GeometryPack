# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Depth + Normals to Mesh - Poisson backend. Screened Poisson surface
reconstruction of the oriented point cloud (Open3D if available, else PyMeshLab)."""

import logging

import numpy as np
import trimesh
from comfy_api.latest import io

from ._depth_normals_to_mesh_common import build_oriented_point_cloud, build_info

log = logging.getLogger("geometrypack")


class DepthNormalsToMeshPoissonNode(io.ComfyNode):
    """Poisson backend for Depth + Normals to Mesh."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackDepthNormalsToMesh_Poisson",
            display_name="Depth + Normals to Mesh Poisson (backend)",
            category="geompack/texture_remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Image.Input("normal_map"),
                io.MultiType.Input("depth", [io.Image, io.Mask]),
                io.Int.Input("resolution", default=512, min=64, max=2048, step=64, optional=True),
                io.Float.Input("depth_scale", default=1.0, min=0.01, max=10.0, step=0.1, optional=True),
                io.Mask.Input("mask", optional=True),
                io.Int.Input("poisson_depth", default=8, min=4, max=12, step=1, optional=True),
                io.Float.Input("poisson_scale", default=1.1, min=1.0, max=2.0, step=0.1, optional=True),
                io.Float.Input("trim_factor", default=2.5, min=0.0, max=100.0, step=0.5, optional=True),
                io.Combo.Input("invert_depth", options=["false", "true"], default="false", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, normal_map=None, depth=None, resolution=512, depth_scale=1.0,
                mask=None, poisson_depth=8, poisson_scale=1.1, trim_factor=2.5,
                invert_depth="false"):
        points, normals, keep_mask, depth_shape, normal_shape, grid_shape = build_oriented_point_cloud(
            normal_map, depth, int(resolution), depth_scale, mask, invert_depth)

        mesh, method_info = cls._poisson_reconstruct(points, normals, int(poisson_depth), poisson_scale)
        log.info("Output: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))

        mesh, method_info = cls._trim_to_points(mesh, points, float(trim_factor), method_info)

        info = build_info(depth_shape, normal_shape, grid_shape, depth_scale, "poisson",
                          points, keep_mask, mesh, method_info)
        return io.NodeOutput(mesh, info, ui={"text": [info]})

    @staticmethod
    def _trim_to_points(mesh, points, trim_factor, method_info):
        """Drop faces farther than trim_factor x point-spacing from the input cloud.

        Poisson solves over a CUBIC octree domain and extrapolates a membrane into
        sample-free regions -- a rectangular cloud comes back square without this."""
        if trim_factor <= 0 or len(mesh.faces) == 0:
            return mesh, method_info
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        stride = max(1, len(points) // 5000)
        nn, _ = tree.query(points[::stride], k=2)
        spacing = float(np.median(nn[:, 1]))
        threshold = trim_factor * spacing

        vdist, _ = tree.query(mesh.vertices)
        keep_v = vdist <= threshold
        keep_f = keep_v[mesh.faces].all(axis=1)
        n_before = len(mesh.faces)
        if keep_f.all():
            return mesh, method_info + "\n  Trim: nothing to remove"
        mesh.update_faces(keep_f)
        mesh.remove_unreferenced_vertices()
        log.info("Trim: removed %d/%d extrapolated faces (threshold %.4g)",
                 n_before - len(mesh.faces), n_before, threshold)
        return mesh, method_info + (
            f"\n  Trim: removed {n_before - len(mesh.faces):,} extrapolated faces "
            f"(> {trim_factor:g} x point spacing from the input cloud)")

    @staticmethod
    def _poisson_reconstruct(points, normals, depth, scale):
        """Poisson surface reconstruction using Open3D or PyMeshLab."""
        # Try Open3D first
        try:
            import open3d as o3d

            log.info("Using Open3D Poisson reconstruction...")

            # Create point cloud with normals
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)

            # Poisson reconstruction
            mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, scale=scale, linear_fit=False
            )

            # Remove low density vertices (noise at boundaries)
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.01)
            vertices_to_remove = densities < density_threshold
            mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

            # Convert to trimesh
            result = trimesh.Trimesh(
                vertices=np.asarray(mesh_o3d.vertices),
                faces=np.asarray(mesh_o3d.triangles),
                process=False
            )

            method_info = f"""Poisson Reconstruction (Open3D):
  Octree Depth: {depth}
  Scale: {scale}
  Density Filtering: 1% quantile removed"""

            return result, method_info

        except ImportError:
            pass

        # Fallback to PyMeshLab
        try:
            import pymeshlab

            log.info("Using PyMeshLab Poisson reconstruction...")

            ms = pymeshlab.MeshSet()
            pml_mesh = pymeshlab.Mesh(
                vertex_matrix=points,
                v_normals_matrix=normals
            )
            ms.add_mesh(pml_mesh)

            # Poisson reconstruction
            ms.generate_surface_reconstruction_screened_poisson(
                depth=depth,
                scale=scale
            )

            result_mesh = ms.current_mesh()
            result = trimesh.Trimesh(
                vertices=result_mesh.vertex_matrix(),
                faces=result_mesh.face_matrix(),
                process=False
            )

            method_info = f"""Poisson Reconstruction (PyMeshLab):
  Octree Depth: {depth}
  Scale: {scale}"""

            return result, method_info

        except ImportError:
            raise ImportError(
                "Poisson reconstruction requires Open3D or PyMeshLab.\n"
                "Install with: pip install open3d  or  pip install pymeshlab"
            )


NODE_CLASS_MAPPINGS = {"GeomPackDepthNormalsToMesh_Poisson": DepthNormalsToMeshPoissonNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackDepthNormalsToMesh_Poisson": "Depth + Normals to Mesh Poisson (backend)"}
