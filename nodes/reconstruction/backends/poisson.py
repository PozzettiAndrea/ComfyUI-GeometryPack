# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Poisson surface reconstruction backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _estimate_oriented_normals(vertices: np.ndarray, radius: float) -> np.ndarray:
    """Estimate consistently oriented per-point normals.

    Uses Open3D when available (kNN + tangent-plane orientation propagation),
    falls back to point-cloud-utils' kNN normal estimation. Either path returns
    a (N, 3) float64 array.
    """
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=10)
        return np.asarray(pcd.normals, dtype=np.float64)
    except ImportError:
        pass

    import point_cloud_utils as pcu
    _, est_normals = pcu.estimate_point_cloud_normals_knn(vertices, num_neighbors=16)
    return np.asarray(est_normals, dtype=np.float64)


class ReconstructPoissonNode(io.ComfyNode):
    """Poisson surface reconstruction using Open3D or PyMeshLab."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackReconstruct_Poisson",
            display_name="Reconstruct Poisson (backend)",
            category="geompack/reconstruction",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("points"),
                io.Int.Input("poisson_depth", default=8, min=1, max=12, step=1, tooltip="Octree depth for the Poisson solver. Higher values capture finer detail but use exponentially more memory and time. 6=coarse, 8=balanced, 10+=high detail."),
                io.Float.Input("poisson_scale", default=1.1, min=1.0, max=2.0, step=0.1, tooltip="Scale factor for the reconstruction grid relative to the bounding box. Values >1.0 add padding to avoid boundary artifacts. 1.1 is usually sufficient."),
                io.Combo.Input("estimate_normals", options=["true", "false"], default="true", tooltip="Re-estimate point normals using k-nearest neighbors. Poisson reconstruction requires oriented normals — enable this if the input has no normals or unreliable normals."),
                io.Float.Input("normal_radius", default=0.1, min=0.001, max=10.0, step=0.01, tooltip="Search radius for normal estimation via k-nearest neighbors. Should be 2-3x the average point spacing. Only used when estimate_normals is true."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="reconstructed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, points, poisson_depth=8, poisson_scale=1.1,
                estimate_normals="true", normal_radius=0.1):
        log.info("Backend: poisson")
        vertices = np.asarray(points.vertices, dtype=np.float64)
        normals = None
        if hasattr(points, 'vertex_normals') and len(points.vertex_normals) > 0:
            normals = np.asarray(points.vertex_normals, dtype=np.float64)
            log.info("Using normals from input")

        do_estimate = estimate_normals == "true"
        depth = poisson_depth
        scale = poisson_scale

        # Try Open3D first
        try:
            import open3d as o3d

            log.info("Using Open3D Poisson reconstruction...")
            log.info("Step 1/5: Creating point cloud...")

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)

            log.info("Step 2/5: Estimating normals...")
            if normals is None or do_estimate:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=normal_radius, max_nn=30
                    )
                )
                log.info("Step 3/5: Orienting normals...")
                pcd.orient_normals_consistent_tangent_plane(k=10)
            else:
                pcd.normals = o3d.utility.Vector3dVector(normals)

            log.info("Step 4/5: Running Poisson reconstruction (depth=%d)... This may take a while.", depth)
            mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, scale=scale, linear_fit=False
            )

            # Remove low density vertices (noise)
            log.info("Step 5/5: Cleaning up mesh...")
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.01)
            vertices_to_remove = densities < density_threshold
            mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

            result = trimesh_module.Trimesh(
                vertices=np.asarray(mesh_o3d.vertices),
                faces=np.asarray(mesh_o3d.triangles),
                process=False
            )

            log.info("Done! Output: %s vertices, %s faces",
                     f"{len(result.vertices):,}", f"{len(result.faces):,}")

            info = f"""Reconstruct Surface Results (Poisson):

Engine: Open3D
Depth: {depth}
Scale: {scale}

Input Points: {len(vertices):,}
Output Vertices: {len(result.vertices):,}
Output Faces: {len(result.faces):,}

Watertight: {result.is_watertight}

Poisson reconstruction creates smooth, watertight surfaces.
"""
            # Preserve metadata
            if hasattr(points, 'metadata') and points.metadata:
                result.metadata = points.metadata.copy()
            else:
                result.metadata = {}
            result.metadata['reconstruction'] = {
                'method': 'poisson',
                'input_points': len(vertices),
                'output_vertices': len(result.vertices),
                'output_faces': len(result.faces),
            }

            return io.NodeOutput(result, info, ui={"text": [info]})

        except ImportError:
            pass
        except (RuntimeError, ValueError) as err:
            log.warning("Open3D Poisson failed: %s — falling back to PyMeshLab", err)

        # Fallback to PyMeshLab
        try:
            import pymeshlab

            log.info("Using PyMeshLab Poisson reconstruction...")

            # Screened Poisson requires consistently oriented normals. PyMeshLab's
            # compute_normal_for_point_clouds does not orient them, and on
            # macOS-arm64 the CGAL plugin aborts the process via _exit(0) on
            # unoriented input. Always estimate-and-orient up front.
            if normals is None or do_estimate:
                oriented_normals = _estimate_oriented_normals(vertices, normal_radius)
            else:
                oriented_normals = normals

            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(
                vertex_matrix=vertices,
                v_normals_matrix=oriented_normals,
            ))

            ms.generate_surface_reconstruction_screened_poisson(
                depth=depth,
                scale=scale
            )

            result_mesh = ms.current_mesh()
            result = trimesh_module.Trimesh(
                vertices=result_mesh.vertex_matrix(),
                faces=result_mesh.face_matrix(),
                process=False
            )

            info = f"""Reconstruct Surface Results (Poisson):

Engine: PyMeshLab
Depth: {depth}
Scale: {scale}

Input Points: {len(vertices):,}
Output Vertices: {len(result.vertices):,}
Output Faces: {len(result.faces):,}

Watertight: {result.is_watertight}
"""
            # Preserve metadata
            if hasattr(points, 'metadata') and points.metadata:
                result.metadata = points.metadata.copy()
            else:
                result.metadata = {}
            result.metadata['reconstruction'] = {
                'method': 'poisson',
                'input_points': len(vertices),
                'output_vertices': len(result.vertices),
                'output_faces': len(result.faces),
            }

            return io.NodeOutput(result, info, ui={"text": [info]})

        except ImportError:
            raise ImportError(
                "Poisson reconstruction requires Open3D or PyMeshLab.\n"
                "Install with: pip install open3d  or  pip install pymeshlab"
            )


NODE_CLASS_MAPPINGS = {"GeomPackReconstruct_Poisson": ReconstructPoissonNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackReconstruct_Poisson": "Reconstruct Poisson (backend)"}
