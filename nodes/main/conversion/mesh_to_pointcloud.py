# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Mesh to Point Cloud Node - Sample points from mesh surface
"""

import numpy as np
import trimesh


class MeshToPointCloudNode:
    """
    Convert mesh to point cloud by sampling surface points.

    Samples points from the mesh surface using various sampling methods.
    Can optionally include normals and colors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "mode": (["strip_adjacency", "surface_sampling"], {
                    "default": "strip_adjacency",
                    "tooltip": "strip_adjacency: use mesh vertices directly. surface_sampling: sample points from surface."
                }),
            },
            "optional": {
                "sample_count": ("INT", {
                    "default": 10000,
                    "min": 100,
                    "max": 10000000,
                    "step": 100,
                    "tooltip": "Number of points to sample (only for surface_sampling mode)"
                }),
                "sampling_method": (["uniform", "even", "face_weighted"], {
                    "default": "uniform",
                    "tooltip": "Sampling strategy (only for surface_sampling mode)"
                }),
                "include_normals": (["true", "false"], {
                    "default": "true"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)  # Changed from POINT_CLOUD to TRIMESH for compatibility
    RETURN_NAMES = ("point_cloud",)
    FUNCTION = "mesh_to_pointcloud"
    CATEGORY = "geompack/conversion"

    def mesh_to_pointcloud(self, trimesh, mode, sample_count=10000, sampling_method="uniform", include_normals="true"):
        """
        Convert mesh to point cloud.

        Args:
            trimesh: Input trimesh.Trimesh object
            mode: "strip_adjacency" to use vertices directly, "surface_sampling" to sample surface
            sample_count: Number of points to sample (only for surface_sampling)
            sampling_method: Sampling strategy (only for surface_sampling)
            include_normals: Whether to compute surface normals

        Returns:
            tuple: (point_cloud_as_trimesh,) - TRIMESH with vertices only (no faces)
        """
        import trimesh as trimesh_module

        face_indices = None
        normals = None

        if mode == "strip_adjacency":
            # Simply use mesh vertices directly (strip face adjacency)
            points = np.asarray(trimesh.vertices, dtype=np.float32)
            print(f"[MeshToPointCloud] Strip adjacency: extracted {len(points):,} vertices")

            # Use vertex normals if available and requested
            if include_normals == "true" and hasattr(trimesh, 'vertex_normals'):
                normals = trimesh.vertex_normals

        else:  # surface_sampling
            print(f"[MeshToPointCloud] Sampling {sample_count:,} points using {sampling_method} method")

            if sampling_method == "uniform":
                # Uniform random sampling
                points, face_indices = trimesh.sample(sample_count, return_index=True)

            elif sampling_method == "even":
                # Approximately even spacing (rejection sampling)
                # Calculate radius based on surface area and desired point count
                radius = np.sqrt(trimesh.area / sample_count) * 2.0
                points, face_indices = trimesh_module.sample.sample_surface_even(
                    trimesh, sample_count, radius=radius
                )
                print(f"[MeshToPointCloud] Even sampling produced {len(points):,} points (target: {sample_count:,})")

            elif sampling_method == "face_weighted":
                # Weight by face area (default behavior)
                points, face_indices = trimesh.sample(
                    sample_count,
                    return_index=True,
                    face_weight=trimesh.area_faces
                )

            # Compute normals at sample points from face normals
            if include_normals == "true" and face_indices is not None:
                normals = trimesh.face_normals[face_indices]

        # Create point cloud as TRIMESH object (vertices only, no faces)
        # This ensures compatibility with all TRIMESH-expecting nodes
        point_cloud = trimesh_module.PointCloud(vertices=points)

        # Add normals as vertex_normals if computed
        if normals is not None:
            point_cloud.vertex_normals = normals

        # Store point cloud metadata
        point_cloud.metadata = {
            'is_point_cloud': True,
            'mode': mode,
            'face_indices': face_indices,
            'source_mesh_vertices': len(trimesh.vertices),
            'source_mesh_faces': len(trimesh.faces),
            'sample_count': len(points),
            'sampling_method': sampling_method if mode == "surface_sampling" else None,
            'has_normals': normals is not None
        }

        print(f"[MeshToPointCloud] Generated point cloud with {len(points):,} points")

        return (point_cloud,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackMeshToPointCloud": MeshToPointCloudNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackMeshToPointCloud": "Mesh to Point Cloud",
}
