"""
Conversion Nodes - Convert between mesh and point cloud representations
"""

import numpy as np
import trimesh


class StripMeshAdjacencyNode:
    """
    Strip mesh adjacency information, leaving only vertex positions.

    Removes all face connectivity data from the mesh, effectively converting
    it to a point cloud while preserving the original vertex positions.
    No sampling or resampling is performed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
            },
            "optional": {
                "include_normals": (["true", "false"], {
                    "default": "false"
                }),
            }
        }

    RETURN_TYPES = ("POINT_CLOUD",)
    RETURN_NAMES = ("point_cloud",)
    FUNCTION = "strip_adjacency"
    CATEGORY = "geompack/conversion"

    def strip_adjacency(self, mesh, include_normals="false"):
        """
        Strip face adjacency from mesh, keeping only vertices.

        Args:
            mesh: Input trimesh.Trimesh object
            include_normals: Whether to include vertex normals

        Returns:
            tuple: (point_cloud_dict,)
        """
        print(f"[StripMeshAdjacency] Stripping adjacency from mesh with {len(mesh.vertices):,} vertices")

        # Extract vertices
        points = mesh.vertices.copy()

        # Optional: include vertex normals
        normals = None
        if include_normals == "true" and hasattr(mesh, 'vertex_normals'):
            normals = mesh.vertex_normals.copy()
            print(f"[StripMeshAdjacency] Including vertex normals")

        # Create point cloud data structure
        pointcloud = {
            'points': points,
            'normals': normals,
            'source_mesh_vertices': len(mesh.vertices),
            'source_mesh_faces': len(mesh.faces),
            'stripped_adjacency': True,
        }

        print(f"[StripMeshAdjacency] Created point cloud with {len(points):,} points (original vertices, no sampling)")

        return (pointcloud,)


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
                "mesh": ("MESH",),
                "sample_count": ("INT", {
                    "default": 10000,
                    "min": 100,
                    "max": 10000000,
                    "step": 100
                }),
                "sampling_method": (["uniform", "even", "face_weighted"], {
                    "default": "uniform"
                }),
            },
            "optional": {
                "include_normals": (["true", "false"], {
                    "default": "true"
                }),
            }
        }

    RETURN_TYPES = ("POINT_CLOUD",)
    RETURN_NAMES = ("point_cloud",)
    FUNCTION = "mesh_to_pointcloud"
    CATEGORY = "geompack/conversion"

    def mesh_to_pointcloud(self, mesh, sample_count, sampling_method, include_normals="true"):
        """
        Sample points from mesh surface.

        Args:
            mesh: Input trimesh.Trimesh object
            sample_count: Number of points to sample
            sampling_method: Sampling strategy
            include_normals: Whether to compute surface normals

        Returns:
            tuple: (point_cloud_dict,)
        """
        print(f"[MeshToPointCloud] Sampling {sample_count:,} points using {sampling_method} method")

        if sampling_method == "uniform":
            # Uniform random sampling
            points, face_indices = mesh.sample(sample_count, return_index=True)

        elif sampling_method == "even":
            # Approximately even spacing (rejection sampling)
            # Calculate radius based on surface area and desired point count
            radius = np.sqrt(mesh.area / sample_count) * 2.0
            points, face_indices = trimesh.sample.sample_surface_even(
                mesh, sample_count, radius=radius
            )
            print(f"[MeshToPointCloud] Even sampling produced {len(points):,} points (target: {sample_count:,})")

        elif sampling_method == "face_weighted":
            # Weight by face area (default behavior)
            points, face_indices = mesh.sample(
                sample_count,
                return_index=True,
                face_weight=mesh.area_faces
            )

        # Optional: compute normals at sample points
        normals = None
        if include_normals == "true":
            # Get face normals for each sampled point
            normals = mesh.face_normals[face_indices]

        # Create point cloud data structure
        pointcloud = {
            'points': points,
            'normals': normals,
            'face_indices': face_indices,
            'source_mesh_vertices': len(mesh.vertices),
            'source_mesh_faces': len(mesh.faces),
            'sample_count': len(points),
            'sampling_method': sampling_method,
        }

        print(f"[MeshToPointCloud] Generated point cloud with {len(points):,} points")

        return (pointcloud,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackStripMeshAdjacency": StripMeshAdjacencyNode,
    "GeomPackMeshToPointCloud": MeshToPointCloudNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackStripMeshAdjacency": "Strip Mesh Adjacency",
    "GeomPackMeshToPointCloud": "Mesh to Point Cloud",
}
