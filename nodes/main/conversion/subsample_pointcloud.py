# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Subsample Point Cloud Node - Reduce point cloud density while preserving attributes
"""

import numpy as np
import trimesh


class SubsamplePointCloudNode:
    """
    Subsample a point cloud to reduce point count while preserving attributes.

    Supports multiple sampling methods and preserves colors, normals, and other vertex data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "point_cloud": ("TRIMESH",),
                "method": (["random", "uniform_grid", "farthest_point"], {
                    "default": "random",
                    "tooltip": "random: fast random selection. uniform_grid: voxel-based uniform spacing. farthest_point: maximize coverage (slow for large clouds)."
                }),
                "target_count": ("INT", {
                    "default": 100000,
                    "min": 100,
                    "max": 10000000,
                    "step": 1000,
                    "tooltip": "Target number of points to keep"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": "Random seed for reproducible results (random method only)"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("point_cloud",)
    FUNCTION = "subsample"
    CATEGORY = "geompack/conversion"

    def _random_subsample(self, vertices, target_count, seed):
        """Random subsampling - fast and simple."""
        np.random.seed(seed)
        indices = np.random.choice(len(vertices), size=target_count, replace=False)
        indices.sort()  # Keep spatial ordering
        return indices

    def _uniform_grid_subsample(self, vertices, target_count):
        """Voxel-based uniform subsampling for even spatial distribution."""
        # Calculate voxel size to achieve target count
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_size = bbox_max - bbox_min

        # Estimate voxel size based on target count and bounding box volume
        volume = np.prod(bbox_size)
        voxel_size = (volume / target_count) ** (1/3)

        # Quantize points to voxel grid
        voxel_indices = ((vertices - bbox_min) / voxel_size).astype(np.int32)

        # Use dictionary to keep one point per voxel (first encountered)
        voxel_to_point = {}
        for i, voxel in enumerate(voxel_indices):
            key = tuple(voxel)
            if key not in voxel_to_point:
                voxel_to_point[key] = i

        indices = np.array(sorted(voxel_to_point.values()))

        # If we got more points than target, randomly subsample
        if len(indices) > target_count:
            np.random.seed(42)
            indices = np.random.choice(indices, size=target_count, replace=False)
            indices.sort()

        return indices

    def _farthest_point_subsample(self, vertices, target_count):
        """Farthest point sampling for maximum coverage (slow for large clouds)."""
        n_points = len(vertices)

        # Start with a random point
        np.random.seed(42)
        indices = [np.random.randint(n_points)]

        # Track minimum distance to selected set for each point
        min_distances = np.full(n_points, np.inf)

        for _ in range(target_count - 1):
            # Update distances based on last selected point
            last_selected = vertices[indices[-1]]
            distances = np.linalg.norm(vertices - last_selected, axis=1)
            min_distances = np.minimum(min_distances, distances)

            # Select point with maximum minimum distance
            # Exclude already selected points
            min_distances[indices] = -1
            next_idx = np.argmax(min_distances)
            indices.append(next_idx)

            # Progress logging for slow operations
            if len(indices) % 10000 == 0:
                print(f"[SubsamplePointCloud] FPS progress: {len(indices):,}/{target_count:,}")

        return np.array(sorted(indices))

    def subsample(self, point_cloud, method, target_count, seed=42):
        """
        Subsample point cloud while preserving all vertex attributes.

        Args:
            point_cloud: Input trimesh.PointCloud or Trimesh object
            method: Subsampling method
            target_count: Target number of points
            seed: Random seed for reproducibility

        Returns:
            tuple: (subsampled_point_cloud,)
        """
        vertices = np.asarray(point_cloud.vertices)
        n_points = len(vertices)

        print(f"[SubsamplePointCloud] Input: {n_points:,} points, target: {target_count:,}")

        # If already at or below target, return as-is
        if n_points <= target_count:
            print(f"[SubsamplePointCloud] Point count already at or below target, returning unchanged")
            return (point_cloud,)

        # Get indices based on method
        if method == "random":
            indices = self._random_subsample(vertices, target_count, seed)
        elif method == "uniform_grid":
            indices = self._uniform_grid_subsample(vertices, target_count)
        elif method == "farthest_point":
            # Warn if using FPS on large clouds
            if n_points > 100000:
                print(f"[SubsamplePointCloud] WARNING: Farthest point sampling on {n_points:,} points will be slow. Consider 'random' or 'uniform_grid'.")
            indices = self._farthest_point_subsample(vertices, target_count)
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"[SubsamplePointCloud] Selected {len(indices):,} points using {method} method")

        # Extract subsampled vertices
        new_vertices = vertices[indices]

        # Create new point cloud
        new_cloud = trimesh.PointCloud(vertices=new_vertices)

        # Preserve vertex colors if present
        if hasattr(point_cloud, 'colors') and point_cloud.colors is not None:
            colors = np.asarray(point_cloud.colors)
            if len(colors) == n_points:
                new_cloud.colors = colors[indices]
                print(f"[SubsamplePointCloud] Preserved vertex colors")

        # Also check visual.vertex_colors (trimesh stores colors here sometimes)
        if hasattr(point_cloud, 'visual') and hasattr(point_cloud.visual, 'vertex_colors'):
            vc = point_cloud.visual.vertex_colors
            if vc is not None and len(vc) == n_points:
                new_cloud.colors = vc[indices]
                print(f"[SubsamplePointCloud] Preserved visual.vertex_colors")

        # Preserve vertex normals if present
        if hasattr(point_cloud, 'vertex_normals'):
            normals = point_cloud.vertex_normals
            if normals is not None and len(normals) == n_points:
                new_cloud.vertex_normals = normals[indices]
                print(f"[SubsamplePointCloud] Preserved vertex normals")

        # Preserve metadata
        if hasattr(point_cloud, 'metadata') and point_cloud.metadata:
            new_cloud.metadata = dict(point_cloud.metadata)
            new_cloud.metadata['subsampled'] = True
            new_cloud.metadata['subsample_method'] = method
            new_cloud.metadata['original_point_count'] = n_points
        else:
            new_cloud.metadata = {
                'is_point_cloud': True,
                'subsampled': True,
                'subsample_method': method,
                'original_point_count': n_points,
                'sample_count': len(indices)
            }

        print(f"[SubsamplePointCloud] Output: {len(new_vertices):,} points")

        return (new_cloud,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackSubsamplePointCloud": SubsamplePointCloudNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackSubsamplePointCloud": "Subsample Point Cloud",
}
