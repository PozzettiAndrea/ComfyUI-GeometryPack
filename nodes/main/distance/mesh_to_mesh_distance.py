# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Mesh Distance Node - Compare two meshes using various metrics
"""

import numpy as np


class MeshDistanceNode:
    """
    Unified Mesh Distance - Compare two meshes using various metrics.

    Metrics:
    - hausdorff: Maximum distance (worst-case deviation)
    - chamfer: Average squared distance (overall similarity)

    Both metrics sample points from mesh surfaces and compute distances.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_a": ("TRIMESH",),
                "mesh_b": ("TRIMESH",),
                "metric": (["hausdorff", "chamfer"], {"default": "hausdorff"}),
            },
            "optional": {
                "sample_count": ("INT", {
                    "default": 10000,
                    "min": 1000,
                    "max": 1000000,
                    "step": 1000
                }),
                "symmetric": (["true", "false"], {"default": "true"}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("distance", "info")
    OUTPUT_NODE = True
    FUNCTION = "compute_distance"
    CATEGORY = "geompack/distance"

    def compute_distance(self, mesh_a, mesh_b, metric,
                         sample_count=10000, symmetric="true"):
        """
        Compute distance metric between two meshes.

        Args:
            mesh_a: First mesh
            mesh_b: Second mesh
            metric: Distance metric to use
            sample_count: Number of points to sample
            symmetric: Whether to compute symmetric distance

        Returns:
            tuple: (distance_value, info_string)
        """
        try:
            import point_cloud_utils as pcu
        except ImportError:
            raise ImportError(
                "point-cloud-utils not installed. Install with: pip install point-cloud-utils"
            )

        print(f"[MeshDistance] Metric: {metric}, Samples: {sample_count}")
        print(f"[MeshDistance] Mesh A: {len(mesh_a.vertices)} vertices, {len(mesh_a.faces)} faces")
        print(f"[MeshDistance] Mesh B: {len(mesh_b.vertices)} vertices, {len(mesh_b.faces)} faces")

        # Sample point clouds from meshes
        points_a = mesh_a.sample(sample_count)
        points_b = mesh_b.sample(sample_count)

        if metric == "hausdorff":
            if symmetric == "true":
                # Symmetric Hausdorff distance
                dist = pcu.hausdorff_distance(points_a, points_b)
                dist = dist[0] if isinstance(dist, tuple) else dist

                # Also compute one-sided for info
                hd_a_to_b = pcu.one_sided_hausdorff_distance(points_a, points_b)
                hd_a_to_b = hd_a_to_b[0] if isinstance(hd_a_to_b, tuple) else hd_a_to_b
                hd_b_to_a = pcu.one_sided_hausdorff_distance(points_b, points_a)
                hd_b_to_a = hd_b_to_a[0] if isinstance(hd_b_to_a, tuple) else hd_b_to_a

                info = f"""Mesh Distance Results (Hausdorff):

Symmetric Distance: {dist:.6f}
A → B (one-sided): {hd_a_to_b:.6f}
B → A (one-sided): {hd_b_to_a:.6f}

Samples: {sample_count:,} points per mesh
Mesh A: {len(mesh_a.vertices):,} vertices, {len(mesh_a.faces):,} faces
Mesh B: {len(mesh_b.vertices):,} vertices, {len(mesh_b.faces):,} faces

Hausdorff distance measures the maximum deviation (worst-case error).
"""
            else:
                # One-sided: A → B
                dist = pcu.one_sided_hausdorff_distance(points_a, points_b)
                dist = dist[0] if isinstance(dist, tuple) else dist

                info = f"""Mesh Distance Results (Hausdorff - One-Sided):

Distance A → B: {dist:.6f}

Samples: {sample_count:,} points per mesh
Mesh A: {len(mesh_a.vertices):,} vertices, {len(mesh_a.faces):,} faces
Mesh B: {len(mesh_b.vertices):,} vertices, {len(mesh_b.faces):,} faces
"""

        elif metric == "chamfer":
            dist = pcu.chamfer_distance(points_a, points_b)

            info = f"""Mesh Distance Results (Chamfer):

Chamfer Distance: {dist:.6f}

Samples: {sample_count:,} points per mesh
Mesh A: {len(mesh_a.vertices):,} vertices, {len(mesh_a.faces):,} faces
Mesh B: {len(mesh_b.vertices):,} vertices, {len(mesh_b.faces):,} faces

Chamfer distance measures average nearest-neighbor distance (overall similarity).
"""

        else:
            raise ValueError(f"Unknown metric: {metric}")

        print(f"[MeshDistance] Result: {dist:.6f}")
        return {
            "result": (float(dist), info),
            "ui": {"text": [info]}
        }


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackMeshToMeshDistance": MeshDistanceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackMeshToMeshDistance": "Mesh to Mesh Distance",
}
