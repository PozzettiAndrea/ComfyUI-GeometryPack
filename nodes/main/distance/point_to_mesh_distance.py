# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Point to Mesh Distance Node - Compute distances from points to mesh surface
"""

import numpy as np


class PointToMeshDistanceNode:
    """
    Point to Mesh Distance - Compute distance field from point cloud/mesh to target mesh surface.

    For each vertex in the input (point cloud or mesh), finds the closest point on the
    target mesh surface and computes the distance. Returns the input geometry with a
    'distance' field added to vertex_attributes.

    Supports both:
    - Unsigned distances (using trimesh): Always positive, measures surface proximity
    - Signed distances (using igl.signed_distance): Positive outside, negative inside mesh

    Useful for proximity analysis, error measurements, distance-based visualizations,
    and implicit surface representations (SDF).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_mesh": ("TRIMESH",),
                "pointcloud": ("TRIMESH",),
                "distance_type": (["unsigned", "signed"],),
                "sign_method": (["default", "winding_number", "fast_winding_number", "pseudonormal", "unsigned"],),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("pointcloud", "info")
    OUTPUT_NODE = True
    FUNCTION = "compute_distance"
    CATEGORY = "geompack/distance"

    def compute_distance(self, target_mesh, pointcloud, distance_type="unsigned", sign_method="default"):
        """
        Compute distances from point cloud/mesh vertices to target mesh surface.

        Args:
            target_mesh: Target trimesh.Trimesh object to measure distance to
            pointcloud: Input point cloud or mesh (TRIMESH) to compute distances for
            distance_type: Type of distance computation - "unsigned" (trimesh) or "signed" (igl)
            sign_method: Method for computing sign (only used for signed distances):
                - "default": Fast and robust sign computation
                - "winding_number": Accurate, handles non-watertight meshes
                - "fast_winding_number": Faster winding number approximation
                - "pseudonormal": Legacy pseudonormal test
                - "unsigned": Unsigned distance only (always positive)

        Returns:
            tuple: (pointcloud_with_distance_field, info_string)
        """
        # Extract vertices from input (works for both mesh and point cloud)
        points = pointcloud.vertices

        # Determine if input is a mesh or point cloud
        is_mesh = hasattr(pointcloud, 'faces') and len(pointcloud.faces) > 0
        input_type = "Mesh" if is_mesh else "Point Cloud"

        print(f"[PointToMeshDistance] Computing {distance_type} distances for {len(points):,} points")
        print(f"[PointToMeshDistance] Input: {input_type}")
        print(f"[PointToMeshDistance] Target Mesh: {len(target_mesh.vertices):,} vertices, {len(target_mesh.faces):,} faces")

        # Compute distances based on selected type
        if distance_type == "signed":
            try:
                import igl
            except (ImportError, OSError):
                raise ImportError("libigl is required for signed distance. Install with: pip install libigl")

            # Map sign method to igl enum
            sign_type_map = {
                "default": igl.SIGNED_DISTANCE_TYPE_DEFAULT,
                "winding_number": igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER,
                "fast_winding_number": igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
                "pseudonormal": igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
                "unsigned": igl.SIGNED_DISTANCE_TYPE_UNSIGNED,
            }
            igl_sign_type = sign_type_map.get(sign_method, igl.SIGNED_DISTANCE_TYPE_DEFAULT)

            print(f"[PointToMeshDistance] Using igl.signed_distance with sign method: {sign_method}")

            # Use igl's signed distance function
            # Returns: S (signed distances), I (closest face indices), C (closest points), N (normals)
            distances, face_indices, closest_points, normals = igl.signed_distance(
                points.astype(np.float64),
                target_mesh.vertices.astype(np.float64),
                target_mesh.faces.astype(np.int64),
                sign_type=igl_sign_type
            )
        else:
            # Use trimesh's proximity query to find closest points and distances (unsigned)
            import trimesh
            print(f"[PointToMeshDistance] Using trimesh.proximity.closest_point (unsigned)")
            closest_points, distances, triangle_ids = trimesh.proximity.closest_point(target_mesh, points)

        # Create a copy of the input to add distance field
        result = pointcloud.copy()

        # Ensure vertex_attributes exists (PointCloud objects don't have it by default)
        if not hasattr(result, 'vertex_attributes'):
            result.vertex_attributes = {}

        # Add distance field to vertex attributes
        result.vertex_attributes['distance'] = distances.astype(np.float32)

        # Add metadata
        if not hasattr(result, 'metadata') or result.metadata is None:
            result.metadata = {}

        result.metadata['has_distance_field'] = True
        result.metadata['distance_type'] = distance_type
        result.metadata['sign_method'] = sign_method if distance_type == "signed" else None
        result.metadata['target_mesh_vertices'] = len(target_mesh.vertices)
        result.metadata['target_mesh_faces'] = len(target_mesh.faces)
        result.metadata['num_points'] = len(points)

        # Compute statistics for info string
        min_dist = float(np.min(distances))
        max_dist = float(np.max(distances))
        mean_dist = float(np.mean(distances))
        median_dist = float(np.median(distances))
        std_dist = float(np.std(distances))

        # Find percentiles
        percentile_25 = float(np.percentile(distances, 25))
        percentile_75 = float(np.percentile(distances, 75))
        percentile_95 = float(np.percentile(distances, 95))

        # Build distance type info
        distance_info = f"Distance Type: {distance_type.upper()}"
        if distance_type == "signed":
            distance_info += f" (sign method: {sign_method})"
            sign_note = "\n  Note: Positive = outside, Negative = inside"
            # Count points inside/outside for signed distances
            num_outside = np.sum(distances > 0)
            num_inside = np.sum(distances < 0)
            num_on_surface = np.sum(np.abs(distances) < 1e-6)
        else:
            sign_note = ""

        # Count points within certain distance thresholds
        if distance_type == "signed":
            # For signed, use absolute values for thresholds
            abs_distances = np.abs(distances)
            threshold_01 = np.sum(abs_distances < 0.1)
            threshold_05 = np.sum(abs_distances < 0.5)
            threshold_10 = np.sum(abs_distances < 1.0)
        else:
            threshold_01 = np.sum(distances < 0.1)
            threshold_05 = np.sum(distances < 0.5)
            threshold_10 = np.sum(distances < 1.0)

        info = f"""Point to Mesh Distance Field:

Input:
  {input_type}: {len(points):,} {'vertices' if is_mesh else 'points'}
  Target Mesh: {len(target_mesh.vertices):,} vertices, {len(target_mesh.faces):,} faces
  {distance_info}{sign_note}

Distance Statistics:
  Minimum: {min_dist:.6f}
  Maximum: {max_dist:.6f}
  Mean: {mean_dist:.6f}
  Median: {median_dist:.6f}
  Std Dev: {std_dist:.6f}

Percentiles:
  25th: {percentile_25:.6f}
  75th: {percentile_75:.6f}
  95th: {percentile_95:.6f}
"""

        # Add inside/outside counts for signed distances
        if distance_type == "signed":
            info += f"""
Point Classification:
  Outside (d > 0): {num_outside:,} points ({100.0 * num_outside / len(points):.1f}%)
  Inside (d < 0): {num_inside:,} points ({100.0 * num_inside / len(points):.1f}%)
  On surface (d ~= 0): {num_on_surface:,} points ({100.0 * num_on_surface / len(points):.1f}%)
"""

        # Add distance distribution using absolute values for signed mode
        threshold_label = "|distance|" if distance_type == "signed" else "distance"
        info += f"""
Distance Distribution ({threshold_label}):
  < 0.1: {threshold_01:,} points ({100.0 * threshold_01 / len(points):.1f}%)
  < 0.5: {threshold_05:,} points ({100.0 * threshold_05 / len(points):.1f}%)
  < 1.0: {threshold_10:,} points ({100.0 * threshold_10 / len(points):.1f}%)

Output: {input_type} with 'distance' field in vertex_attributes
"""

        print(f"[PointToMeshDistance] Min: {min_dist:.6f}, Max: {max_dist:.6f}, Mean: {mean_dist:.6f}")
        print(f"[PointToMeshDistance] Distance field added to vertex_attributes['distance']")

        return {
            "result": (result, info),
            "ui": {"text": [info]}
        }


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackPointToMeshDistance": PointToMeshDistanceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPointToMeshDistance": "Point to Mesh Distance",
}
