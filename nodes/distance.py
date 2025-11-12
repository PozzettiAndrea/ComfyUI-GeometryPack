"""
Distance Nodes - Distance metrics and signed distance fields
"""

import numpy as np


class HausdorffDistanceNode:
    """
    Compute Hausdorff distance between two meshes or point clouds.

    Hausdorff distance measures the maximum distance from any point in one set
    to its nearest point in the other set. Useful for measuring worst-case
    deviation between meshes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh_a": ("TRIMESH",),
                "trimesh_b": ("TRIMESH",),
                "sample_count": ("INT", {
                    "default": 10000,
                    "min": 1000,
                    "max": 1000000,
                    "step": 1000
                }),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("hausdorff_distance", "details")
    FUNCTION = "compute_distance"
    CATEGORY = "geompack/analysis/distance"

    def compute_distance(self, trimesh_a, trimesh_b, sample_count):
        """
        Compute Hausdorff distance between two meshes.

        Args:
            trimesh_a: First trimesh.Trimesh object
            trimesh_b: Second trimesh.Trimesh object
            sample_count: Number of points to sample from each mesh

        Returns:
            tuple: (hausdorff_distance, details_string)
        """
        try:
            import point_cloud_utils as pcu
        except ImportError:
            raise ImportError(
                "point-cloud-utils not installed. Install with: pip install point-cloud-utils"
            )

        print(f"[HausdorffDistance] Comparing meshes with {sample_count} samples each")

        # Sample point clouds from meshes
        points_a = trimesh_a.sample(sample_count)
        points_b = trimesh_b.sample(sample_count)

        # Compute Hausdorff distance (symmetric)
        hd = pcu.hausdorff_distance(points_a, points_b)
        # Handle potential tuple return (some versions return tuple)
        hd = hd[0] if isinstance(hd, tuple) else hd

        # Compute one-sided distances
        hd_a_to_b = pcu.one_sided_hausdorff_distance(points_a, points_b)
        hd_a_to_b = hd_a_to_b[0] if isinstance(hd_a_to_b, tuple) else hd_a_to_b
        hd_b_to_a = pcu.one_sided_hausdorff_distance(points_b, points_a)
        hd_b_to_a = hd_b_to_a[0] if isinstance(hd_b_to_a, tuple) else hd_b_to_a

        details = f"""Hausdorff Distance Analysis:
Total (symmetric): {hd:.6f}
A → B (one-sided): {hd_a_to_b:.6f}
B → A (one-sided): {hd_b_to_a:.6f}

Sampled {sample_count:,} points from each trimesh.
Mesh A: {len(trimesh_a.vertices):,} vertices, {len(trimesh_a.faces):,} faces
Mesh B: {len(trimesh_b.vertices):,} vertices, {len(trimesh_b.faces):,} faces
"""

        print(f"[HausdorffDistance] Result: {hd:.6f}")

        return (float(hd), details)


class ChamferDistanceNode:
    """
    Compute Chamfer distance between two meshes or point clouds.

    Chamfer distance is the average of squared distances from each point
    to its nearest neighbor in the other set. More sensitive to overall
    shape similarity than Hausdorff distance.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh_a": ("TRIMESH",),
                "trimesh_b": ("TRIMESH",),
                "sample_count": ("INT", {
                    "default": 10000,
                    "min": 1000,
                    "max": 1000000,
                    "step": 1000
                }),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("chamfer_distance", "info")
    FUNCTION = "compute_distance"
    CATEGORY = "geompack/analysis/distance"

    def compute_distance(self, trimesh_a, trimesh_b, sample_count):
        """
        Compute Chamfer distance between two meshes.

        Args:
            trimesh_a: First trimesh.Trimesh object
            trimesh_b: Second trimesh.Trimesh object
            sample_count: Number of points to sample from each mesh

        Returns:
            tuple: (chamfer_distance, info_string)
        """
        try:
            import point_cloud_utils as pcu
        except ImportError:
            raise ImportError(
                "point-cloud-utils not installed. Install with: pip install point-cloud-utils"
            )

        print(f"[ChamferDistance] Comparing meshes with {sample_count} samples each")

        # Sample point clouds from meshes
        points_a = trimesh_a.sample(sample_count)
        points_b = trimesh_b.sample(sample_count)

        # Compute Chamfer distance
        cd = pcu.chamfer_distance(points_a, points_b)

        info = f"""Chamfer Distance: {cd:.6f}

Sampled {sample_count:,} points from each trimesh.
Mesh A: {len(trimesh_a.vertices):,} vertices, {len(trimesh_a.faces):,} faces
Mesh B: {len(trimesh_b.vertices):,} vertices, {len(trimesh_b.faces):,} faces

Note: Chamfer distance is more sensitive to overall shape
similarity compared to Hausdorff distance.
"""

        print(f"[ChamferDistance] Result: {cd:.6f}")

        return (float(cd), info)


class ComputeSDFNode:
    """
    Compute Signed Distance Field (SDF) for a trimesh using libigl.

    The SDF represents the distance from any point in 3D space to the
    nearest surface, with negative values inside the mesh and positive
    values outside. Useful for occupancy queries and implicit surface
    representations.

    Uses libigl's signed_distance with multiple computation methods:
    - default: Fast and robust for most meshes
    - winding_number: Accurate, handles non-watertight meshes
    - fast_winding_number: Faster version of winding number
    - pseudonormal: Legacy method using pseudo-normals
    - unsigned: Compute unsigned distance only
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "resolution": ("INT", {
                    "default": 64,
                    "min": 16,
                    "max": 256,
                    "step": 16
                }),
                "sign_type": (["default", "winding_number", "fast_winding_number", "pseudonormal", "unsigned"], {
                    "default": "default"
                }),
            },
        }

    RETURN_TYPES = ("SDF_VOLUME", "STRING")
    RETURN_NAMES = ("sdf_volume", "info")
    FUNCTION = "compute_sdf"
    CATEGORY = "geompack/analysis/distance"

    def compute_sdf(self, trimesh, resolution, sign_type="default"):
        """
        Compute signed distance field voxel grid for trimesh using libigl.

        Args:
            trimesh: Input trimesh.Trimesh object
            resolution: Grid resolution (N x N x N voxels)
            sign_type: Distance computation method

        Returns:
            tuple: (sdf_data_dict, info_string)
        """
        try:
            import igl
        except ImportError:
            raise ImportError(
                "libigl not installed. Install with: pip install igl"
            )

        print(f"[ComputeSDF] Computing {resolution}³ SDF for mesh with {len(trimesh.vertices):,} vertices")
        print(f"[ComputeSDF] Method: {sign_type}")

        # Map sign_type string to igl constant
        sign_type_map = {
            "default": igl.SIGNED_DISTANCE_TYPE_DEFAULT,
            "winding_number": igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER,
            "fast_winding_number": igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
            "pseudonormal": igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
            "unsigned": igl.SIGNED_DISTANCE_TYPE_UNSIGNED,
        }
        igl_sign_type = sign_type_map[sign_type]

        # Generate 3D grid of query points
        bounds = trimesh.bounds
        grid_x = np.linspace(bounds[0, 0], bounds[1, 0], resolution)
        grid_y = np.linspace(bounds[0, 1], bounds[1, 1], resolution)
        grid_z = np.linspace(bounds[0, 2], bounds[1, 2], resolution)
        xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
        query_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        print(f"[ComputeSDF] Query points: {query_points.shape[0]:,}")

        # Compute signed distance using libigl
        # Returns: (distances, face_indices, closest_points, normals)
        S, I, C, N = igl.signed_distance(
            query_points,
            np.asarray(trimesh.vertices, dtype=np.float64),
            np.asarray(trimesh.faces, dtype=np.int64),
            sign_type=igl_sign_type
        )

        # Reshape distance array to 3D voxel grid
        voxels = S.reshape(resolution, resolution, resolution)

        info = f"""Signed Distance Field (libigl):
Method: {sign_type}
Resolution: {resolution}³ = {resolution**3:,} voxels
Value range: [{voxels.min():.3f}, {voxels.max():.3f}]

Mesh bounds: {trimesh.bounds.tolist()}
Mesh extents: {trimesh.extents.tolist()}

Negative values = inside mesh
Positive values = outside mesh
Zero = on surface
"""

        # Package SDF data
        sdf_data = {
            'voxels': voxels,
            'resolution': resolution,
            'bounds': trimesh.bounds.copy(),
            'extents': trimesh.extents.copy(),
            'sign_type': sign_type,
        }

        print(f"[ComputeSDF] Complete - range: [{voxels.min():.3f}, {voxels.max():.3f}]")

        return (sdf_data, info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackHausdorffDistance": HausdorffDistanceNode,
    "GeomPackChamferDistance": ChamferDistanceNode,
    "GeomPackComputeSDF": ComputeSDFNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackHausdorffDistance": "Hausdorff Distance",
    "GeomPackChamferDistance": "Chamfer Distance",
    "GeomPackComputeSDF": "Compute SDF",
}
