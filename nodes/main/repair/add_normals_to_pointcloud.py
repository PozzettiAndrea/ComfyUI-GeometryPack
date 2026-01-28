"""Add normals to point clouds using various estimation methods."""

import numpy as np
import trimesh


class AddNormalsToPointCloud:
    """Estimate and add normals to a point cloud using various methods."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pointcloud": ("TRIMESH", {
                    "tooltip": "Input point cloud (will reject meshes with faces)"
                }),
                "method": (["open3d_knn", "open3d_radius", "pymeshlab_mls"], {
                    "default": "open3d_knn",
                    "tooltip": "Normal estimation method"
                }),
            },
            "optional": {
                # For open3d_knn method
                "k_neighbors": ("INT", {
                    "default": 30,
                    "min": 3,
                    "max": 100,
                    "step": 1,
                    "tooltip": "[open3d_knn] Number of nearest neighbors for PCA"
                }),

                # For open3d_radius method
                "search_radius": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "[open3d_radius] Search radius for neighborhood (in normalized space)"
                }),

                # For pymeshlab_mls
                "mls_smoothing": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "tooltip": "[pymeshlab_mls] MLS smoothing iterations"
                }),

                # Common parameters
                "orient_normals": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Orient normals consistently across surface"
                }),

                "add_as_attributes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also store normals as vertex_attributes (normal_x/y/z) for VTK visualization"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("pointcloud_with_normals", "info")
    FUNCTION = "add_normals"
    CATEGORY = "geompack/repair"
    DESCRIPTION = "Estimate and add normals to a point cloud using Open3D or PyMeshLab methods."

    def add_normals(
        self,
        pointcloud,
        method,
        k_neighbors=30,
        search_radius=0.05,
        mls_smoothing=5,
        orient_normals=True,
        add_as_attributes=True
    ):
        """
        Estimate and add normals to a point cloud.

        Args:
            pointcloud: Input point cloud (trimesh.PointCloud)
            method: Normal estimation method
            k_neighbors: Number of neighbors for k-NN methods
            search_radius: Radius for radius-based search
            mls_smoothing: MLS smoothing parameter
            orient_normals: Whether to orient normals consistently
            add_as_attributes: Store normals as vertex_attributes

        Returns:
            Tuple of (point cloud with normals, info string)
        """
        # Check that input is actually a point cloud
        if hasattr(pointcloud, 'faces') and len(pointcloud.faces) > 0:
            raise ValueError(
                "Input must be a point cloud (0 faces). "
                "Use MeshToPointCloud node to convert a mesh to point cloud."
            )

        # Get vertices
        vertices = np.asarray(pointcloud.vertices).astype(np.float32)
        num_points = len(vertices)

        if num_points == 0:
            raise ValueError("Point cloud has no vertices")

        print(f"[AddNormalsToPointCloud] Processing {num_points} points with method: {method}")

        # Estimate normals based on method
        try:
            if method == "open3d_knn":
                normals = self._estimate_normals_open3d_knn(vertices, k_neighbors, orient_normals)
            elif method == "open3d_radius":
                normals = self._estimate_normals_open3d_radius(vertices, search_radius, orient_normals)
            elif method == "pymeshlab_mls":
                normals = self._estimate_normals_pymeshlab_mls(vertices, mls_smoothing, orient_normals)
            else:
                raise ValueError(f"Unknown method: {method}")
        except ImportError as e:
            raise ImportError(
                f"Method '{method}' requires additional dependencies. "
                f"Please install the required package: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Normal estimation failed with method '{method}': {e}")

        # Validate normals
        if normals.shape != vertices.shape:
            raise RuntimeError(
                f"Normal estimation produced wrong shape: {normals.shape} vs {vertices.shape}"
            )

        # Create result point cloud
        # Use Trimesh with 0 faces instead of PointCloud to support vertex_attributes
        if add_as_attributes:
            # Create as Trimesh with no faces to get vertex_attributes support
            result = trimesh.Trimesh(vertices=vertices, faces=[])
        else:
            # Use PointCloud if we don't need vertex_attributes
            result = trimesh.PointCloud(vertices=vertices)

        # Store normals as trimesh property
        result.vertex_normals = normals

        # Preserve metadata
        if hasattr(pointcloud, 'metadata'):
            result.metadata = pointcloud.metadata.copy()
        else:
            result.metadata = {}

        result.metadata['has_normals'] = True
        result.metadata['normal_estimation_method'] = method
        result.metadata['is_point_cloud'] = True

        # Optionally add as vertex attributes for VTK visualization
        if add_as_attributes:
            result.vertex_attributes['normal_x'] = normals[:, 0]
            result.vertex_attributes['normal_y'] = normals[:, 1]
            result.vertex_attributes['normal_z'] = normals[:, 2]
            result.vertex_attributes['normal_magnitude'] = np.linalg.norm(normals, axis=1)

        # Create info string
        info = f"Added normals to {num_points} points using {method}"
        if add_as_attributes:
            info += " (stored as vertex_attributes for visualization)"

        print(f"[AddNormalsToPointCloud] ✓ {info}")

        return (result, info)

    def _estimate_normals_open3d_knn(self, points, k_neighbors, orient_normals):
        """
        Estimate normals using Open3D k-nearest neighbors PCA.

        Args:
            points: Nx3 numpy array of point coordinates
            k_neighbors: Number of nearest neighbors
            orient_normals: Whether to orient normals consistently

        Returns:
            Nx3 numpy array of normals
        """
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
        )

        if orient_normals:
            pcd.orient_normals_consistent_tangent_plane(k=k_neighbors)

        normals = np.asarray(pcd.normals).astype(np.float32)
        return normals

    def _estimate_normals_open3d_radius(self, points, search_radius, orient_normals):
        """
        Estimate normals using Open3D radius-based search PCA.

        Args:
            points: Nx3 numpy array of point coordinates
            search_radius: Search radius for neighbors
            orient_normals: Whether to orient normals consistently

        Returns:
            Nx3 numpy array of normals
        """
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamRadius(radius=search_radius)
        )

        if orient_normals:
            # For radius search, use adaptive k based on average neighbors found
            pcd.orient_normals_consistent_tangent_plane(k=15)

        normals = np.asarray(pcd.normals).astype(np.float32)
        return normals

    def _estimate_normals_pymeshlab_mls(self, points, mls_smoothing, orient_normals):
        """
        Estimate normals using PyMeshLab Moving Least Squares.

        Args:
            points: Nx3 numpy array of point coordinates
            mls_smoothing: MLS smoothing parameter
            orient_normals: Whether to orient normals consistently

        Returns:
            Nx3 numpy array of normals
        """
        import pymeshlab as ml

        # Create MeshSet and add point cloud
        ms = ml.MeshSet()

        # PyMeshLab requires a mesh, so create one with no faces
        mesh = ml.Mesh(vertex_matrix=points)
        ms.add_mesh(mesh)

        # Compute normals using MLS
        ms.compute_normal_for_point_clouds(
            k=mls_smoothing,
            smoothiter=mls_smoothing,
            flipflag=orient_normals,
            viewpos=np.array([0.0, 0.0, 0.0])  # Origin for orientation
        )

        # Extract normals
        current_mesh = ms.current_mesh()
        normals = current_mesh.vertex_normal_matrix().astype(np.float32)

        return normals


# Node registration
NODE_CLASS_MAPPINGS = {
    "GeomPackAddNormalsToPointCloud": AddNormalsToPointCloud,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackAddNormalsToPointCloud": "Add Normals to PointCloud",
}
