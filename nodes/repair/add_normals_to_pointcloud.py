"""Add normals to point clouds using various estimation methods."""

import logging

import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")

class AddNormalsToPointCloud(io.ComfyNode):
    """Estimate and add normals to a point cloud using various methods."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackAddNormalsToPointCloud",
            display_name="Add Normals to PointCloud",
            category="geompack/repair",
            description='Estimate and add normals to a point cloud using Open3D or PyMeshLab methods.',
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("pointcloud", tooltip="Input point cloud (will reject meshes with faces)"),
                io.Combo.Input("method", options=["open3d_knn", "open3d_radius", "pymeshlab_mls"], default="open3d_knn", tooltip="Normal estimation method"),
                io.Int.Input("k_neighbors", default=30, min=3, max=100, step=1, tooltip="[open3d_knn] Number of nearest neighbors for PCA", optional=True),
                io.Float.Input("search_radius", default=0.05, min=0.001, max=1.0, step=0.001, tooltip="[open3d_radius] Search radius for neighborhood (in normalized space)", optional=True),
                io.Int.Input("mls_smoothing", default=5, min=1, max=20, tooltip="[pymeshlab_mls] MLS smoothing iterations", optional=True),
                io.Boolean.Input("orient_normals", default=True, tooltip="Orient normals consistently across surface", optional=True),
                io.Boolean.Input("add_as_attributes", default=True, tooltip="Also store normals as vertex_attributes (normal_x/y/z) for VTK visualization", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="pointcloud_with_normals"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(
        cls,
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

        log.info("Processing %d points with method: %s", num_points, method)

        # Estimate normals based on method
        try:
            if method == "open3d_knn":
                normals = cls._estimate_normals_open3d_knn(vertices, k_neighbors, orient_normals)
            elif method == "open3d_radius":
                normals = cls._estimate_normals_open3d_radius(vertices, search_radius, orient_normals)
            elif method == "pymeshlab_mls":
                normals = cls._estimate_normals_pymeshlab_mls(vertices, mls_smoothing, orient_normals)
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

        # Create result as Trimesh with no faces (compatible with IPC serialization)
        result = trimesh.Trimesh(vertices=vertices)

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

        log.info("%s", info)

        return io.NodeOutput(result, info, ui={"text": [info]})

    @staticmethod
    def _estimate_normals_open3d_knn(points, k_neighbors, orient_normals):
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

    @staticmethod
    def _estimate_normals_open3d_radius(points, search_radius, orient_normals):
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

    @staticmethod
    def _estimate_normals_pymeshlab_mls(points, mls_smoothing, orient_normals):
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
