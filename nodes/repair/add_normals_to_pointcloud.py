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
            description=(
                'Estimate and add normals to a point cloud.\n\n'
                'pymeshlab_mls: Moving Least Squares -- fits a smooth local surface at each '
                'point and reads its normal off that fit. More robust to noisy/irregular '
                'sampling, but iterative and comparatively expensive.\n\n'
                'geogram_co3ne: k-nearest-neighbor local plane fit (PCA over each point\'s '
                'neighborhood) -- the standard fast, cheap baseline normal estimator. Good '
                'default for large or noisy point clouds where MLS\'s iterative cost adds up; '
                'less robust than MLS on very sparse/irregular sampling.'
            ),
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("pointcloud", tooltip="Input point cloud (will reject meshes with faces)"),
                io.DynamicCombo.Input("method", tooltip="Normal estimation method.", options=[
                    io.DynamicCombo.Option("pymeshlab_mls", [
                        io.Int.Input("mls_smoothing", default=5, min=1, max=20, tooltip=(
                            "MLS smoothing iterations (also used as the neighborhood size k for the "
                            "underlying moving-least-squares fit). Higher = smoother, more noise-"
                            "resistant normals but a more washed-out result on genuinely sharp "
                            "features; lower = more locally accurate but more sensitive to noise. "
                            "The default, 5, is a reasonable middle ground for typical scans -- raise "
                            "toward 10-15 for visibly noisy point clouds, lower toward 2-3 if the "
                            "cloud is already clean and you want normals that track fine detail "
                            "closely.")),
                    ]),
                    io.DynamicCombo.Option("geogram_co3ne", [
                        io.Int.Input("nb_neighbors", default=30, min=3, max=500, step=1, tooltip=(
                            "Number of nearest neighbors used for each point's local plane fit (PCA "
                            "over the neighborhood's covariance -- the standard, simple normal-"
                            "estimation approach, distinct from MLS's smooth-surface fit). Too FEW "
                            "neighbors gives a noisy, unstable normal estimate on irregular sampling; "
                            "too MANY over-smooths the estimate across a wider area, blurring sharp "
                            "features and costing more time per point. The default, 30, suits "
                            "moderately dense point clouds; drop to 15-20 for dense, clean scans "
                            "where tight neighborhoods are already reliable, or raise to 50-100 for "
                            "sparse/noisy input where you need to average over more neighbors to get "
                            "a stable estimate.")),
                    ]),
                ]),
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
        orient_normals=True,
        add_as_attributes=True
    ):
        """
        Estimate and add normals to a point cloud.

        Args:
            pointcloud: Input point cloud (trimesh.PointCloud)
            method: DynamicCombo dict -- {"method": "pymeshlab_mls", "mls_smoothing": ...} or
                    {"method": "geogram_co3ne", "nb_neighbors": ...}
            orient_normals: Whether to orient normals consistently
            add_as_attributes: Store normals as vertex_attributes

        Returns:
            Tuple of (point cloud with normals, info string)
        """
        method_sel = method["method"]
        mls_smoothing = method.get("mls_smoothing", 5)
        nb_neighbors = method.get("nb_neighbors", 30)

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

        log.info("Processing %d points with method: %s", num_points, method_sel)

        # Estimate normals based on method
        try:
            if method_sel == "pymeshlab_mls":
                normals = cls._estimate_normals_pymeshlab_mls(vertices, mls_smoothing, orient_normals)
            elif method_sel == "geogram_co3ne":
                normals = cls._estimate_normals_geogram_co3ne(vertices, nb_neighbors, orient_normals)
            else:
                raise ValueError(f"Unknown method: {method_sel}")
        except ImportError as e:
            raise ImportError(
                f"Method '{method_sel}' requires additional dependencies. "
                f"Please install the required package: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Normal estimation failed with method '{method_sel}': {e}")

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
        result.metadata['normal_estimation_method'] = method_sel
        result.metadata['is_point_cloud'] = True

        # Optionally add as vertex attributes for VTK visualization
        if add_as_attributes:
            result.vertex_attributes['normal_x'] = normals[:, 0]
            result.vertex_attributes['normal_y'] = normals[:, 1]
            result.vertex_attributes['normal_z'] = normals[:, 2]
            result.vertex_attributes['normal_magnitude'] = np.linalg.norm(normals, axis=1)

        # Create info string
        info = f"Added normals to {num_points} points using {method_sel}"
        if add_as_attributes:
            info += " (stored as vertex_attributes for visualization)"

        log.info("%s", info)

        return io.NodeOutput(result, info, ui={"text": [info]})

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

    @staticmethod
    def _estimate_normals_geogram_co3ne(points, nb_neighbors, orient_normals):
        """
        Estimate normals using Geogram's CO3NE k-NN local plane fit (PCA).

        Args:
            points: Nx3 numpy array of point coordinates
            nb_neighbors: Number of nearest neighbors for the local plane fit
            orient_normals: Whether to orient normals consistently

        Returns:
            Nx3 numpy array of normals
        """
        import pygeogram

        vertices = np.ascontiguousarray(points, dtype=np.float64)
        normals = pygeogram.co3ne_compute_normals(
            vertices, nb_neighbors=int(nb_neighbors), reorient=bool(orient_normals),
        )
        return np.asarray(normals, dtype=np.float32)

# Node registration
NODE_CLASS_MAPPINGS = {
    "GeomPackAddNormalsToPointCloud": AddNormalsToPointCloud,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackAddNormalsToPointCloud": "Add Normals to PointCloud",
}
