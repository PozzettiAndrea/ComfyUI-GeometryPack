# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Texture to Geometry Node - Convert texture heightmap to 3D geometry
Supports multiple backends: grid, Poisson (PyMeshLab/Open3D), Delaunay
"""

import numpy as np
import trimesh


class TextureToGeometryNode:
    """
    Texture to Geometry - Convert a heightmap texture to 3D mesh geometry.

    Takes an IMAGE (heightmap) and converts it to a 3D mesh.
    Multiple backends available:
    - grid: Fast grid-based displacement (may have stair-step artifacts)
    - poisson_pymeshlab: PyMeshLab Screened Poisson reconstruction (smooth)
    - poisson_open3d: Open3D Poisson reconstruction (smooth, requires Open3D)
    - delaunay_2d: 2D Delaunay triangulation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                }),
            },
            "optional": {
                "mask": ("MASK",),
                "depth_image": ("IMAGE",),
                "field": ("MASK", {
                    "tooltip": "Optional field to store as vertex attribute on the mesh (e.g., face IDs, curvature)"
                }),
                "field_name": ("STRING", {
                    "default": "field",
                    "tooltip": "Name for the vertex attribute"
                }),
                "backend": ([
                    "grid",
                    "poisson_pymeshlab",
                    "poisson_open3d",
                    "delaunay_2d",
                ], {
                    "default": "grid",
                    "tooltip": "Reconstruction backend: grid (fast), poisson (smooth), delaunay"
                }),
                "poisson_depth": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 12,
                    "step": 1,
                    "tooltip": "Octree depth for Poisson reconstruction (higher = more detail)",
                    "visible_when": {"backend": ["poisson_pymeshlab", "poisson_open3d"]},
                }),
                "invert_height": (["false", "true"], {"default": "false"}),
                "smooth_normals": (["true", "false"], {
                    "default": "true",
                    "visible_when": {"backend": ["grid"]},
                }),
                "skip_black": (["false", "true"], {"default": "false", "tooltip": "Skip faces connected to near-black pixels in the depth map"}),
                "black_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Threshold below which pixels are considered black (only used when skip_black is true)"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    OUTPUT_NODE = True
    FUNCTION = "texture_to_geometry"
    CATEGORY = "geompack/texture_remeshing"

    def texture_to_geometry(self, height_scale,
                           mask=None, depth_image=None,
                           field=None, field_name="field",
                           backend="grid", poisson_depth=8,
                           invert_height="false", smooth_normals="true",
                           skip_black="false", black_threshold=0.01):
        """
        Convert binary mask to 3D mesh with height displacement.

        Args:
            mask: Input MASK tensor (B, H, W) from ComfyUI
            height_scale: Scale factor for height displacement
            depth_image: Optional IMAGE tensor (B, H, W, C) - if provided, averages RGB to grayscale
            backend: Reconstruction backend (grid, poisson_pymeshlab, poisson_open3d, delaunay_2d)
            poisson_depth: Octree depth for Poisson reconstruction
            invert_height: Invert the mask (0=high, 1=low)
            smooth_normals: Compute smooth vertex normals
            skip_black: Skip faces connected to near-black pixels
            black_threshold: Threshold for black pixel detection

        Returns:
            tuple: (mesh, info_string)
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("torch required. Install with: pip install torch")

        # Validate that at least one input is provided
        if mask is None and depth_image is None:
            raise ValueError("Either 'mask' or 'depth_image' must be provided")

        print(f"[TextureToGeometry] Converting to geometry using backend: {backend}")

        # Use depth_image if provided, otherwise use mask
        if depth_image is not None:
            print(f"[TextureToGeometry] Using depth_image input (averaging RGB channels)")
            if isinstance(depth_image, torch.Tensor):
                img_arr = depth_image[0].cpu().numpy()
            else:
                img_arr = np.array(depth_image)

            # Average RGB channels to create grayscale
            if len(img_arr.shape) == 3 and img_arr.shape[2] >= 3:
                heightmap = np.mean(img_arr[:, :, :3], axis=2)
            elif len(img_arr.shape) == 3 and img_arr.shape[2] == 1:
                heightmap = img_arr[:, :, 0]
            else:
                heightmap = img_arr
        else:
            print(f"[TextureToGeometry] Using mask input")
            # Extract mask from ComfyUI tensor format (B, H, W)
            if isinstance(mask, torch.Tensor):
                heightmap = mask[0].cpu().numpy()
            else:
                heightmap = np.array(mask)

            # Ensure 2D array (masks are single-channel)
            if len(heightmap.shape) > 2:
                heightmap = heightmap[:, :, 0] if heightmap.shape[2] == 1 else np.mean(heightmap, axis=2)

        # Use native resolution
        height, width = heightmap.shape
        print(f"[TextureToGeometry] Using native resolution: {width}x{height}, range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")

        # Ensure float in [0, 1] range
        heightmap = heightmap.astype(np.float32)
        if heightmap.max() > 1.0:
            heightmap = heightmap / 255.0

        # Invert if requested
        if invert_height == "true":
            heightmap = 1.0 - heightmap

        # Build point cloud from heightmap
        points, valid_mask = self._heightmap_to_points(
            heightmap, height_scale,
            skip_black == "true", black_threshold
        )

        print(f"[TextureToGeometry] Generated {len(points)} points")

        # Dispatch to appropriate backend
        if backend == "grid":
            mesh = self._build_grid_mesh(
                heightmap, height_scale, width, height,
                skip_black == "true", black_threshold,
                smooth_normals == "true",
                field, field_name
            )
            backend_info = "Grid-based displacement mesh"
        elif backend == "poisson_pymeshlab":
            mesh = self._build_poisson_pymeshlab(points, poisson_depth)
            backend_info = f"PyMeshLab Screened Poisson reconstruction (depth={poisson_depth})"
        elif backend == "poisson_open3d":
            mesh = self._build_poisson_open3d(points, poisson_depth)
            backend_info = f"Open3D Poisson reconstruction (depth={poisson_depth})"
        elif backend == "delaunay_2d":
            mesh = self._build_delaunay_2d(points)
            backend_info = "2D Delaunay triangulation"
        else:
            raise ValueError(f"Unknown backend: {backend}")

        print(f"[TextureToGeometry] Created mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Compute statistics
        height_min = mesh.vertices[:, 2].min()
        height_max = mesh.vertices[:, 2].max()
        height_range = height_max - height_min

        info = f"""Depth Map to Mesh Results:

Input:
  Resolution: {width}x{height}
  Height Scale: {height_scale}
  Inverted: {invert_height}
  Skip Black: {skip_black} (threshold: {black_threshold})

Backend: {backend}
  {backend_info}

Output Mesh:
  Vertices: {len(mesh.vertices):,}
  Faces: {len(mesh.faces):,}
  Height Range: [{height_min:.3f}, {height_max:.3f}] (span: {height_range:.3f})
  Bounds: {mesh.bounds.tolist()}
  Watertight: {mesh.is_watertight}
"""

        return {
            "result": (mesh, info),
            "ui": {"text": [info]}
        }

    def _heightmap_to_points(self, heightmap, height_scale, skip_black, black_threshold):
        """Convert heightmap to 3D point cloud."""
        height, width = heightmap.shape
        points = []
        valid_mask = np.ones((height, width), dtype=bool)

        for y in range(height):
            for x in range(width):
                h = heightmap[y, x]

                if skip_black and h <= black_threshold:
                    valid_mask[y, x] = False
                    continue

                # Normalize x, y to [-1, 1]
                nx = (x / (width - 1)) * 2.0 - 1.0
                ny = (y / (height - 1)) * 2.0 - 1.0
                nz = h * height_scale

                points.append([nx, ny, nz])

        return np.array(points, dtype=np.float64), valid_mask

    def _build_grid_mesh(self, heightmap, height_scale, width, height,
                         skip_black, black_threshold, smooth_normals,
                         field=None, field_name="field"):
        """Build mesh using grid-based displacement (original algorithm)."""
        import torch

        # Generate vertices
        vertices = []
        for y in range(height):
            for x in range(width):
                nx = (x / (width - 1)) * 2.0 - 1.0
                ny = (y / (height - 1)) * 2.0 - 1.0
                h = heightmap[y, x] * height_scale
                vertices.append([nx, ny, h])

        vertices = np.array(vertices, dtype=np.float32)

        # Generate faces (triangles)
        faces = []
        for y in range(height - 1):
            for x in range(width - 1):
                i = y * width + x

                if skip_black:
                    h00 = heightmap[y, x]
                    h10 = heightmap[y, x + 1]
                    h01 = heightmap[y + 1, x]
                    h11 = heightmap[y + 1, x + 1]

                    if h00 > black_threshold and h10 > black_threshold and h01 > black_threshold:
                        faces.append([i, i + 1, i + width])
                    if h10 > black_threshold and h11 > black_threshold and h01 > black_threshold:
                        faces.append([i + 1, i + width + 1, i + width])
                else:
                    faces.append([i, i + 1, i + width])
                    faces.append([i + 1, i + width + 1, i + width])

        faces = np.array(faces, dtype=np.int32)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Add field as vertex attribute if provided
        if field is not None:
            if isinstance(field, torch.Tensor):
                field_arr = field[0].cpu().numpy()
            else:
                field_arr = np.array(field)

            # Handle different shapes
            if len(field_arr.shape) > 2:
                field_arr = field_arr.squeeze()

            # Check resolution matches
            if field_arr.shape == (height, width):
                # Sample field at each vertex position (same order as vertices)
                field_values = field_arr.flatten()
                mesh.vertex_attributes[field_name] = field_values.astype(np.float32)
                print(f"[TextureToGeometry] Added vertex attribute '{field_name}' with {len(field_values)} values, range: [{field_values.min():.3f}, {field_values.max():.3f}]")
            else:
                print(f"[TextureToGeometry] Warning: field shape {field_arr.shape} doesn't match heightmap ({height}, {width}), skipping")

        if smooth_normals:
            mesh.fix_normals()

        return mesh

    def _build_poisson_open3d(self, points, depth):
        """Build mesh using Open3D Poisson reconstruction."""
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError(
                "Open3D is required for poisson_open3d backend.\n"
                "Install with: pip install open3d"
            )

        print(f"[TextureToGeometry] Using Open3D Poisson reconstruction...")

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Estimate normals from point positions
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        # Orient normals consistently (pointing upward for depth maps)
        pcd.orient_normals_consistent_tangent_plane(k=10)

        # For depth maps, normals should generally point "up" (positive Z)
        # Check and flip if needed
        normals = np.asarray(pcd.normals)
        if np.mean(normals[:, 2]) < 0:
            pcd.normals = o3d.utility.Vector3dVector(-normals)

        # Poisson reconstruction
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, scale=1.1, linear_fit=False
        )

        # Remove low density vertices (noise at boundaries)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.01)
        vertices_to_remove = densities < density_threshold
        mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

        # Convert to trimesh
        mesh = trimesh.Trimesh(
            vertices=np.asarray(mesh_o3d.vertices),
            faces=np.asarray(mesh_o3d.triangles),
            process=False
        )

        return mesh

    def _build_poisson_pymeshlab(self, points, depth):
        """Build mesh using PyMeshLab Screened Poisson reconstruction."""
        try:
            import pymeshlab
        except ImportError:
            raise ImportError(
                "PyMeshLab is required for poisson_pymeshlab backend.\n"
                "Install with: pip install pymeshlab"
            )

        print(f"[TextureToGeometry] Using PyMeshLab Screened Poisson reconstruction...")

        # Create MeshSet and add point cloud
        ms = pymeshlab.MeshSet()
        pml_mesh = pymeshlab.Mesh(vertex_matrix=points)
        ms.add_mesh(pml_mesh)

        # Estimate normals for point cloud
        print(f"[TextureToGeometry] Estimating normals...")
        ms.compute_normal_for_point_clouds(k=10)

        # For depth maps, normals should point "up" (positive Z)
        # Check and flip if needed
        current_mesh = ms.current_mesh()
        normals = current_mesh.vertex_normal_matrix()
        if np.mean(normals[:, 2]) < 0:
            ms.meshing_invert_face_orientation()

        # Screened Poisson reconstruction
        print(f"[TextureToGeometry] Running Screened Poisson reconstruction (depth={depth})...")
        ms.generate_surface_reconstruction_screened_poisson(
            depth=depth,
            scale=1.1
        )

        # Get result mesh
        result_mesh = ms.current_mesh()
        vertices = result_mesh.vertex_matrix()
        faces = result_mesh.face_matrix()

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False
        )

        return mesh

    def _build_delaunay_2d(self, points):
        """Build mesh using 2D Delaunay triangulation."""
        try:
            from scipy.spatial import Delaunay
        except ImportError:
            raise ImportError(
                "scipy is required for delaunay_2d backend.\n"
                "Install with: pip install scipy"
            )

        print(f"[TextureToGeometry] Using 2D Delaunay triangulation...")

        # Project to XY plane for triangulation
        points_2d = points[:, :2]
        tri = Delaunay(points_2d)

        # Create mesh with original 3D coordinates
        mesh = trimesh.Trimesh(
            vertices=points,
            faces=tri.simplices,
            process=False
        )

        return mesh


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackTextureToGeometry": TextureToGeometryNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackTextureToGeometry": "Depth Map to Mesh",
}
