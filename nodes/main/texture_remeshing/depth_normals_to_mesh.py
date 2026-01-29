# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Depth + Normals to Mesh Node - Convert depth map and normal map to smooth 3D mesh.
Designed for CAD raytracing output using Poisson surface reconstruction.
"""

import numpy as np
import trimesh


class DepthNormalsToMeshNode:
    """
    Depth + Normals to Mesh - Convert depth map and normal map to smooth 3D mesh.

    Takes a depth map (MASK) and normal map (IMAGE with Nx in R, Ny in G) and
    reconstructs a smooth surface using Poisson or Ball Pivoting algorithms.

    This approach eliminates the "Minecraft-like" stair-step artifacts of grid-based
    heightmap displacement by using proper surface reconstruction from oriented
    point clouds.

    Designed for CAD raytracing workflows where depth and normals are available.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_map": ("IMAGE",),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Resolution for point cloud sampling (higher = more detail, slower)"
                }),
                "depth_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Scale factor for depth values"
                }),
            },
            "optional": {
                "depth": ("MASK",),
                "depth_image": ("IMAGE",),
                "method": ([
                    "poisson",
                    "ball_pivoting",
                ], {"default": "poisson"}),
                "poisson_depth": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 12,
                    "step": 1,
                    "tooltip": "Octree depth for Poisson reconstruction (higher = more detail)"
                }),
                "poisson_scale": ("FLOAT", {
                    "default": 1.1,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Scale factor for Poisson bounding box"
                }),
                "skip_background": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Skip pixels with depth below threshold (background removal)"
                }),
                "background_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Depth threshold for background pixels"
                }),
                "invert_depth": (["false", "true"], {
                    "default": "false",
                    "tooltip": "Invert depth values (for depth maps where white = far)"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "depth_normals_to_mesh"
    CATEGORY = "geompack/texture_remeshing"

    def depth_normals_to_mesh(self, normal_map, resolution, depth_scale,
                               depth=None, depth_image=None,
                               method="poisson", poisson_depth=8, poisson_scale=1.1,
                               skip_background="true", background_threshold=0.01,
                               invert_depth="false"):
        """
        Convert depth map and normal map to smooth 3D mesh.

        Args:
            normal_map: Normal map as IMAGE tensor (B, H, W, C) with Nx in R, Ny in G
            resolution: Target resolution for point cloud
            depth_scale: Scale factor for depth values
            depth: Depth map as MASK tensor (B, H, W) - optional
            depth_image: Depth map as IMAGE tensor (B, H, W, C) - optional, takes priority
            method: Reconstruction method ("poisson" or "ball_pivoting")
            poisson_depth: Octree depth for Poisson reconstruction
            poisson_scale: Scale factor for Poisson bounding box
            skip_background: Whether to skip background pixels
            background_threshold: Threshold for background detection
            invert_depth: Whether to invert depth values

        Returns:
            tuple: (mesh, info_string)
        """
        try:
            import torch
            from PIL import Image
        except ImportError:
            raise RuntimeError("torch and PIL required. Install with: pip install torch Pillow")

        # Validate that at least one depth input is provided
        if depth is None and depth_image is None:
            raise ValueError("Either 'depth' (MASK) or 'depth_image' (IMAGE) must be provided")

        print(f"[DepthNormalsToMesh] Converting depth + normals to mesh")
        print(f"[DepthNormalsToMesh] Method: {method}, Resolution: {resolution}")

        # Extract depth map - prefer depth_image if provided
        if depth_image is not None:
            print(f"[DepthNormalsToMesh] Using depth_image input (averaging RGB channels)")
            if isinstance(depth_image, torch.Tensor):
                depth_img_arr = depth_image[0].cpu().numpy()
            else:
                depth_img_arr = np.array(depth_image)

            # Average RGB channels to create grayscale depth
            if len(depth_img_arr.shape) == 3 and depth_img_arr.shape[2] >= 3:
                depth_arr = np.mean(depth_img_arr[:, :, :3], axis=2)
            elif len(depth_img_arr.shape) == 3 and depth_img_arr.shape[2] == 1:
                depth_arr = depth_img_arr[:, :, 0]
            else:
                depth_arr = depth_img_arr
        else:
            print(f"[DepthNormalsToMesh] Using depth mask input")
            if isinstance(depth, torch.Tensor):
                depth_arr = depth[0].cpu().numpy()
            else:
                depth_arr = np.array(depth)

            # Ensure 2D
            if len(depth_arr.shape) > 2:
                depth_arr = depth_arr[:, :, 0] if depth_arr.shape[2] == 1 else np.mean(depth_arr, axis=2)

        # Normalize to [0, 1]
        depth_min, depth_max = depth_arr.min(), depth_arr.max()
        if depth_max > depth_min:
            depth_arr = (depth_arr - depth_min) / (depth_max - depth_min)

        print(f"[DepthNormalsToMesh] Depth size: {depth_arr.shape}, range: [{depth_min:.3f}, {depth_max:.3f}]")

        # Extract normal map from tensor (B, H, W, C)
        if isinstance(normal_map, torch.Tensor):
            normal_arr = normal_map[0].cpu().numpy()
        else:
            normal_arr = np.array(normal_map)

        # Ensure we have at least 2 channels (R, G for Nx, Ny)
        if len(normal_arr.shape) == 2:
            raise ValueError("Normal map must be RGB image with Nx in R, Ny in G channels")

        print(f"[DepthNormalsToMesh] Normal map size: {normal_arr.shape}")

        # Resize both to target resolution
        depth_pil = Image.fromarray((depth_arr * 255).astype(np.uint8))
        depth_pil = depth_pil.resize((resolution, resolution), Image.Resampling.LANCZOS)
        depth_resized = np.array(depth_pil).astype(np.float32) / 255.0

        # Resize normal map (use BILINEAR to preserve direction)
        normal_pil = Image.fromarray((normal_arr * 255).astype(np.uint8))
        normal_pil = normal_pil.resize((resolution, resolution), Image.Resampling.BILINEAR)
        normal_resized = np.array(normal_pil).astype(np.float32) / 255.0

        # Invert depth if requested
        if invert_depth == "true":
            depth_resized = 1.0 - depth_resized

        # Build oriented point cloud
        height, width = resolution, resolution
        points = []
        normals = []
        threshold = background_threshold if skip_background == "true" else -1.0

        for y in range(height):
            for x in range(width):
                d = depth_resized[y, x]

                # Skip background pixels
                if d <= threshold:
                    continue

                # Position from depth (normalize x,y to [-1, 1])
                px = (x / (width - 1)) * 2.0 - 1.0
                py = (y / (height - 1)) * 2.0 - 1.0
                pz = d * depth_scale
                points.append([px, py, pz])

                # Normal from RGB (R=Nx, G=Ny, derive Nz)
                nx = normal_resized[y, x, 0] * 2.0 - 1.0  # [0,1] -> [-1,1]
                ny = normal_resized[y, x, 1] * 2.0 - 1.0

                # Derive Nz from unit normal constraint: Nx² + Ny² + Nz² = 1
                nz_sq = max(0.0, 1.0 - nx*nx - ny*ny)
                nz = np.sqrt(nz_sq)

                # Normalize to ensure unit length
                length = np.sqrt(nx*nx + ny*ny + nz*nz)
                if length > 0:
                    nx, ny, nz = nx/length, ny/length, nz/length

                normals.append([nx, ny, nz])

        points = np.array(points, dtype=np.float64)
        normals = np.array(normals, dtype=np.float64)

        print(f"[DepthNormalsToMesh] Point cloud: {len(points)} points")

        if len(points) < 10:
            raise ValueError(f"Too few valid points ({len(points)}). Check depth map and threshold settings.")

        # Create point cloud as trimesh with normals
        point_cloud = trimesh.Trimesh(vertices=points, faces=[], process=False)
        point_cloud.vertex_normals = normals

        # Reconstruct surface
        if method == "poisson":
            mesh, method_info = self._poisson_reconstruct(points, normals, poisson_depth, poisson_scale)
        elif method == "ball_pivoting":
            mesh, method_info = self._ball_pivoting_reconstruct(points, normals)
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"[DepthNormalsToMesh] Output: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Generate info string
        info = f"""Depth + Normals to Mesh Results:

Input:
  Depth Map: {depth_arr.shape[0]}x{depth_arr.shape[1]}
  Normal Map: {normal_arr.shape[0]}x{normal_arr.shape[1]}
  Resolution: {resolution}x{resolution}
  Depth Scale: {depth_scale}
  Method: {method}

Point Cloud:
  Valid Points: {len(points):,}
  Background Skipped: {skip_background} (threshold: {background_threshold})

Output Mesh:
  Vertices: {len(mesh.vertices):,}
  Faces: {len(mesh.faces):,}
  Watertight: {mesh.is_watertight}
  Bounds: {mesh.bounds.tolist()}

{method_info}
"""
        return (mesh, info)

    def _poisson_reconstruct(self, points, normals, depth, scale):
        """Poisson surface reconstruction using Open3D or PyMeshLab."""
        # Try Open3D first
        try:
            import open3d as o3d

            print(f"[DepthNormalsToMesh] Using Open3D Poisson reconstruction...")

            # Create point cloud with normals
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)

            # Poisson reconstruction
            mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, scale=scale, linear_fit=False
            )

            # Remove low density vertices (noise at boundaries)
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.01)
            vertices_to_remove = densities < density_threshold
            mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

            # Convert to trimesh
            result = trimesh.Trimesh(
                vertices=np.asarray(mesh_o3d.vertices),
                faces=np.asarray(mesh_o3d.triangles),
                process=False
            )

            method_info = f"""Poisson Reconstruction (Open3D):
  Octree Depth: {depth}
  Scale: {scale}
  Density Filtering: 1% quantile removed"""

            return result, method_info

        except ImportError:
            pass

        # Fallback to PyMeshLab
        try:
            import pymeshlab

            print(f"[DepthNormalsToMesh] Using PyMeshLab Poisson reconstruction...")

            ms = pymeshlab.MeshSet()
            pml_mesh = pymeshlab.Mesh(
                vertex_matrix=points,
                v_normals_matrix=normals
            )
            ms.add_mesh(pml_mesh)

            # Poisson reconstruction
            ms.generate_surface_reconstruction_screened_poisson(
                depth=depth,
                scale=scale
            )

            result_mesh = ms.current_mesh()
            result = trimesh.Trimesh(
                vertices=result_mesh.vertex_matrix(),
                faces=result_mesh.face_matrix(),
                process=False
            )

            method_info = f"""Poisson Reconstruction (PyMeshLab):
  Octree Depth: {depth}
  Scale: {scale}"""

            return result, method_info

        except ImportError:
            raise ImportError(
                "Poisson reconstruction requires Open3D or PyMeshLab.\n"
                "Install with: pip install open3d  or  pip install pymeshlab"
            )

    def _ball_pivoting_reconstruct(self, points, normals):
        """Ball pivoting algorithm using PyMeshLab."""
        try:
            import pymeshlab

            print(f"[DepthNormalsToMesh] Using PyMeshLab Ball Pivoting...")

            ms = pymeshlab.MeshSet()
            pml_mesh = pymeshlab.Mesh(
                vertex_matrix=points,
                v_normals_matrix=normals
            )
            ms.add_mesh(pml_mesh)

            # Ball pivoting with auto radius
            ms.generate_surface_reconstruction_ball_pivoting()

            result_mesh = ms.current_mesh()
            result = trimesh.Trimesh(
                vertices=result_mesh.vertex_matrix(),
                faces=result_mesh.face_matrix(),
                process=False
            )

            method_info = """Ball Pivoting Reconstruction (PyMeshLab):
  Radius: auto
  Note: May have holes in regions with sparse points"""

            return result, method_info

        except ImportError:
            raise ImportError(
                "Ball pivoting requires PyMeshLab.\n"
                "Install with: pip install pymeshlab"
            )


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackDepthNormalsToMesh": DepthNormalsToMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackDepthNormalsToMesh": "Depth + Normals to Mesh",
}
