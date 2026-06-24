# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Texture to Geometry Node - Convert texture heightmap to 3D geometry
Supports multiple backends: grid, Poisson (PyMeshLab/Open3D), Delaunay
"""

import logging

import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")

def _to_numpy(x):
    """Convert tensor or array to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    return np.array(x)

class TextureToGeometryNode(io.ComfyNode):
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
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackTextureToGeometry",
            display_name="Depth Map to Mesh",
            category="geompack/texture_remeshing",
            is_output_node=True,
            inputs=[
                io.Float.Input("height_scale", default=1.0, min=0.01, max=10.0, step=0.1, display_mode="number"),
                io.MultiType.Input("depth", [io.Image, io.Mask],
                    tooltip="The depth / height map. Each pixel becomes a vertex, displaced in Z by its value x height_scale. Accepts an IMAGE (RGB is averaged to grayscale) OR a MASK (single channel)."),
                io.MultiType.Input("field", [io.Image, io.Mask], optional=True,
                    tooltip="Optional per-pixel value sampled at each vertex and stored as a named vertex attribute (field_name). Does NOT affect geometry -- e.g. a face-ID / segmentation / curvature map carried onto the mesh. IMAGE or MASK."),
                io.String.Input("field_name", default="field", tooltip="Name for the vertex attribute", optional=True),
                io.Mask.Input("mask", optional=True,
                    tooltip="Optional MASK selecting which pixels to use: only pixels where the "
                            "mask is > 0.5 become mesh geometry; masked-out (0) pixels are "
                            "skipped entirely (holes). Independent of the depth values (unlike "
                            "skip_black). If its resolution differs from the depth map it is "
                            "nearest-neighbour resized to match. Combines with skip_black if both set."),
                io.DynamicCombo.Input("backend", tooltip="Reconstruction backend: grid (fast), poisson (smooth), delaunay", options=[
                    io.DynamicCombo.Option("grid", [
                        io.Combo.Input("smooth_normals", options=["true", "false"], default="true"),
                    ]),
                    io.DynamicCombo.Option("poisson_pymeshlab", [
                        io.Int.Input("poisson_depth", default=8, min=4, max=12, step=1, tooltip="Octree depth for Poisson reconstruction (higher = more detail)"),
                    ]),
                    io.DynamicCombo.Option("poisson_open3d", [
                        io.Int.Input("poisson_depth", default=8, min=4, max=12, step=1, tooltip="Octree depth for Poisson reconstruction (higher = more detail)"),
                    ]),
                    io.DynamicCombo.Option("delaunay_2d", []),
                ]),
                io.Combo.Input("invert_height", options=["false", "true"], default="false", optional=True),
                io.Combo.Input("skip_black", options=["false", "true"], default="false", tooltip="Skip faces connected to near-black pixels in the depth map", optional=True),
                io.Float.Input("black_threshold", default=0.01, min=0.0, max=1.0, step=0.01, tooltip="Threshold below which pixels are considered black (only used when skip_black is true)", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, height_scale,
                           depth=None,
                           field=None, field_name="field", mask=None,
                           backend=None, poisson_depth=8,
                           invert_height="false",
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
        if depth is None:
            raise ValueError("'depth' input is required (an IMAGE or a MASK)")

        # Extract DynamicCombo values
        if backend is None:
            backend = {"backend": "grid"}
        selected_backend = backend["backend"]
        poisson_depth = backend.get("poisson_depth", 8)
        smooth_normals = backend.get("smooth_normals", "true")

        log.info("Converting to geometry using backend: %s", selected_backend)

        # The 'depth' input may be an IMAGE (B,H,W,C) or a MASK (B,H,W) -> grayscale heightmap.
        arr = _to_numpy(depth)
        if arr.ndim == 4:                       # IMAGE batch (B,H,W,C)
            img = arr[0]
            heightmap = np.mean(img[:, :, :3], axis=2) if img.shape[2] >= 3 else img[:, :, 0]
            log.info("depth: IMAGE input (RGB averaged to grayscale)")
        elif arr.ndim == 3:                     # could be MASK batch (B,H,W) or single image (H,W,C)
            if arr.shape[2] in (3, 4):          # (H,W,C) RGB(A) image
                heightmap = np.mean(arr[:, :, :3], axis=2)
            elif arr.shape[2] == 1:             # (H,W,1)
                heightmap = arr[:, :, 0]
            else:                                # (B,H,W) mask -> first
                heightmap = arr[0]
            log.info("depth: 3D input -> heightmap %s", heightmap.shape)
        elif arr.ndim == 2:                     # (H,W)
            heightmap = arr
            log.info("depth: 2D input")
        else:
            raise ValueError(f"Unexpected 'depth' shape {arr.shape}; expected IMAGE or MASK")

        # Use native resolution
        height, width = heightmap.shape
        log.info("Using native resolution: %dx%d, range: [%.3f, %.3f]", width, height, heightmap.min(), heightmap.max())

        # Ensure float in [0, 1] range
        heightmap = heightmap.astype(np.float32)
        if heightmap.max() > 1.0:
            heightmap = heightmap / 255.0

        # Invert if requested
        if invert_height == "true":
            heightmap = 1.0 - heightmap

        # Optional explicit pixel mask: True = keep this pixel, False = skip (hole)
        keep_mask = cls._pixel_keep_mask(mask, height, width)
        if keep_mask is not None:
            log.info("mask input: keeping %d/%d pixels (%.1f%%)",
                     int(keep_mask.sum()), keep_mask.size, 100.0 * keep_mask.mean())

        # Build point cloud from heightmap
        points, valid_mask = cls._heightmap_to_points(
            heightmap, height_scale,
            skip_black == "true", black_threshold, keep_mask
        )

        log.info("Generated %d points", len(points))

        # Dispatch to appropriate backend
        if selected_backend == "grid":
            mesh = cls._build_grid_mesh(
                heightmap, height_scale, width, height,
                skip_black == "true", black_threshold,
                smooth_normals == "true",
                field, field_name, keep_mask
            )
            backend_info = "Grid-based displacement mesh"
        elif selected_backend == "poisson_pymeshlab":
            mesh = cls._build_poisson_pymeshlab(points, poisson_depth)
            backend_info = f"PyMeshLab Screened Poisson reconstruction (depth={poisson_depth})"
        elif selected_backend == "poisson_open3d":
            mesh = cls._build_poisson_open3d(points, poisson_depth)
            backend_info = f"Open3D Poisson reconstruction (depth={poisson_depth})"
        elif selected_backend == "delaunay_2d":
            mesh = cls._build_delaunay_2d(points)
            backend_info = "2D Delaunay triangulation"
        else:
            raise ValueError(f"Unknown backend: {selected_backend}")

        log.info("Created mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))

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
  Mask: {'yes (%d%% kept)' % int(100*keep_mask.mean()) if keep_mask is not None else 'none'}

Backend: {selected_backend}
  {backend_info}

Output Mesh:
  Vertices: {len(mesh.vertices):,}
  Faces: {len(mesh.faces):,}
  Height Range: [{height_min:.3f}, {height_max:.3f}] (span: {height_range:.3f})
  Bounds: {mesh.bounds.tolist()}
  Watertight: {mesh.is_watertight}
"""

        return io.NodeOutput(mesh, info, ui={"text": [info]})

    @staticmethod
    def _pixel_keep_mask(mask, height, width):
        """Parse a ComfyUI MASK/IMAGE into a boolean (H,W) keep-mask (True = use pixel).
        Returns None if no mask. Nearest-neighbour resizes to (height, width) if needed."""
        if mask is None:
            return None
        arr = _to_numpy(mask).astype(np.float32)
        if arr.ndim == 4:                       # (B,H,W,C)
            arr = arr[0]
            arr = np.mean(arr[:, :, :3], axis=2) if arr.shape[2] >= 3 else arr[:, :, 0]
        elif arr.ndim == 3:                     # (B,H,W) mask  or  (H,W,C) image
            if arr.shape[2] in (3, 4):
                arr = np.mean(arr[:, :, :3], axis=2)
            elif arr.shape[2] == 1:
                arr = arr[:, :, 0]
            else:
                arr = arr[0]
        if arr.ndim != 2:
            log.warning("mask: unexpected shape %s, ignoring", np.shape(mask))
            return None
        if arr.max() > 1.0:
            arr = arr / 255.0
        keep = arr > 0.5
        if keep.shape != (height, width):       # nearest-neighbour resize to depth res
            ys = np.linspace(0, keep.shape[0] - 1, height).round().astype(int)
            xs = np.linspace(0, keep.shape[1] - 1, width).round().astype(int)
            keep = keep[np.ix_(ys, xs)]
            log.info("mask: resized to depth resolution %dx%d", width, height)
        return keep

    @staticmethod
    def _heightmap_to_points(heightmap, height_scale, skip_black, black_threshold, keep_mask=None):
        """Convert heightmap to 3D point cloud."""
        height, width = heightmap.shape
        points = []
        valid_mask = np.ones((height, width), dtype=bool)

        for y in range(height):
            for x in range(width):
                h = heightmap[y, x]

                if keep_mask is not None and not keep_mask[y, x]:
                    valid_mask[y, x] = False
                    continue

                if skip_black and h <= black_threshold:
                    valid_mask[y, x] = False
                    continue

                # Normalize x, y to [-1, 1]
                nx = (x / (width - 1)) * 2.0 - 1.0
                ny = (y / (height - 1)) * 2.0 - 1.0
                nz = h * height_scale

                points.append([nx, ny, nz])

        return np.array(points, dtype=np.float64), valid_mask

    @staticmethod
    def _build_grid_mesh(heightmap, height_scale, width, height,
                         skip_black, black_threshold, smooth_normals,
                         field=None, field_name="field", keep_mask=None):
        """Build mesh using grid-based displacement (original algorithm)."""
        # Generate vertices
        vertices = []
        for y in range(height):
            for x in range(width):
                nx = (x / (width - 1)) * 2.0 - 1.0
                ny = (y / (height - 1)) * 2.0 - 1.0
                h = heightmap[y, x] * height_scale
                vertices.append([nx, ny, h])

        vertices = np.array(vertices, dtype=np.float32)

        # Per-pixel keep test: a triangle is emitted only if all 3 corner pixels are
        # kept. Combines skip_black (depth-derived) and the explicit mask input.
        pix_ok = np.ones((height, width), dtype=bool)
        if skip_black:
            pix_ok &= heightmap > black_threshold
        if keep_mask is not None:
            pix_ok &= keep_mask
        drop_any = skip_black or (keep_mask is not None)

        # Generate faces (triangles)
        faces = []
        for y in range(height - 1):
            for x in range(width - 1):
                i = y * width + x
                a = pix_ok[y, x]; b = pix_ok[y, x + 1]
                c = pix_ok[y + 1, x]; d = pix_ok[y + 1, x + 1]
                if a and b and c:
                    faces.append([i, i + 1, i + width])
                if b and d and c:
                    faces.append([i + 1, i + width + 1, i + width])

        faces = np.array(faces, dtype=np.int32)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Add field as vertex attribute if provided (IMAGE or MASK -> (H,W) scalar)
        if field is not None:
            field_arr = _to_numpy(field)
            if field_arr.ndim == 4:                 # IMAGE (B,H,W,C)
                field_arr = field_arr[0]
            if field_arr.ndim == 3:
                if field_arr.shape[2] in (3, 4):    # (H,W,C) -> grayscale
                    field_arr = np.mean(field_arr[:, :, :3], axis=2)
                elif field_arr.shape[2] == 1:
                    field_arr = field_arr[:, :, 0]
                else:                                # (B,H,W) mask
                    field_arr = field_arr[0]
            if field_arr.ndim > 2:
                field_arr = field_arr.squeeze()

            # Check resolution matches
            if field_arr.shape == (height, width):
                # Sample field at each vertex position (same order as vertices)
                field_values = field_arr.flatten()
                mesh.vertex_attributes[field_name] = field_values.astype(np.float32)
                log.info("Added vertex attribute '%s' with %d values, range: [%.3f, %.3f]", field_name, len(field_values), field_values.min(), field_values.max())
            else:
                log.warning("Warning: field shape %s doesn't match heightmap (%d, %d), skipping", field_arr.shape, height, width)

        # drop orphan vertices left by masked-out / skipped pixels (reindexes
        # vertex_attributes like the field too)
        if drop_any:
            try:
                mesh.remove_unreferenced_vertices()
            except Exception as e:
                log.debug("vertex cleanup skipped: %s", e)

        if smooth_normals:
            mesh.fix_normals()

        return mesh

    @staticmethod
    def _build_poisson_open3d(points, depth):
        """Build mesh using Open3D Poisson reconstruction."""
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError(
                "Open3D is required for poisson_open3d backend.\n"
                "Install with: pip install open3d"
            )

        log.info("Using Open3D Poisson reconstruction...")

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

    @staticmethod
    def _build_poisson_pymeshlab(points, depth):
        """Build mesh using PyMeshLab Screened Poisson reconstruction."""
        try:
            import pymeshlab
        except ImportError:
            raise ImportError(
                "PyMeshLab is required for poisson_pymeshlab backend.\n"
                "Install with: pip install pymeshlab"
            )

        log.info("Using PyMeshLab Screened Poisson reconstruction...")

        # Create MeshSet and add point cloud
        ms = pymeshlab.MeshSet()
        pml_mesh = pymeshlab.Mesh(vertex_matrix=points)
        ms.add_mesh(pml_mesh)

        # Estimate normals for point cloud
        log.info("Estimating normals...")
        ms.compute_normal_for_point_clouds(k=10)

        # For depth maps, normals should point "up" (positive Z)
        # Check and flip if needed
        current_mesh = ms.current_mesh()
        normals = current_mesh.vertex_normal_matrix()
        if np.mean(normals[:, 2]) < 0:
            ms.meshing_invert_face_orientation()

        # Screened Poisson reconstruction
        log.info("Running Screened Poisson reconstruction (depth=%d)...", depth)
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

    @staticmethod
    def _build_delaunay_2d(points):
        """Build mesh using 2D Delaunay triangulation."""
        try:
            from scipy.spatial import Delaunay
        except ImportError:
            raise ImportError(
                "scipy is required for delaunay_2d backend.\n"
                "Install with: pip install scipy"
            )

        log.info("Using 2D Delaunay triangulation...")

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
