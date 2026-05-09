# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Backdraft detection visualization node.

Renders a mesh from above (Z-axis parallel projection) and colors pixels based on
ray intersection count:
- BLACK: Ray misses mesh (background)
- GREEN: Ray hits exactly 1 face (clean geometry, no undercut)
- RED: Ray hits 2+ faces (backdraft/undercut - material would be trapped)

This is useful for manufacturing analysis to detect undercuts that would
prevent clean mold release.

Supports three backends:
- trimesh: Uses embree for fast batch ray casting (recommended)
- pyvista: Uses VTK's multi_ray_trace
- face_normals: Checks face normal Z-component consistency (detects flipped faces)
"""

import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class BackdraftViewNode(io.ComfyNode):
    """
    Render mesh from Z-axis and detect backdraft regions.

    Backdraft = when a ray from above hits MORE than 1 face.
    This indicates an undercut that would trap material in manufacturing.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackBackdraftView",
            display_name="Backdraft View",
            category="geompack/visualization",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("resolution", default=1024, min=128, max=4096, step=64, tooltip="Output image resolution. Higher = more detail but slower."),
                io.Combo.Input("backend", options=["trimesh", "pyvista", "face_normals"], default="trimesh", tooltip="trimesh (embree) is faster, pyvista uses VTK, face_normals checks Z-normal consistency (requires single connected component)."),
                io.Boolean.Input("show_filename", default=True, tooltip="Display mesh filename on the output image", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="backdraft_image"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, resolution=1024, backend="trimesh", show_filename=True):
        """
        Render mesh with backdraft detection using batch ray casting.

        Args:
            trimesh: Input trimesh object
            resolution: Output image size
            backend: Ray tracing backend ("trimesh" or "pyvista")
            show_filename: Whether to display the filename on the image

        Returns:
            tuple: (IMAGE tensor in BHWC format)
        """
        log.info("Processing mesh: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Resolution: %dx%d, Backend: %s", resolution, resolution, backend)

        # Get mesh bounds
        bounds = trimesh.bounds
        xmin, ymin, zmin = bounds[0]
        xmax, ymax, zmax = bounds[1]
        log.info("Mesh bounds: X[%.2f, %.2f] Y[%.2f, %.2f] Z[%.2f, %.2f]", xmin, xmax, ymin, ymax, zmin, zmax)

        # Add padding around bounds
        x_range = xmax - xmin
        y_range = ymax - ymin
        padding = max(x_range, y_range) * 0.05
        xmin -= padding
        xmax += padding
        ymin -= padding
        ymax += padding

        # Calculate grid dimensions maintaining aspect ratio
        x_range = xmax - xmin
        y_range = ymax - ymin
        aspect = x_range / y_range

        if aspect > 1:
            nx = resolution
            ny = max(1, int(resolution / aspect))
        else:
            ny = resolution
            nx = max(1, int(resolution * aspect))

        total_rays = nx * ny
        log.info("Ray grid: %d x %d = %d rays", nx, ny, total_rays)

        # Create ray grid
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        z_start = zmax + 1.0
        z_end = zmin - 1.0

        # Dispatch to appropriate backend
        if backend == "trimesh":
            hit_counts = cls._raytrace_trimesh(trimesh, xs, ys, z_start, nx, ny, total_rays)
        elif backend == "pyvista":
            hit_counts = cls._raytrace_pyvista(trimesh, xs, ys, z_start, z_end, nx, ny)
        elif backend == "face_normals":
            hit_counts = cls._check_face_normals(trimesh, xs, ys, z_start, nx, ny)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Flip Y axis for image coordinates (image origin is top-left)
        hit_counts = hit_counts[::-1, :]

        # Vectorized color assignment
        # BLACK (0,0,0) for no hits, GREEN (0,255,0) for 1 hit, RED (255,0,0) for 2+ hits
        image = np.zeros((ny, nx, 3), dtype=np.uint8)
        image[hit_counts == 1] = [0, 255, 0]   # GREEN - clean geometry
        image[hit_counts >= 2] = [255, 0, 0]   # RED - backdraft
        # hit_counts == 0 stays BLACK (background)

        # Calculate stats
        hit_count = np.sum(hit_counts >= 1)
        backdraft_count = np.sum(hit_counts >= 2)
        backdraft_pct = backdraft_count / max(1, hit_count) * 100

        log.info("Complete: %d pixels with geometry, %d backdraft pixels (%.1f%%)", hit_count, backdraft_count, backdraft_pct)

        # Draw filename on image if requested
        if show_filename:
            # Get filename from mesh metadata
            filename = trimesh.metadata.get('file_name', '')
            if filename:
                # Remove extension for cleaner display
                filename = os.path.splitext(filename)[0]
            else:
                filename = "unknown"

            # Convert to PIL Image for text drawing
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)

            # Try to get a reasonable font size based on image dimensions
            font_size = max(12, min(ny, nx) // 30)
            try:
                # Try to load a monospace font
                font = ImageFont.truetype("arial.ttf", font_size)
            except (IOError, OSError):
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except (IOError, OSError):
                    # Fall back to default font
                    font = ImageFont.load_default()

            # Draw text with white color and slight shadow for visibility
            text_x = 10
            text_y = 10
            # Shadow
            draw.text((text_x + 1, text_y + 1), filename, fill=(0, 0, 0), font=font)
            # Main text
            draw.text((text_x, text_y), filename, fill=(255, 255, 255), font=font)

            # Convert back to numpy array
            image = np.array(pil_image)

        # Convert to ComfyUI IMAGE format: (B, H, W, C) float32 0-1
        img_arr = (image.astype(np.float32) / 255.0)[np.newaxis, ...]

        return io.NodeOutput(img_arr)

    @staticmethod
    def _raytrace_trimesh(mesh, xs, ys, z_start, nx, ny, total_rays):
        """Batch ray casting using trimesh (embree backend) with progress display."""
        log.info("Using trimesh backend (embree)...")

        # Create batch ray origins and directions
        xx, yy = np.meshgrid(xs, ys)
        origins = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            np.full(total_rays, z_start)
        ]).astype(np.float64)

        # All rays point straight down (-Z)
        directions = np.tile([0.0, 0.0, -1.0], (total_rays, 1))

        # Process in chunks for progress display (each chunk = ~100k rays)
        chunk_size = 100000
        hit_counts = np.zeros(total_rays, dtype=np.int32)
        total_intersections = 0

        if total_rays <= chunk_size:
            # Small enough to do in one batch
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=origins,
                ray_directions=directions,
                multiple_hits=True
            )
            np.add.at(hit_counts, index_ray, 1)
            total_intersections = len(locations)
        else:
            # Process in chunks with progress
            num_chunks = (total_rays + chunk_size - 1) // chunk_size
            log.info("Processing %d chunks of ~%d rays...", num_chunks, chunk_size)

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_rays)

                chunk_origins = origins[start_idx:end_idx]
                chunk_directions = directions[start_idx:end_idx]

                locations, index_ray, index_tri = mesh.ray.intersects_location(
                    ray_origins=chunk_origins,
                    ray_directions=chunk_directions,
                    multiple_hits=True
                )

                # Offset indices back to global array
                np.add.at(hit_counts, index_ray + start_idx, 1)
                total_intersections += len(locations)

                progress = (i + 1) / num_chunks * 100
                log.info("Progress: %.0f%% (%d/%d rays)", progress, end_idx, total_rays)

        log.info("Ray casting complete, %d total intersections", total_intersections)

        hit_counts = hit_counts.reshape(ny, nx)
        return hit_counts

    @staticmethod
    def _raytrace_pyvista(mesh, xs, ys, z_start, z_end, nx, ny):
        """Batch ray casting using PyVista multi_ray_trace."""
        import pyvista as pv

        log.info("Using pyvista backend (multi_ray_trace)...")

        # Convert trimesh to PyVista PolyData
        faces_pv = np.hstack([
            np.full((len(mesh.faces), 1), 3),
            mesh.faces
        ]).astype(np.int64).flatten()
        pv_mesh = pv.PolyData(np.array(mesh.vertices), faces_pv)

        # Create batch ray origins and directions
        total_rays = nx * ny
        xx, yy = np.meshgrid(xs, ys)
        origins = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            np.full(total_rays, z_start)
        ])
        # Directions: pointing down (-Z)
        directions = np.zeros((total_rays, 3))
        directions[:, 2] = -1.0

        log.info("Casting %d rays...", total_rays)

        # Use multi_ray_trace with first_point=False for ALL intersections
        points, ind_ray, ind_tri = pv_mesh.multi_ray_trace(
            origins=origins,
            directions=directions,
            first_point=False
        )

        log.info("Ray casting complete, %d total intersections", len(points))

        # Count hits per ray
        hit_counts = np.zeros(total_rays, dtype=np.int32)
        if len(ind_ray) > 0:
            np.add.at(hit_counts, ind_ray, 1)
        hit_counts = hit_counts.reshape(ny, nx)

        return hit_counts

    @staticmethod
    def _check_face_normals(mesh, xs, ys, z_start, nx, ny):
        """
        Check face normal Z-component consistency and render visualization.

        Requires mesh to be a single connected component.
        Flags faces whose Z-normal points opposite to the majority direction.

        Returns:
            hit_counts: (ny, nx) array where:
                0 = no geometry (background)
                1 = face with correct normal (majority Z direction)
                2 = face with flipped normal (minority Z direction)
        """
        import trimesh as trimesh_module

        log.info("Using face_normals backend...")

        # 1. Check for single connected component
        # Use edges_unique to build adjacency for connected component analysis
        adjacency = mesh.face_adjacency
        if len(adjacency) == 0:
            # Single face or no adjacency - treat as single component
            num_components = 1
        else:
            components = trimesh_module.graph.connected_components(adjacency, nodes=np.arange(len(mesh.faces)))
            num_components = len(components)

        if num_components > 1:
            raise ValueError(
                f"Mesh has {num_components} disconnected components. "
                f"Face normals check requires a single connected mesh. "
                f"Use 'Split by Connectivity' node first to separate components."
            )

        log.info("Mesh is single connected component")

        # 2. Analyze face normal Z-components
        normals_z = mesh.face_normals[:, 2]
        up_count = np.sum(normals_z > 0)
        down_count = np.sum(normals_z < 0)
        zero_count = np.sum(normals_z == 0)  # Faces pointing sideways

        log.info("Face normals: %d pointing up (+Z), %d pointing down (-Z), %d sideways", up_count, down_count, zero_count)

        # Determine majority direction
        majority_up = up_count >= down_count

        if majority_up:
            flipped_mask = normals_z < 0  # Faces pointing down are flipped
            log.info("Majority direction: UP (+Z), %d flipped faces", down_count)
        else:
            flipped_mask = normals_z > 0  # Faces pointing up are flipped
            log.info("Majority direction: DOWN (-Z), %d flipped faces", up_count)

        flipped_count = np.sum(flipped_mask)
        flipped_pct = flipped_count / len(mesh.faces) * 100
        log.info("Flipped faces: %d/%d (%.1f%%)", flipped_count, len(mesh.faces), flipped_pct)

        # 3. Cast rays to find which face is visible at each pixel (single hit)
        total_rays = nx * ny
        xx, yy = np.meshgrid(xs, ys)
        origins = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            np.full(total_rays, z_start)
        ]).astype(np.float64)
        directions = np.tile([0.0, 0.0, -1.0], (total_rays, 1))

        # Single-hit ray cast to get the topmost face at each pixel
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False  # Only get first (topmost) hit
        )

        log.info("Ray casting complete, %d hits", len(index_ray))

        # 4. Build result grid based on whether topmost face is flipped
        hit_counts = np.zeros(total_rays, dtype=np.int32)

        if len(index_ray) > 0:
            # Vectorized: check if each hit face is flipped
            is_flipped = flipped_mask[index_tri]
            # 1 = correct normal (green), 2 = flipped normal (red)
            hit_counts[index_ray] = np.where(is_flipped, 2, 1)

        return hit_counts.reshape(ny, nx)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeomPackBackdraftView": BackdraftViewNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackBackdraftView": "Backdraft View",
}
