# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Shared helpers for the Depth + Normals to Mesh backends.

Builds an oriented point cloud from a depth map (positions) + a normal map
(orientations, Nx in R / Ny in G, Nz derived from the unit constraint), which the
backends then surface-reconstruct (Poisson / Ball Pivoting)."""

import logging

import numpy as np

log = logging.getLogger("geometrypack")


def _to_numpy(x):
    """Convert tensor or array to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    return np.array(x)


def build_oriented_point_cloud(normal_map, depth, resolution, depth_scale,
                               mask=None, invert_depth="false"):
    """Parse depth (IMAGE or MASK) + normal map, resize to `resolution`, and build
    the oriented point cloud. Returns (points (N,3), normals (N,3), keep_mask or None,
    depth_shape, normal_shape, grid_shape (H,W))."""
    if depth is None:
        raise ValueError("'depth' input is required (an IMAGE or a MASK)")

    # The 'depth' input may be an IMAGE (B,H,W,C) or a MASK (B,H,W) -> grayscale.
    arr = _to_numpy(depth)
    if arr.ndim == 4:                       # IMAGE batch (B,H,W,C)
        img = arr[0]
        depth_arr = np.mean(img[:, :, :3], axis=2) if img.shape[2] >= 3 else img[:, :, 0]
        log.info("depth: IMAGE input (RGB averaged to grayscale)")
    elif arr.ndim == 3:                     # MASK batch (B,H,W) or single image (H,W,C)
        if arr.shape[2] in (3, 4):
            depth_arr = np.mean(arr[:, :, :3], axis=2)
        elif arr.shape[2] == 1:
            depth_arr = arr[:, :, 0]
        else:
            depth_arr = arr[0]
        log.info("depth: 3D input -> %s", depth_arr.shape)
    elif arr.ndim == 2:
        depth_arr = arr
    else:
        raise ValueError(f"Unexpected 'depth' shape {arr.shape}; expected IMAGE or MASK")

    # Normalize to [0, 1]
    depth_min, depth_max = depth_arr.min(), depth_arr.max()
    if depth_max > depth_min:
        depth_arr = (depth_arr - depth_min) / (depth_max - depth_min)

    log.info("Depth size: %s, range: [%.3f, %.3f]", depth_arr.shape, depth_min, depth_max)

    # Extract normal map from tensor (B, H, W, C)
    normal_arr = _to_numpy(normal_map)
    if normal_arr.ndim == 4:
        normal_arr = normal_arr[0]
    if len(normal_arr.shape) == 2:
        raise ValueError("Normal map must be RGB image with Nx in R, Ny in G channels")

    log.info("Normal map size: %s", normal_arr.shape)

    # Sampling grid: `resolution` targets the LONGEST side, aspect preserved,
    # never upsampled past the input's native size.
    in_H, in_W = depth_arr.shape
    scale = min(float(resolution) / max(in_H, in_W), 1.0)
    out_H = max(2, int(round(in_H * scale)))
    out_W = max(2, int(round(in_W * scale)))
    log.info("Sampling grid: %dx%d (native %dx%d, resolution=%d)",
             out_W, out_H, in_W, in_H, resolution)

    # Float-space resize (bilinear) -- no uint8 round-trip, so float depth
    # (e.g. 32-bit EXR) keeps its full precision.
    from scipy.ndimage import zoom
    if (out_H, out_W) != (in_H, in_W):
        depth_resized = zoom(depth_arr.astype(np.float64),
                             (out_H / in_H, out_W / in_W), order=1).astype(np.float32)
    else:
        depth_resized = depth_arr.astype(np.float32)

    n_H, n_W = normal_arr.shape[:2]
    if (out_H, out_W) != (n_H, n_W):
        normal_resized = zoom(normal_arr[:, :, :3].astype(np.float64),
                              (out_H / n_H, out_W / n_W, 1), order=1).astype(np.float32)
    else:
        normal_resized = normal_arr[:, :, :3].astype(np.float32)

    if invert_depth == "true":
        depth_resized = 1.0 - depth_resized

    # Optional pixel-selection mask, nearest-resized to the sampling grid
    keep_mask = None
    if mask is not None:
        marr = _to_numpy(mask).astype(np.float32)
        if marr.ndim == 3:
            marr = marr[0]
        elif marr.ndim == 4:
            marr = marr[0, :, :, 0]
        if marr.max() > 1.0:
            marr = marr / 255.0
        keep = marr > 0.5
        if keep.shape != (out_H, out_W):
            ys = np.linspace(0, keep.shape[0] - 1, out_H).round().astype(int)
            xs = np.linspace(0, keep.shape[1] - 1, out_W).round().astype(int)
            keep = keep[np.ix_(ys, xs)]
        keep_mask = keep
        log.info("mask input: keeping %d/%d pixels (%.1f%%)",
                 int(keep.sum()), keep.size, 100.0 * keep.mean())

    # Build oriented point cloud. XY are normalised by the LONGEST side (centered),
    # preserving the image's aspect ratio; for square inputs this reduces to the
    # classic x/(W-1)*2-1 mapping.
    height, width = out_H, out_W
    s = 2.0 / (max(out_W, out_H) - 1)
    points = []
    normals = []

    for y in range(height):
        for x in range(width):
            d = depth_resized[y, x]

            if keep_mask is not None and not keep_mask[y, x]:
                continue

            px = (x - (width - 1) / 2.0) * s
            py = (y - (height - 1) / 2.0) * s
            pz = d * depth_scale
            points.append([px, py, pz])

            # Normal from RGB (R=Nx, G=Ny, derive Nz). The point positions use
            # image-row-down as +y, while normal maps encode +ny as image-UP --
            # flip G so the orientation field is consistent with the geometry
            # (Poisson fits the implicit surface's gradient to these normals;
            # a y-inconsistent field produces mush).
            nx = normal_resized[y, x, 0] * 2.0 - 1.0  # [0,1] -> [-1,1]
            ny = -(normal_resized[y, x, 1] * 2.0 - 1.0)

            # Derive Nz from unit normal constraint: Nx^2 + Ny^2 + Nz^2 = 1
            nz_sq = max(0.0, 1.0 - nx*nx - ny*ny)
            nz = np.sqrt(nz_sq)

            # Normalize to ensure unit length
            length = np.sqrt(nx*nx + ny*ny + nz*nz)
            if length > 0:
                nx, ny, nz = nx/length, ny/length, nz/length

            normals.append([nx, ny, nz])

    points = np.array(points, dtype=np.float64)
    normals = np.array(normals, dtype=np.float64)

    log.info("Point cloud: %d points", len(points))
    if len(points) < 10:
        raise ValueError(f"Too few valid points ({len(points)}). Check the depth map and mask.")

    return points, normals, keep_mask, depth_arr.shape, normal_arr.shape, (out_H, out_W)


def build_info(depth_shape, normal_shape, grid_shape, depth_scale, method,
               points, keep_mask, mesh, method_info):
    return f"""Depth + Normals to Mesh Results:

Input:
  Depth Map: {depth_shape[0]}x{depth_shape[1]}
  Normal Map: {normal_shape[0]}x{normal_shape[1]}
  Sampling Grid: {grid_shape[1]}x{grid_shape[0]}
  Depth Scale: {depth_scale}
  Method: {method}

Point Cloud:
  Valid Points: {len(points):,}
  Mask: {'yes (%d%% kept)' % int(100 * keep_mask.mean()) if keep_mask is not None else 'none'}

Output Mesh:
  Vertices: {len(mesh.vertices):,}
  Faces: {len(mesh.faces):,}
  Watertight: {mesh.is_watertight}
  Bounds: {mesh.bounds.tolist()}

{method_info}
"""
