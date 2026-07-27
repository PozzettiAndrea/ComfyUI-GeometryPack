# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Shared helpers for the Depth Map to Mesh (TextureToGeometry) backends."""

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


def extract_heightmap(depth, invert_height):
    """Parse the 'depth' input (IMAGE or MASK, various shapes) into a float32 (H,W)
    heightmap in [0,1], applying the invert_height toggle. Raises on missing/bad input."""
    if depth is None:
        raise ValueError("'depth' input is required (an IMAGE or a MASK)")

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

    height, width = heightmap.shape
    log.info("Using native resolution: %dx%d, range: [%.3f, %.3f]",
             width, height, heightmap.min(), heightmap.max())

    heightmap = heightmap.astype(np.float32)
    if heightmap.max() > 1.0:
        heightmap = heightmap / 255.0

    if invert_height == "true":
        heightmap = 1.0 - heightmap

    return heightmap


def pixel_keep_mask(mask, height, width):
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


def heightmap_to_points(heightmap, height_scale, keep_mask=None):
    """Convert heightmap to 3D point cloud (x,y in [-1,1], z = value * height_scale)."""
    height, width = heightmap.shape
    points = []
    valid_mask = np.ones((height, width), dtype=bool)

    for y in range(height):
        for x in range(width):
            h = heightmap[y, x]

            if keep_mask is not None and not keep_mask[y, x]:
                valid_mask[y, x] = False
                continue

            nx = (x / (width - 1)) * 2.0 - 1.0
            ny = (y / (height - 1)) * 2.0 - 1.0
            nz = h * height_scale

            points.append([nx, ny, nz])

    return np.array(points, dtype=np.float64), valid_mask


def build_info(width, height, height_scale, invert_height,
               keep_mask, backend_name, backend_detail, mesh):
    """Shared info-string formatting for every Depth Map to Mesh backend."""
    height_min = mesh.vertices[:, 2].min()
    height_max = mesh.vertices[:, 2].max()
    height_range = height_max - height_min

    return f"""Depth Map to Mesh Results:

Input:
  Resolution: {width}x{height}
  Height Scale: {height_scale}
  Inverted: {invert_height}
  Mask: {'yes (%d%% kept)' % int(100 * keep_mask.mean()) if keep_mask is not None else 'none'}

Backend: {backend_name}
  {backend_detail}

Output Mesh:
  Vertices: {len(mesh.vertices):,}
  Faces: {len(mesh.faces):,}
  Height Range: [{height_min:.3f}, {height_max:.3f}] (span: {height_range:.3f})
  Bounds: {mesh.bounds.tolist()}
  Watertight: {mesh.is_watertight}
"""
