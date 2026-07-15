# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Shared helpers for the Rebake Texture backends (CPU + GPU)."""

import numpy as np


def get_texture_array(mesh):
    """Extract the (H,W,3) uint8 texture array + per-vertex UV array from a textured trimesh."""
    from PIL import Image

    if not hasattr(mesh, 'visual') or mesh.visual is None:
        raise ValueError("original_mesh has no visual/texture data")
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        raise ValueError("original_mesh has no UV coordinates")
    material = getattr(mesh.visual, 'material', None)
    img = None
    if material is not None:
        img = getattr(material, 'baseColorTexture', None)
        if img is None:
            img = getattr(material, 'image', None)
    if img is None:
        raise ValueError("original_mesh material has no texture image to bake from")
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    arr = np.array(img.convert('RGB'))
    return arr, np.asarray(mesh.visual.uv, dtype=np.float64)


def sample_bilinear(image, u, v):
    """Bilinear-sample an (H,W,3) image array at UV coords (arrays in [0,1]).
    v=0 is texture bottom -- matches this pack's v-flip convention used throughout
    (e.g. preview_mesh_vtk.py's texture extraction)."""
    H, W = image.shape[:2]
    x = np.clip(u, 0.0, 1.0) * (W - 1)
    y = (1.0 - np.clip(v, 0.0, 1.0)) * (H - 1)
    x0 = np.clip(np.floor(x).astype(np.int64), 0, W - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    img = image.astype(np.float64)
    c00 = img[y0, x0]
    c10 = img[y0, x1]
    c01 = img[y1, x0]
    c11 = img[y1, x1]
    top = c00 * (1 - fx) + c10 * fx
    bot = c01 * (1 - fx) + c11 * fx
    return top * (1 - fy) + bot * fy


def dilate_texture(rgb, mask, margin_px):
    """Grow filled texels into empty ones within margin_px (nearest-neighbor fill via
    Euclidean distance transform) -- standard texture-bake padding, prevents black seam
    bleeding at UV island borders when mipmapped/filtered at render time."""
    if margin_px <= 0 or mask.all():
        return rgb, mask
    from scipy.ndimage import distance_transform_edt

    empty = ~mask
    dist, (iy, ix) = distance_transform_edt(empty, return_indices=True)
    fill = (dist > 0) & (dist <= margin_px) & empty
    out = rgb.copy()
    out[fill] = rgb[iy[fill], ix[fill]]
    return out, (mask | fill)


def build_textured_mesh(uv_mesh, texture_u8):
    """Attach a baked (H,W,3) uint8 texture as a real UV-mapped material on a copy of uv_mesh."""
    from PIL import Image
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import SimpleMaterial

    result_mesh = uv_mesh.copy()
    pil_img = Image.fromarray(texture_u8, mode='RGB')
    result_mesh.visual = TextureVisuals(uv=uv_mesh.visual.uv, material=SimpleMaterial(image=pil_img))
    return result_mesh


def to_comfy_image(texture_u8):
    """Convert an (H,W,3) uint8 array to ComfyUI IMAGE format ([1,H,W,3] float32 torch tensor)."""
    import torch
    return torch.from_numpy(texture_u8.astype(np.float32) / 255.0)[None, ...]
