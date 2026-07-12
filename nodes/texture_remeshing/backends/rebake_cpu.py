# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Rebake Texture (CPU backend).

Bakes original_mesh's texture onto uv_mesh's own UV layout, via per-TEXEL closest-point
projection onto original_mesh -- a real texture bake, not a per-vertex color hack. For
every texel covered by a uv_mesh UV triangle: interpolate its 3D surface position,
closest-point-project onto original_mesh, barycentric-interpolate the ORIGINAL mesh's
own UV at that point, and bilinear-sample the original texture there.
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

from ._rebake_common import get_texture_array, sample_bilinear, dilate_texture, build_textured_mesh, to_comfy_image

log = logging.getLogger("geometrypack")


def _barycentric_2d(px, py, ax, ay, bx, by, cx, cy):
    """Vectorized 2D barycentric coords of points (px,py) w.r.t. triangle (a,b,c)."""
    v0x, v0y = bx - ax, by - ay
    v1x, v1y = cx - ax, cy - ay
    v2x, v2y = px - ax, py - ay
    d00 = v0x * v0x + v0y * v0y
    d01 = v0x * v1x + v0y * v1y
    d11 = v1x * v1x + v1y * v1y
    d20 = v2x * v0x + v2y * v0y
    d21 = v2x * v1x + v2y * v1y
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-14:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def rasterize_uv_mesh(uv_mesh, texture_size, pbar=None, pbar_range=(0, 100)):
    """Rasterize every UV triangle of uv_mesh into texel centers. Returns parallel arrays:
    pixel x, pixel y (int, image coords), and the interpolated 3D surface position.

    pbar: optional comfy.utils.ProgressBar (0-100 scale) shared across bake phases.
    pbar_range: (start, end) percent this phase should advance the bar through."""
    if not hasattr(uv_mesh, 'visual') or not hasattr(uv_mesh.visual, 'uv') or uv_mesh.visual.uv is None:
        raise ValueError("uv_mesh has no UV coordinates -- run a UV Unwrap node on it first "
                          "(Xatlas, ARAP, LSCM, Harmonic, Geogram ABF, ...).")
    uvs = np.asarray(uv_mesh.visual.uv, dtype=np.float64)
    verts = np.asarray(uv_mesh.vertices, dtype=np.float64)
    faces = np.asarray(uv_mesh.faces, dtype=np.int64)
    T = int(texture_size)
    n_faces = len(faces)
    p_start, p_end = pbar_range
    report_every = max(1, n_faces // 50)

    px_all, py_all, pos_all = [], [], []
    for fi, f in enumerate(faces):
        if pbar is not None and fi % report_every == 0:
            pbar.update_absolute(int(p_start + (p_end - p_start) * fi / n_faces))
        tri_uv = uvs[f]
        tri_3d = verts[f]
        fx = tri_uv[:, 0] * (T - 1)
        fy = (1.0 - tri_uv[:, 1]) * (T - 1)
        xmin = max(0, int(np.floor(fx.min())))
        xmax = min(T - 1, int(np.ceil(fx.max())))
        ymin = max(0, int(np.floor(fy.min())))
        ymax = min(T - 1, int(np.ceil(fy.max())))
        if xmax < xmin or ymax < ymin:
            continue
        xs, ys = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
        xs = xs.ravel().astype(np.float64) + 0.5
        ys = ys.ravel().astype(np.float64) + 0.5
        bary = _barycentric_2d(xs, ys, fx[0], fy[0], fx[1], fy[1], fx[2], fy[2])
        if bary is None:
            continue
        u, v, w = bary
        inside = (u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6)
        if not inside.any():
            continue
        u, v, w = u[inside], v[inside], w[inside]
        pos = u[:, None] * tri_3d[0] + v[:, None] * tri_3d[1] + w[:, None] * tri_3d[2]
        px_all.append((xs[inside] - 0.5).astype(np.int64))
        py_all.append((ys[inside] - 0.5).astype(np.int64))
        pos_all.append(pos)

    if not pos_all:
        raise ValueError("No texels rasterized -- UV layout may be degenerate or texture_size too small.")

    if pbar is not None:
        pbar.update_absolute(p_end)

    return np.concatenate(px_all), np.concatenate(py_all), np.concatenate(pos_all, axis=0)


_CLOSEST_POINT_CHUNK = 20000


def chunked_closest_point(mesh, points, pbar=None, pbar_range=(0, 100)):
    """trimesh.proximity.closest_point in batches, reporting progress between them --
    a single call against hundreds of thousands of points can run for a long time with
    no feedback otherwise."""
    n = len(points)
    p_start, p_end = pbar_range
    n_chunks = max(1, (n + _CLOSEST_POINT_CHUNK - 1) // _CLOSEST_POINT_CHUNK)
    closest_all, dist_all, tri_all = [], [], []
    for ci, start in enumerate(range(0, n, _CLOSEST_POINT_CHUNK)):
        chunk = points[start:start + _CLOSEST_POINT_CHUNK]
        c, d, t = trimesh_module.proximity.closest_point(mesh, chunk)
        closest_all.append(c)
        dist_all.append(d)
        tri_all.append(t)
        if pbar is not None:
            pbar.update_absolute(int(p_start + (p_end - p_start) * (ci + 1) / n_chunks))
    return np.concatenate(closest_all), np.concatenate(dist_all), np.concatenate(tri_all)


class RebakeTextureCPUNode(io.ComfyNode):
    """Bake original_mesh's texture onto uv_mesh's UV layout via per-texel closest-point
    projection (CPU, numpy-vectorized)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRebakeTexture_CPU",
            display_name="Rebake Texture (CPU)",
            category="geompack/texture_remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("original_mesh"),
                io.Custom("TRIMESH").Input("uv_mesh"),
                io.Int.Input("texture_size", default=1024, min=64, max=8192, step=64),
                io.Int.Input("bake_margin", default=8, min=0, max=64),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="textured_mesh"),
                io.Image.Output(display_name="texture"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, original_mesh, uv_mesh, texture_size=1024, bake_margin=8):
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(100)
        except Exception:
            pbar = None

        src_tex, src_uv = get_texture_array(original_mesh)
        log.info("Source texture: %dx%d", src_tex.shape[1], src_tex.shape[0])
        log.info("Rasterizing %d faces at %dx%d...", len(uv_mesh.faces), texture_size, texture_size)

        px, py, pos = rasterize_uv_mesh(uv_mesh, texture_size, pbar=pbar, pbar_range=(0, 15))
        log.info("Rasterized %d texels", len(px))

        log.info("Closest-point projection onto original_mesh (%d faces)...", len(original_mesh.faces))
        closest, dist, tri_ids = chunked_closest_point(original_mesh, pos, pbar=pbar, pbar_range=(15, 95))

        src_faces = np.asarray(original_mesh.faces)[tri_ids]
        src_tris = np.asarray(original_mesh.vertices)[src_faces]
        bary = trimesh_module.triangles.points_to_barycentric(src_tris, closest)
        src_face_uvs = src_uv[src_faces]  # (N,3,2)
        sampled_uv = np.einsum('ij,ijk->ik', bary, src_face_uvs)

        colors = sample_bilinear(src_tex, sampled_uv[:, 0], sampled_uv[:, 1])

        T = int(texture_size)
        out_tex = np.zeros((T, T, 3), dtype=np.float64)
        out_mask = np.zeros((T, T), dtype=bool)
        out_tex[py, px] = colors
        out_mask[py, px] = True

        out_tex, out_mask = dilate_texture(out_tex, out_mask, bake_margin)
        out_tex_u8 = np.clip(out_tex, 0, 255).astype(np.uint8)

        filled_pct = 100.0 * out_mask.sum() / (T * T)
        log.info("Bake complete: %.1f%% of texture filled", filled_pct)
        if pbar is not None:
            pbar.update_absolute(100)

        result_mesh = build_textured_mesh(uv_mesh, out_tex_u8)
        comfy_image = to_comfy_image(out_tex_u8)

        info = f"""Rebake Texture (CPU) Results:

Source: {len(original_mesh.vertices):,} verts, {len(original_mesh.faces):,} faces, {src_tex.shape[1]}x{src_tex.shape[0]} texture
Target: {len(uv_mesh.vertices):,} verts, {len(uv_mesh.faces):,} faces
Baked texture: {T}x{T}, {filled_pct:.1f}% filled, margin={bake_margin}px
"""
        return io.NodeOutput(result_mesh, comfy_image, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRebakeTexture_CPU": RebakeTextureCPUNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRebakeTexture_CPU": "Rebake Texture (CPU)"}
