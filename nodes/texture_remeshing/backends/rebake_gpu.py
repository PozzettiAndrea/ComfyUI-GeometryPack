# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Rebake Texture (GPU backend).

Same bake as the CPU backend (per-texel closest-point projection from uv_mesh onto
original_mesh, sampling original_mesh's own UVs/texture at the projected point), but the
closest-point search runs on a real GPU BVH: cumesh.bvh.cuBVH -- already a declared
GeometryPack dependency (nodes/comfy-env.toml [cuda] packages, also used by
repair/fix_normals_backends/cumesh_raystab.py). unsigned_distance(..., return_uvw=True)
gives (distance, face_id, barycentric_uvw) directly, so no separate points_to_barycentric
step is needed either.
"""

import logging

import numpy as np
from comfy_api.latest import io

from ._rebake_common import get_texture_array, sample_bilinear, dilate_texture, build_textured_mesh, to_comfy_image
from .rebake_cpu import rasterize_uv_mesh

log = logging.getLogger("geometrypack")

_QUERY_CHUNK = 500_000  # cuBVH itself is O(N log M); chunking here is just to bound
                        # per-call GPU memory and to give the progress bar something to report.


def closest_point_gpu(query_points, mesh_vertices, mesh_faces, device, pbar=None, pbar_range=(0, 100)):
    """Closest point of query_points (N,3) onto a triangle mesh via cumesh's cuBVH.
    Returns (tri_ids (N,) np.int64, uvw (N,3) np.float64) -- uvw are the barycentric
    weights of the closest point w.r.t. mesh_faces[tri_ids], in vertex order.

    pbar: optional comfy.utils.ProgressBar (0-100 scale) shared across bake phases.
    pbar_range: (start, end) percent this phase should advance the bar through."""
    import torch
    from cumesh.bvh import cuBVH

    if len(mesh_faces) <= 8:
        raise ValueError(f"cuBVH requires > 8 triangles in original_mesh, got {len(mesh_faces)} "
                          f"-- use the CPU backend for very small meshes.")

    V = np.ascontiguousarray(mesh_vertices, dtype=np.float32)
    F = np.ascontiguousarray(mesh_faces, dtype=np.int32)
    bvh = cuBVH(V, F)

    p = torch.as_tensor(query_points, dtype=torch.float32, device=device)
    N = p.shape[0]
    p_start, p_end = pbar_range
    n_chunks = max(1, (N + _QUERY_CHUNK - 1) // _QUERY_CHUNK)

    tri_ids = np.empty(N, dtype=np.int64)
    uvw = np.empty((N, 3), dtype=np.float64)
    for ci, qi in enumerate(range(0, N, _QUERY_CHUNK)):
        chunk = p[qi:qi + _QUERY_CHUNK]
        _dist, fid, w = bvh.unsigned_distance(chunk, return_uvw=True)
        tri_ids[qi:qi + chunk.shape[0]] = fid.detach().cpu().numpy()
        uvw[qi:qi + chunk.shape[0]] = w.detach().double().cpu().numpy()
        if pbar is not None:
            pbar.update_absolute(int(p_start + (p_end - p_start) * (ci + 1) / n_chunks))

    del bvh
    try:
        import comfy.model_management
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass

    return tri_ids, uvw


class RebakeTextureGPUNode(io.ComfyNode):
    """Bake original_mesh's texture onto uv_mesh's UV layout via per-texel closest-point
    projection (GPU-accelerated closest-point search)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRebakeTexture_GPU",
            display_name="Rebake Texture (GPU)",
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
        import torch

        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(100)
        except Exception:
            pbar = None

        if not torch.cuda.is_available():
            log.warning("CUDA not available -- Rebake Texture (GPU) will run on CPU tensors "
                        "(no speedup over the CPU backend).")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        src_tex, src_uv = get_texture_array(original_mesh)
        log.info("Source texture: %dx%d", src_tex.shape[1], src_tex.shape[0])
        log.info("Rasterizing %d faces at %dx%d...", len(uv_mesh.faces), texture_size, texture_size)

        try:
            from ._rebake_gl import rasterize_uv_mesh_gl
            px, py, pos = rasterize_uv_mesh_gl(uv_mesh, texture_size)
            if pbar is not None:
                pbar.update_absolute(15)
            log.info("Rasterized via hardware (OpenGL/EGL)")
        except Exception as e:
            log.warning("Hardware rasterization unavailable (%s) -- falling back to CPU rasterizer.", e)
            px, py, pos = rasterize_uv_mesh(uv_mesh, texture_size, pbar=pbar, pbar_range=(0, 15))
        log.info("Rasterized %d texels", len(px))

        n_src_faces = len(original_mesh.faces)
        log.info("Closest-point projection onto original_mesh (%d faces) via cuBVH "
                  "on %s (chunked %d texels/call)...", n_src_faces, device, _QUERY_CHUNK)
        tri_ids, bary = closest_point_gpu(
            pos, np.asarray(original_mesh.vertices), np.asarray(original_mesh.faces), device,
            pbar=pbar, pbar_range=(15, 95))

        src_faces = np.asarray(original_mesh.faces)[tri_ids]
        src_face_uvs = src_uv[src_faces]
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

        info = f"""Rebake Texture (GPU) Results:

Device: {device}
Source: {len(original_mesh.vertices):,} verts, {len(original_mesh.faces):,} faces, {src_tex.shape[1]}x{src_tex.shape[0]} texture
Target: {len(uv_mesh.vertices):,} verts, {len(uv_mesh.faces):,} faces
Baked texture: {T}x{T}, {filled_pct:.1f}% filled, margin={bake_margin}px
"""
        return io.NodeOutput(result_mesh, comfy_image, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRebakeTexture_GPU": RebakeTextureGPUNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRebakeTexture_GPU": "Rebake Texture (GPU)"}
