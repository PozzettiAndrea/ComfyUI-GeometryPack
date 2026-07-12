# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Rebake Texture (GPU backend).

Same bake as the CPU backend (per-texel closest-point projection from uv_mesh onto
original_mesh, sampling original_mesh's own UVs/texture at the projected point), but the
closest-point search itself runs on GPU via a from-scratch point-to-triangle-mesh
distance (chunked brute force -- no pytorch3d/kaolin/nvdiffrast in this env).

Note this trades CPU trimesh's spatial-index search (O(N log M)) for GPU-parallel brute
force (O(N*M), chunked to bound memory) -- a real win when hardware parallelism outweighs
the worse asymptotics (large N texels against a moderate-size original_mesh), but not
guaranteed to beat the CPU backend on every mesh/resolution combination.
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

from ._rebake_common import get_texture_array, sample_bilinear, dilate_texture, build_textured_mesh, to_comfy_image
from .rebake_cpu import rasterize_uv_mesh

log = logging.getLogger("geometrypack")

_QUERY_CHUNK = 8192
_FACE_CHUNK = 4096


def _closest_point_on_triangles(p, a, b, c):
    """Ericson (Real-Time Collision Detection, 5.1.5) closest point on a batch of triangles.
    p: (N,3) query points. a,b,c: (M,3) triangle vertices. Returns (N,M,3) closest points --
    caller must chunk N and M to bound memory (this is O(N*M) in time and memory)."""
    import torch

    p_ = p[:, None, :]
    a_ = a[None, :, :]
    b_ = b[None, :, :]
    c_ = c[None, :, :]

    ab = b_ - a_
    ac = c_ - a_
    ap = p_ - a_
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    mask_a = (d1 <= 0) & (d2 <= 0)

    bp = p_ - b_
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    mask_b = (d3 >= 0) & (d4 <= d3)

    vc = d1 * d4 - d3 * d2
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)

    cp = p_ - c_
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    mask_c = (d6 >= 0) & (d5 <= d6)

    vb = d5 * d2 - d1 * d6
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 >= 0)

    va = d3 * d6 - d5 * d4
    mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)

    eps = 1e-20
    denom = 1.0 / (va + vb + vc + eps)
    v_f = vb * denom
    w_f = vc * denom
    face_pt = a_ + ab * v_f[..., None] + ac * w_f[..., None]

    v_ab = d1 / (d1 - d3 + eps)
    ab_pt = a_ + v_ab[..., None] * ab

    v_ac = d2 / (d2 - d6 + eps)
    ac_pt = a_ + v_ac[..., None] * ac

    v_bc = (d4 - d3) / ((d4 - d3) - (d5 - d6) + eps)
    bc_pt = b_ + v_bc[..., None] * (c_ - b_)

    out = face_pt
    remaining = ~mask_a
    out = torch.where(mask_a[..., None], a_.expand_as(out), out)
    m = remaining & mask_b
    out = torch.where(m[..., None], b_.expand_as(out), out)
    remaining = remaining & ~mask_b
    m = remaining & mask_c
    out = torch.where(m[..., None], c_.expand_as(out), out)
    remaining = remaining & ~mask_c
    m = remaining & mask_ab
    out = torch.where(m[..., None], ab_pt, out)
    remaining = remaining & ~mask_ab
    m = remaining & mask_ac
    out = torch.where(m[..., None], ac_pt, out)
    remaining = remaining & ~mask_ac
    m = remaining & mask_bc
    out = torch.where(m[..., None], bc_pt, out)

    return out


def closest_point_gpu(query_points, mesh_vertices, mesh_faces, device, pbar=None, pbar_range=(0, 100)):
    """Chunked brute-force closest point of query_points (N,3) onto a triangle mesh.
    Returns (closest (N,3) np.float64, tri_ids (N,) np.int64).

    pbar: optional comfy.utils.ProgressBar (0-100 scale) shared across bake phases.
    pbar_range: (start, end) percent this phase should advance the bar through."""
    import torch

    p = torch.as_tensor(query_points, dtype=torch.float32, device=device)
    tris = mesh_vertices[mesh_faces]  # (M,3,3) numpy
    a_all = torch.as_tensor(tris[:, 0], dtype=torch.float32, device=device)
    b_all = torch.as_tensor(tris[:, 1], dtype=torch.float32, device=device)
    c_all = torch.as_tensor(tris[:, 2], dtype=torch.float32, device=device)

    N = p.shape[0]
    M = a_all.shape[0]
    best_closest = torch.empty((N, 3), dtype=torch.float32, device=device)
    best_dist = torch.full((N,), float('inf'), dtype=torch.float32, device=device)
    best_tri = torch.zeros((N,), dtype=torch.int64, device=device)

    p_start, p_end = pbar_range
    n_query_chunks = max(1, (N + _QUERY_CHUNK - 1) // _QUERY_CHUNK)

    for qci, qi in enumerate(range(0, N, _QUERY_CHUNK)):
        qp = p[qi:qi + _QUERY_CHUNK]
        qn = qp.shape[0]
        chunk_best_closest = torch.empty((qn, 3), dtype=torch.float32, device=device)
        chunk_best_dist = torch.full((qn,), float('inf'), dtype=torch.float32, device=device)
        chunk_best_tri = torch.zeros((qn,), dtype=torch.int64, device=device)

        for fi in range(0, M, _FACE_CHUNK):
            a_c = a_all[fi:fi + _FACE_CHUNK]
            b_c = b_all[fi:fi + _FACE_CHUNK]
            c_c = c_all[fi:fi + _FACE_CHUNK]
            cand = _closest_point_on_triangles(qp, a_c, b_c, c_c)  # (qn, fchunk, 3)
            d2 = ((cand - qp[:, None, :]) ** 2).sum(-1)  # (qn, fchunk)
            local_min, local_idx = d2.min(dim=1)
            better = local_min < chunk_best_dist
            chunk_best_dist = torch.where(better, local_min, chunk_best_dist)
            chunk_best_tri = torch.where(better, local_idx + fi, chunk_best_tri)
            sel = cand[torch.arange(qn, device=device), local_idx]
            chunk_best_closest = torch.where(better[:, None], sel, chunk_best_closest)

        best_closest[qi:qi + qn] = chunk_best_closest
        best_dist[qi:qi + qn] = chunk_best_dist
        best_tri[qi:qi + qn] = chunk_best_tri

        if pbar is not None:
            pbar.update_absolute(int(p_start + (p_end - p_start) * (qci + 1) / n_query_chunks))

    return best_closest.double().cpu().numpy(), best_tri.cpu().numpy()


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

        px, py, pos = rasterize_uv_mesh(uv_mesh, texture_size, pbar=pbar, pbar_range=(0, 15))
        log.info("Rasterized %d texels", len(px))

        n_src_faces = len(original_mesh.faces)
        log.info("Closest-point projection onto original_mesh (%d faces) on %s "
                  "(chunked %d texels x %d faces)...", n_src_faces, device, _QUERY_CHUNK, _FACE_CHUNK)
        closest, tri_ids = closest_point_gpu(
            pos, np.asarray(original_mesh.vertices), np.asarray(original_mesh.faces), device,
            pbar=pbar, pbar_range=(15, 95))

        src_faces = np.asarray(original_mesh.faces)[tri_ids]
        src_tris = np.asarray(original_mesh.vertices)[src_faces]
        bary = trimesh_module.triangles.points_to_barycentric(src_tris, closest)
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
