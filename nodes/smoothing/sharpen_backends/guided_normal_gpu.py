# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""GPU/vectorized guided mesh normal filtering (Zhang et al. 2015).

Faithful torch port of the CPU `guided_normal` backend: same min-range-metric
guidance selection, same bilateral normal filter, same interleaved vertex
update -- but the per-iteration math runs as batched scatter/gather ops (CUDA
when available, else vectorized CPU torch) instead of Python loops over every
face and neighborhood.

The mesh topology (vertex-based 1-ring face patches + their inner adjacency
edges) is fixed across iterations, so it is built ONCE on the CPU and reused as
padded (m, K) / (m, Emax) index tensors. Everything inside the iteration loop is
GPU work.
"""

import logging

import numpy as np
import trimesh as trimesh_module

from ._helpers import (
    _build_vertex_to_faces,
    _build_vertex_based_face_neighbors,
)

log = logging.getLogger("geometrypack")


def _torch_device():
    import torch
    try:
        import comfy.model_management as mm
        dev = mm.get_torch_device()
        if dev is not None:
            return dev
    except Exception:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pad_patches(neighbors, m):
    """neighbors: list of sorted face-index lists (includes central face).
    Returns padded P (m, K) int64 (pad=0) and validity mask M (m, K) bool."""
    K = max(1, max((len(nb) for nb in neighbors), default=1))
    P = np.zeros((m, K), dtype=np.int64)
    M = np.zeros((m, K), dtype=bool)
    for fi, nb in enumerate(neighbors):
        k = len(nb)
        if k:
            P[fi, :k] = nb
            M[fi, :k] = True
    return P, M, K


def _pad_inner_edges(neighbors, adj_pairs, face_to_adj, m):
    """Per face, the indices of adjacency edges whose BOTH faces lie inside the
    face's patch -- mirrors the CPU seen_edges/patch_set logic exactly. Returns
    padded IE (m, Emax) int64 (pad=0) and validity mask IEM (m, Emax) bool."""
    inner = []
    Emax = 1
    for fi in range(m):
        patch_set = set(neighbors[fi])
        seen = set()
        ies = []
        for pf in neighbors[fi]:
            for ai in face_to_adj[pf]:
                if ai in seen:
                    continue
                seen.add(ai)
                fa, fb = adj_pairs[ai]
                if fa in patch_set and fb in patch_set:
                    ies.append(ai)
        inner.append(ies)
        if len(ies) > Emax:
            Emax = len(ies)
    IE = np.zeros((m, Emax), dtype=np.int64)
    IEM = np.zeros((m, Emax), dtype=bool)
    for fi, ies in enumerate(inner):
        k = len(ies)
        if k:
            IE[fi, :k] = ies
            IEM[fi, :k] = True
    return IE, IEM


def _guided_normal_gpu(mesh, normal_iterations, vertex_iterations, sigma_s, sigma_r):
    import torch

    V0 = np.asarray(mesh.vertices, dtype=np.float64)
    Ff = np.asarray(mesh.faces, dtype=np.int64)
    if len(V0) == 0 or len(Ff) == 0:
        return None, "Empty mesh (no vertices or faces).", "cpu"
    adj = np.asarray(mesh.face_adjacency)
    if len(adj) == 0:
        return None, "Mesh has no face adjacency (disconnected or degenerate).", "cpu"

    m = len(Ff)
    n = len(V0)

    # --- one-time topology (CPU), constant across iterations ---
    vtf = _build_vertex_to_faces(n, Ff)
    neighbors = _build_vertex_based_face_neighbors(Ff, vtf, include_central=True)
    face_to_adj = [[] for _ in range(m)]
    for ai in range(len(adj)):
        fa, fb = adj[ai]
        face_to_adj[fa].append(ai)
        face_to_adj[fb].append(ai)
    P_np, M_np, K = _pad_patches(neighbors, m)
    IE_np, IEM_np = _pad_inner_edges(neighbors, adj, face_to_adj, m)

    dev = _torch_device()
    eps = 1e-12
    V = torch.as_tensor(V0, dtype=torch.float32, device=dev)
    F = torch.as_tensor(Ff, dtype=torch.long, device=dev)
    adj_a = torch.as_tensor(adj[:, 0], dtype=torch.long, device=dev)
    adj_b = torch.as_tensor(adj[:, 1], dtype=torch.long, device=dev)
    P = torch.as_tensor(P_np, dtype=torch.long, device=dev)        # (m, K)
    Mf = torch.as_tensor(M_np, dtype=torch.float32, device=dev)    # (m, K)
    Mb = torch.as_tensor(M_np, device=dev)                         # (m, K) bool
    IE = torch.as_tensor(IE_np, dtype=torch.long, device=dev)      # (m, E)
    IEMf = torch.as_tensor(IEM_np, dtype=torch.float32, device=dev)
    IEMb = torch.as_tensor(IEM_np, device=dev)

    ones_f = torch.ones(m, device=dev)
    arange_m = torch.arange(m, device=dev)

    # chunk the O(K^2) max-pairwise-diff so (chunk * K * K) stays bounded
    chunk = max(1, min(m, 4_000_000 // max(1, K * K)))

    for _ in range(int(normal_iterations)):
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        cross = torch.cross(v1 - v0, v2 - v0, dim=1)
        area2 = cross.norm(dim=1, keepdim=True)
        normals = cross / (area2 + eps)                 # (m, 3)
        areas = area2.squeeze(1) * 0.5                  # (m,)
        centroids = (v0 + v1 + v2) / 3.0                # (m, 3)

        e_all = torch.cat([(v1 - v0).norm(dim=1),
                           (v2 - v1).norm(dim=1),
                           (v0 - v2).norm(dim=1)])
        avg_edge_len = e_all.mean()
        sigma_s_abs = sigma_s * avg_edge_len
        sigma_s_sq2 = 2.0 * sigma_s_abs * sigma_s_abs
        sigma_r_sq2 = 2.0 * sigma_r * sigma_r

        # --- guidance: area-weighted patch average normals ---
        Pn = normals[P]                                 # (m, K, 3)
        Pa = areas[P] * Mf                              # (m, K)
        avg_n = (Pn * Pa.unsqueeze(2)).sum(1)           # (m, 3)
        avg_normals = avg_n / (avg_n.norm(dim=1, keepdim=True) + eps)

        # max pairwise normal diff within each patch (chunked O(K^2))
        maxdiff = torch.zeros(m, device=dev)
        for s in range(0, m, chunk):
            e = min(m, s + chunk)
            pn = Pn[s:e]                                # (c, K, 3)
            mb = Mb[s:e]                                # (c, K)
            d = (pn.unsqueeze(2) - pn.unsqueeze(1)).norm(dim=3)   # (c, K, K)
            pair = mb.unsqueeze(2) & mb.unsqueeze(1)              # (c, K, K)
            d = torch.where(pair, d, torch.zeros_like(d))
            maxdiff[s:e] = d.flatten(1).amax(dim=1)

        # inner-edge total variation over each patch
        tv = (normals[adj_a] - normals[adj_b]).norm(dim=1)       # (E,)
        tv_patch = tv[IE]                                        # (m, Emax)
        sum_tv = (tv_patch * IEMf).sum(1)
        max_tv = torch.where(IEMb, tv_patch, torch.zeros_like(tv_patch)).amax(dim=1)

        metrics = maxdiff * max_tv / (sum_tv + 1e-9)             # (m,)

        # guided normal = avg normal of the min-metric neighbor in the patch
        pm = torch.where(Mb, metrics[P], torch.full_like(metrics[P], 1e30))
        chosen = P[arange_m, pm.argmin(dim=1)]                   # (m,)
        guided_normals = avg_normals[chosen]                    # (m, 3)

        # --- bilateral normal filter over each patch ---
        dist_sq = ((centroids.unsqueeze(1) - centroids[P]) ** 2).sum(2)   # (m, K)
        ws = torch.exp(-dist_sq / (sigma_s_sq2 + eps))
        gdiff_sq = ((guided_normals.unsqueeze(1) - guided_normals[P]) ** 2).sum(2)
        wr = torch.exp(-gdiff_sq / (sigma_r_sq2 + eps))
        w = areas[P] * ws * wr * Mf                             # (m, K)
        n_acc = (normals[P] * w.unsqueeze(2)).sum(1)            # (m, 3)
        w_total = w.sum(1, keepdim=True)
        filtered = torch.where(w_total > eps, n_acc / (w_total + eps), normals)
        filtered_normals = filtered / (filtered.norm(dim=1, keepdim=True) + eps)

        # --- interleaved vertex update to match the filtered normals ---
        for _ in range(int(vertex_iterations)):
            cen = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0
            new_V = torch.zeros_like(V)
            counts = torch.zeros(n, device=dev)
            for k in range(3):
                vid = F[:, k]
                Vv = V[vid]
                d = ((Vv - cen) * filtered_normals).sum(1, keepdim=True)
                new_V.index_add_(0, vid, Vv - d * filtered_normals)
                counts.index_add_(0, vid, ones_f)
            moved = counts > 0
            new_V[moved] /= counts[moved].unsqueeze(1)
            new_V[~moved] = V[~moved]
            V = new_V

    Vout = V.detach().cpu().numpy().astype(np.float64)
    result = trimesh_module.Trimesh(vertices=Vout,
                                    faces=np.asarray(mesh.faces, dtype=np.int32),
                                    process=False)
    return result, "", str(dev)
