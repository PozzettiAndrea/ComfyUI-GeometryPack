# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Variational Shape Approximation (VSA) segmentation -> a field of patches.

Stage 1 of a Mesh2Brep-style pipeline (Shen et al., TVCG 2025): partition the mesh
into connected patches by Lloyd iteration on shape proxies (Cohen-Steiner et al.
2004; Yan-Liu-Wang Quadric VSA 2012). Here the proxy is a plane and the metric is
the area-weighted L2,1 (normal-deviation) distance plus a small spatial term for
coherence:

  assign:  each face -> proxy minimizing  area * ||n_f - N_proxy||^2 + w_pos * ||c_f - C_proxy||^2
  fit:     proxy normal = area-weighted mean normal of its faces; proxy point = centroid
  (Lloyd / k-means in shape space, GPU-vectorized)

then connected-component split (k-means labels can be spatially disconnected) and a
de-speckle pass. This INTENTIONALLY over-segments curved surfaces into normal-coherent
strips -- the downstream "Fit Primitives" node merges those into single cylinders/
cones/spheres by primitive fit. Output: per-face `patch_id` (1..R).

Input MUST be watertight + manifold (a solid); otherwise the node errors -- the B-rep
pipeline assumes a valid solid."""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _vsa_gpu(N, A, C, indptr, indices, adj, num_patches, iterations, w_pos, seed=0):
    """GPU Lloyd VSA. N face normals, A areas, C centroids (normalized), adjacency
    CSR (indptr,indices) + adj pairs. Returns per-face patch label (0..R-1)."""
    import torch
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nF = len(N)
    Nt = torch.tensor(N, dtype=torch.float32, device=dev)
    At = torch.tensor(A, dtype=torch.float32, device=dev)
    Ct = torch.tensor(C, dtype=torch.float32, device=dev)
    feat = torch.cat([Nt, w_pos * Ct], 1)

    # farthest-point seeds in (normal, weighted position) feature space
    g = torch.Generator(device=dev).manual_seed(seed)
    seeds = [int(torch.randint(nF, (1,), generator=g, device=dev).item())]
    d = torch.full((nF,), 1e30, device=dev)
    for _ in range(min(num_patches, nF) - 1):
        d = torch.minimum(d, ((feat - feat[seeds[-1]]) ** 2).sum(1))
        seeds.append(int(d.argmax()))
    seeds = torch.tensor(seeds, device=dev)
    pn = Nt[seeds].clone()
    pc = Ct[seeds].clone()

    lab = None
    for _ in range(int(iterations)):
        # assign: argmin over proxies of L2,1 normal term + spatial term
        dist = (2.0 - 2.0 * (Nt @ pn.T)) + w_pos * (
            (Ct * Ct).sum(1, keepdim=True) - 2.0 * Ct @ pc.T + (pc * pc).sum(1))
        lab = dist.argmin(1)
        # fit: area-weighted proxy normal + centroid
        k = pn.shape[0]
        nacc = torch.zeros(k, 3, device=dev)
        cacc = torch.zeros(k, 3, device=dev)
        wsum = torch.zeros(k, device=dev)
        nacc.index_add_(0, lab, Nt * At[:, None])
        cacc.index_add_(0, lab, Ct * At[:, None])
        wsum.index_add_(0, lab, At)
        good = wsum > 0
        pn[good] = nacc[good] / (nacc[good].norm(dim=1, keepdim=True) + 1e-12)
        pc[good] = cacc[good] / wsum[good, None]

    labels = lab.cpu().numpy()
    # connected-component split: a k-means label can cover disconnected regions
    same = labels[adj[:, 0]] == labels[adj[:, 1]]
    e = adj[same]
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    Asp = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nF, nF)).tocsr()
    nc, comp = connected_components(Asp, directed=False)
    return comp, str(dev)


class VSASegmentNode(io.ComfyNode):
    """Variational Shape Approximation -> per-face patch field (Mesh2Brep stage 1)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackVSASegment",
            display_name="Variational Shape Approximation",
            category="geompack/segmentation",
            description=(
                "Segment a SOLID mesh into connected patches by Variational Shape "
                "Approximation (Lloyd iteration on shape proxies) -- stage 1 of a Mesh2Brep "
                "pipeline. Outputs a per-face `patch_id` field. Over-segments curved surfaces "
                "into normal-coherent strips on purpose; the Fit Primitives node merges those "
                "into single primitives. Input must be WATERTIGHT + manifold or the node errors."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("num_patches", default=200, min=2, max=20000, step=1, tooltip=(
                    "Number of VSA seed proxies (Lloyd k). Set generously HIGH -- it's fine to "
                    "over-segment here; Fit Primitives merges strips back into primitives. "
                    "~150-400 for a typical CAD part.")),
                io.Int.Input("iterations", default=12, min=1, max=100, step=1, tooltip=(
                    "Lloyd iterations (assign<->fit). More = more settled proxies. ~10-15 is plenty.")),
                io.Float.Input("position_weight", default=0.4, min=0.0, max=5.0, step=0.05, tooltip=(
                    "Weight of the spatial term vs the normal term in the assignment metric. "
                    "Higher = patches stay spatially compact (fewer disconnected labels); 0 = "
                    "pure normal clustering. Default 0.4.")),
                io.Int.Input("min_patch_faces", default=20, min=1, max=1000000, step=1, tooltip=(
                    "Patches with fewer faces than this are merged into their strongest neighbour "
                    "(de-speckle).")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="segmented_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, num_patches=200, iterations=12, position_weight=0.4,
                min_patch_faces=20):
        mesh = trimesh.copy()

        # --- require a valid solid (watertight + manifold) ---
        if not mesh.is_watertight:
            raise ValueError(
                "Variational Shape Approximation requires a WATERTIGHT mesh (the Mesh2Brep "
                "pipeline assumes a solid). Repair the mesh first (e.g. Mesh Repair).")
        if not mesh.is_winding_consistent:
            raise ValueError("Mesh is not winding-consistent (non-manifold orientation). "
                             "Fix orientation / repair before segmenting.")

        F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        nF = len(F)
        N = np.asarray(mesh.face_normals, dtype=np.float64)
        A = np.asarray(mesh.area_faces, dtype=np.float64)
        C = np.asarray(mesh.triangles_center, dtype=np.float64)
        # normalize centroids to a unit box so position_weight is scale-stable
        cmin, cmax = C.min(0), C.max(0)
        Cn = (C - cmin) / (np.maximum(cmax - cmin, 1e-12))

        adj = np.asarray(mesh.face_adjacency)
        if len(adj) == 0:
            raise ValueError("Mesh has no face adjacency (degenerate).")
        # CSR neighbours for despeckle
        src = np.concatenate([adj[:, 0], adj[:, 1]])
        dst = np.concatenate([adj[:, 1], adj[:, 0]])
        o = np.argsort(src, kind="stable")
        src, dst = src[o], dst[o]
        ip = np.zeros(nF + 1, dtype=np.int64)
        np.add.at(ip, src + 1, 1)
        np.cumsum(ip, out=ip)

        labels, dev = _vsa_gpu(N, A, C, ip, dst, adj, int(num_patches), int(iterations),
                               float(position_weight))

        # de-speckle: merge tiny patches into the neighbour they share most faces with
        for _ in range(20):
            uq, labels = np.unique(labels, return_inverse=True)
            cnt = np.bincount(labels)
            small = np.where(cnt < int(min_patch_faces))[0]
            if not len(small):
                break
            changed = False
            for r in small:
                fs = np.where(labels == r)[0]
                nb = {}
                for f in fs:
                    for p in range(ip[f], ip[f + 1]):
                        L = labels[dst[p]]
                        if L != r:
                            nb[L] = nb.get(L, 0) + 1
                if nb:
                    labels[fs] = max(nb, key=nb.get)
                    changed = True
            if not changed:
                break
        uq, labels = np.unique(labels, return_inverse=True)
        n_patches = int(labels.max()) + 1

        patch_id = (labels + 1).astype(np.int64)
        mesh.face_attributes["patch_id"] = patch_id
        # vertex splat (majority) for vertex-field viewers
        V = np.asarray(mesh.vertices)
        vacc = np.zeros(len(V), dtype=np.float32)
        vcnt = np.zeros(len(V))
        for k in range(3):
            np.add.at(vacc, F[:, k], patch_id.astype(np.float32))
            np.add.at(vcnt, F[:, k], 1.0)
        mesh.vertex_attributes["patch_id"] = (vacc / np.maximum(vcnt, 1.0)).astype(np.float32)

        sizes = np.bincount(labels)
        info = (
            f"Variational Shape Approximation (device={dev}):\n\n"
            f"Faces: {nF:,} | patches: {n_patches:,}\n"
            f"seeds: {num_patches} | iterations: {iterations} | position_weight: {position_weight}\n\n"
            f"patch face-count: min {int(sizes.min())}, median {int(np.median(sizes))}, "
            f"max {int(sizes.max())}\n"
            f"Field: face_attributes['patch_id'] (1..{n_patches}) -> feed to Fit Primitives"
        )
        log.info("VSA: %d patches (%s)", n_patches, dev)
        return io.NodeOutput(mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackVSASegment": VSASegmentNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackVSASegment": "Variational Shape Approximation"}
