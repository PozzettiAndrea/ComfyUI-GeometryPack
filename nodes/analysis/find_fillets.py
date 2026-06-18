# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Find fillet / rounded-edge regions via the DeFillet method (Jiang et al.,
SIGGRAPH 2025) -- reimplemented clean-room from the paper.

Key idea: a fillet is swept by a rolling ball, so its surface samples all "look at"
a shared 1D trajectory of ball centers. In the VORONOI diagram of the surface
samples, Voronoi vertices CONCENTRATE (high density) along that trajectory and are
robust rolling-ball-center candidates (equidistant to 4 samples -> a robust radius,
not a noisy curvature). Pipeline:

  1. samples = triangle centroids (subsampled for scale), normalized to a unit sphere.
  2. 3D Voronoi via GPU Delaunay (pygDel3D) -> Voronoi vertices = tet circumcenters,
     each with a rolling-ball radius r = circumradius.
  3. density rho(v) of Voronoi vertices (mean-shift estimate, paper eq. 4).
  4. RTT: each sample's candidate centers are the circumcenters of its INCIDENT
     Delaunay tets (the exact dual -- |sample - center| = r by construction, so no
     epsilon shell needed); pick the max-density one -> the sample's fillet radius.
  5. fillet radius variation rate R (eq. 6): a fillet has near-CONSTANT radius -> R~0.
  6. threshold (1-R) + connected strips -> fillet regions, transferred to all faces.

Requires the `pygdel3d` package (GPU Delaunay, BSD-3). Outputs vertex fields
fillet / fillet_radius / fillet_id."""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _defillet(C, F, adj, num_samples, density_k, density_sigma, variation_thresh, rng_seed=0):
    """Core DeFillet detection. C = per-face centroids (M,3), F faces, adj face-adjacency.
    Returns per-face fillet bool + per-face fillet radius (nan where not a fillet)."""
    import pygdel3d
    from scipy.spatial import cKDTree

    M = len(C)
    # normalize samples to a unit sphere (the paper normalizes; makes sigma/scale stable)
    center = 0.5 * (C.max(0) + C.min(0))
    scale = float(np.linalg.norm(C - center, axis=1).max()) + 1e-12
    Cn = (C - center) / scale

    # subsample for tractable exact Voronoi
    if M > num_samples:
        sidx = np.sort(np.random.default_rng(rng_seed).choice(M, num_samples, replace=False))
    else:
        sidx = np.arange(M)
    S = np.ascontiguousarray(Cn[sidx])
    ns = len(S)

    # --- Voronoi via GPU Delaunay ---
    vv, tets, rvor = pygdel3d.voronoi_vertices(S)
    fin = ~np.isnan(vv).any(1) & np.isfinite(rvor)
    tets, vv, rvor = tets[fin], vv[fin], rvor[fin]

    # --- density rho(v_i) of Voronoi vertices (mean-shift, eq. 4) ---
    tree_vv = cKDTree(vv)
    k = min(density_k + 1, len(vv))
    dist, _ = tree_vv.query(vv, k=k, workers=-1)
    d = dist[:, 1:]                                  # drop self
    sig2 = 2.0 * (density_sigma ** 2)
    w = np.exp(-(d ** 2) / sig2)
    rho = w.sum(1) / ((d * w).sum(1) + 1e-12)

    # --- RTT: per sample -> max-density incident circumcenter -> fillet radius ---
    vert = tets.ravel()
    rho_rep = np.repeat(rho, 4)
    r_rep = np.repeat(rvor, 4)
    best = np.full(ns, -np.inf)
    np.maximum.at(best, vert, rho_rep)
    rj = np.full(ns, np.nan)
    is_best = rho_rep >= best[vert] - 1e-12
    rj[vert[is_best]] = r_rep[is_best]              # radius of the densest incident center

    # --- fillet radius variation rate R(s_i) over nearest sample neighbors (eq. 6) ---
    tree_s = cKDTree(S)
    kk = min(4, ns)
    nd, ni = tree_s.query(S, k=kk, workers=-1)
    nb = ni[:, 1:kk]                                  # (ns, kk-1) neighbor indices
    dnb = nd[:, 1:kk]
    ri = rj[:, None]
    rb = rj[nb]
    valid = np.isfinite(ri) & np.isfinite(rb) & (dnb > 1e-9)
    ratio = np.where(valid, np.abs(ri - rb) / np.where(dnb > 1e-9, dnb, 1.0), 0.0)
    cnt = valid.sum(1)
    R = np.where(cnt > 0, ratio.sum(1) / np.maximum(cnt, 1), np.inf)
    R = np.minimum(1.0, R)
    fillet_sample = np.isfinite(rj) & (cnt > 0) & (R < float(variation_thresh))

    # --- transfer per-sample result to ALL faces (nearest sample) ---
    _, nearest = tree_s.query(Cn, k=1, workers=-1)
    face_fillet = fillet_sample[nearest]
    face_radius = np.where(face_fillet, rj[nearest] * scale, np.nan)   # de-normalize radius
    return face_fillet, face_radius


class FindFilletsNode(io.ComfyNode):
    """Detect fillet / rounded-edge strips via the DeFillet Voronoi rolling-ball method."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFindFillets",
            display_name="Find Fillets",
            category="geompack/analysis",
            description=(
                "Detect FILLETS / rounded edges with the DeFillet method (Jiang et al., SIGGRAPH "
                "2025): the Voronoi diagram of surface samples concentrates Voronoi vertices along "
                "each fillet's rolling-ball-center trajectory; a sample is a fillet point when its "
                "rolling-ball radius (from the densest incident Voronoi vertex) is near-CONSTANT "
                "across the strip. More robust than curvature (uses global Voronoi structure, not "
                "noisy 2nd derivatives). Requires the pygdel3d GPU-Delaunay package. Outputs vertex "
                "fields fillet / fillet_radius / fillet_id."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("num_samples", default=60000, min=2000, max=400000, step=1000, tooltip=(
                    "Surface samples (triangle centroids) used for the exact Voronoi. Subsampled "
                    "for tractability; DeFillet is resolution-robust so 40-100k is plenty even for "
                    "1M-face meshes. Higher = sharper but slower Voronoi.")),
                io.Float.Input("variation_thresh", default=0.15, min=0.005, max=1.0, step=0.005, display_mode="number", tooltip=(
                    "A sample is a fillet when its rolling-ball-radius VARIATION RATE (change in r "
                    "per unit surface distance to its neighbors) is below this. SMALLER = stricter "
                    "constant-radius requirement (only clean fillets). ~0.1-0.2 typical.")),
                io.Int.Input("density_k", default=16, min=4, max=128, step=1, tooltip=(
                    "k nearest Voronoi vertices used to estimate vertex DENSITY (paper eq. 4). "
                    "Higher = smoother density. ~16 default.")),
                io.Float.Input("density_sigma", default=0.03, min=0.002, max=0.5, step=0.002, display_mode="number", tooltip=(
                    "Bandwidth of the Voronoi-vertex density kernel, in UNIT-SPHERE-normalized "
                    "units (the mesh is normalized to a unit sphere first). SMALLER = more local "
                    "density. ~0.02-0.05 typical.")),
                io.Int.Input("min_strip_faces", default=30, min=1, max=1000000, step=1, tooltip=(
                    "Drop fillet strips with fewer faces than this (de-speckle).")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh_with_fillets"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, num_samples=60000, variation_thresh=0.15, density_k=16,
                density_sigma=0.03, min_strip_faces=30):
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
        try:
            import pygdel3d  # noqa: F401
        except ImportError:
            raise ImportError(
                "Find Fillets (DeFillet) needs the 'pygdel3d' package (GPU Delaunay). "
                "Install it into this env: pip install <path-to>/pygDel3D")

        mesh = trimesh.copy()
        try:
            mesh.merge_vertices(); mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
        except Exception as e:
            log.debug("preclean skipped: %s", e)

        V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        C = np.ascontiguousarray(mesh.triangles_center, dtype=np.float64)
        adj = np.asarray(mesh.face_adjacency)
        if len(adj) == 0:
            raise ValueError("Mesh has no face adjacency (disconnected or degenerate).")

        face_fillet, face_radius = _defillet(
            C, F, adj, int(num_samples), int(density_k), float(density_sigma),
            float(variation_thresh))

        # connected fillet strips on the face graph + despeckle
        both = face_fillet[adj[:, 0]] & face_fillet[adj[:, 1]]
        e = adj[both]
        m = len(F)
        A = sp.coo_matrix((np.ones(len(e) * 2),
                           (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
                          shape=(m, m)).tocsr()
        ncomp, comp = connected_components(A, directed=False)
        csz = np.bincount(comp, minlength=ncomp)
        keep = face_fillet & (csz[comp] >= int(min_strip_faces))

        fid = np.zeros(m, dtype=np.int64)
        uids = np.unique(comp[keep])
        remap = {c: i + 1 for i, c in enumerate(uids)}
        for f in np.where(keep)[0]:
            fid[f] = remap[comp[f]]
        n_strips = len(uids)

        # face fields -> also splat to vertices for viewers that read vertex fields
        mesh.face_attributes["fillet"] = keep.astype(np.int64)
        mesh.face_attributes["fillet_id"] = fid
        mesh.face_attributes["fillet_radius"] = np.where(keep, face_radius, 0.0).astype(np.float32)
        vfill = np.zeros(len(V), dtype=np.float32); vcnt = np.zeros(len(V))
        for k in range(3):
            np.add.at(vfill, F[:, k], keep.astype(np.float32)); np.add.at(vcnt, F[:, k], 1.0)
        mesh.vertex_attributes["fillet"] = (vfill / np.maximum(vcnt, 1.0) >= 0.5).astype(np.float32)

        rr = face_radius[keep]
        info = (
            f"Find Fillets (DeFillet / Voronoi rolling-ball):\n\n"
            f"Faces: {m:,} | fillet faces: {int(keep.sum()):,} ({100*keep.mean():.1f}%)\n"
            f"fillet strips: {n_strips}\n"
            f"samples: {min(num_samples, m):,} | variation_thresh: {variation_thresh}\n"
        )
        if len(rr):
            info += f"fillet radius: min {np.nanmin(rr):.4f}, median {np.nanmedian(rr):.4f}, max {np.nanmax(rr):.4f}\n"
        info += "\nFields: fillet, fillet_id, fillet_radius (per-face); fillet (per-vertex)"
        log.info("Find Fillets (DeFillet): %d strips, %d fillet faces", n_strips, int(keep.sum()))
        return io.NodeOutput(mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFindFillets": FindFilletsNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFindFillets": "Find Fillets"}
