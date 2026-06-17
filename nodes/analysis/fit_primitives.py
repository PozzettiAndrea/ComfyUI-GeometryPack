# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Fit Primitives + merge -> consolidated CAD patches (Mesh2Brep stage 2).

Takes the over-segmented VSA patch field and, per Mesh2Brep (Shen et al. 2025),
robustly refits each patch to a primitive and re-selects the TYPE by GEOMETRIC
residual (NOT algebraic -- algebraic residuals are biased and not comparable across
types, which the paper's own VSA labels get wrong). Supported primitives:

  plane, sphere, cylinder, cone   (torus -- the natural fit for curved-corner
  fillets -- is not yet included; straight fillets fit a cylinder, tapered ones a cone).

Robustness: each fit runs a light sigma-clip (drop > k*sigma residual points, refit).
Then patches are AGGLOMERATIVELY MERGED: adjacent patches whose COMBINED points still
fit a single primitive within `merge_tol` are merged (greedy, lowest combined residual
first). This is what collapses the VSA bands of a cylinder/cone/sphere back into ONE
patch. Output: consolidated face `patch_id`, `patch_type` (0 plane,1 sphere,2 cylinder,
3 cone), `fit_residual`, and per-patch primitive parameters in mesh.metadata."""

import logging
import heapq

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")

_TYPE = {"plane": 0, "sphere": 1, "cylinder": 2, "cone": 3}


def _fit_plane(P, Nrm=None):
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[-1]
    return "plane", {"point": c, "normal": n}, np.abs((P - c) @ n)


def _fit_sphere(P, Nrm=None):
    ps = np.linalg.norm(P.max(0) - P.min(0)) + 1e-12
    A = np.hstack([2.0 * P, np.ones((len(P), 1))])
    b = (P * P).sum(1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = x[:3]
    r = float(np.sqrt(max(x[3] + c @ c, 1e-18)))
    pr = {"center": c, "radius": r}
    if r > 8.0 * ps:                                     # huge sphere = really a plane/cylinder
        return "sphere", pr, np.full(len(P), 1e9)
    return "sphere", pr, np.abs(np.linalg.norm(P - c, axis=1) - r)


def _circle_in_plane(P, a):
    # orthonormal basis perpendicular to axis a
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, t); u /= np.linalg.norm(u) + 1e-12
    w = np.cross(a, u)
    x = P @ u; y = P @ w
    A = np.column_stack([2 * x, 2 * y, np.ones(len(P))])
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0], sol[1]
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-18)))
    center = cx * u + cy * w + float(np.mean(P @ a)) * a
    return center, r


def _fit_cylinder(P, Nrm):
    ps = np.linalg.norm(P.max(0) - P.min(0)) + 1e-12
    _, _, Vt = np.linalg.svd(Nrm, full_matrices=False)
    a = Vt[-1]; a /= np.linalg.norm(a) + 1e-12          # axis ~ least-aligned-with-normals dir
    c, r = _circle_in_plane(P, a)
    pr = {"axis": a, "point": c, "radius": r}
    if not (1e-4 * ps < r < 30.0 * ps):                 # reject plane-as-huge-cylinder / degenerate
        return "cylinder", pr, np.full(len(P), 1e9)
    d = np.linalg.norm((P - c) - np.outer((P - c) @ a, a), axis=1)
    return "cylinder", pr, np.abs(d - r)


def _fit_cone(P, Nrm):
    ps = np.linalg.norm(P.max(0) - P.min(0)) + 1e-12
    # apex from n.v = n.p (the normal is perpendicular to the ruling line p-v)
    v, *_ = np.linalg.lstsq(Nrm, (Nrm * P).sum(1), rcond=None)
    nc = Nrm - Nrm.mean(0)
    w, V = np.linalg.eigh(nc.T @ nc)
    a = V[:, 0]; a /= np.linalg.norm(a) + 1e-12          # axis = min-variance dir of normals
    half = float(np.arcsin(np.clip(abs(np.mean(Nrm @ a)), 0, 1)))
    pr = {"apex": v, "axis": a, "half_angle": half}
    # reject degenerate cones (near-plane / near-cylinder / runaway apex) that fit anything
    if not (0.09 < half < 1.48) or np.linalg.norm(v - P.mean(0)) > 50.0 * ps:
        return "cone", pr, np.full(len(P), 1e9)
    pv = P - v
    L = np.linalg.norm(pv, axis=1) + 1e-12
    ang = np.arccos(np.clip((pv @ a) / L, -1, 1))
    ang = np.minimum(ang, np.pi - ang)
    return "cone", pr, L * np.abs(np.sin(ang - half))


_FITTERS = [_fit_plane, _fit_sphere, _fit_cylinder, _fit_cone]


def _best_primitive(P, Nrm, kclip=2.5, do_clip=True):
    """Return (type, params, residual_array) of the best-fitting primitive by GEOMETRIC
    RMS residual. With do_clip, one sigma-clip refit per type (final fits only; skipped
    in the merge search for speed). Falls back to plane if tiny."""
    if len(P) < 8:
        return _fit_plane(P)
    best = None
    best_rms = None
    for fit in _FITTERS:
        try:
            t, pr, res = fit(P, Nrm)
            if do_clip:
                s = res.std() + 1e-12
                keep = res < kclip * s
                if keep.sum() >= 8 and keep.sum() < len(P):
                    t, pr, _ = fit(P[keep], Nrm[keep])
                    res = _residual(t, pr, P, Nrm)    # eval on all points
            rms = float(np.sqrt((res ** 2).mean()))
            if best_rms is None or rms < best_rms:
                best, best_rms = (t, pr, res), rms
        except Exception:
            continue
    return best


def _group_by_label(labels, n):
    """Faces grouped by label in O(nF log nF) (one argsort), not n*O(nF) np.where scans."""
    order = np.argsort(labels, kind="stable")
    bnd = np.searchsorted(labels[order], np.arange(n + 1))
    return [order[bnd[r]:bnd[r + 1]] for r in range(n)]


def _residual(t, pr, P, Nrm):
    if t == "plane":
        return np.abs((P - pr["point"]) @ pr["normal"])
    if t == "sphere":
        return np.abs(np.linalg.norm(P - pr["center"], axis=1) - pr["radius"])
    if t == "cylinder":
        a, c = pr["axis"], pr["point"]
        d = np.linalg.norm((P - c) - np.outer((P - c) @ a, a), axis=1)
        return np.abs(d - pr["radius"])
    a, v, half = pr["axis"], pr["apex"], pr["half_angle"]
    pv = P - v; L = np.linalg.norm(pv, axis=1) + 1e-12
    ang = np.arccos(np.clip((pv @ a) / L, -1, 1)); ang = np.minimum(ang, np.pi - ang)
    return L * np.abs(np.sin(ang - half))


class FitPrimitivesNode(io.ComfyNode):
    """Fit a primitive per patch + merge adjacent same-primitive patches (Mesh2Brep stage 2)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFitPrimitives",
            display_name="Fit Primitives",
            category="geompack/segmentation",
            description=(
                "Refit each VSA patch to a primitive (plane/sphere/cylinder/cone), select the "
                "TYPE by geometric residual, and AGGLOMERATIVELY MERGE adjacent patches whose "
                "combined points still fit one primitive within merge_tol -- collapsing the VSA "
                "bands of a cylinder/cone/sphere into a single patch. Needs a `patch_id` face "
                "field (from Variational Shape Approximation). Outputs consolidated patch_id, "
                "patch_type (0 plane,1 sphere,2 cylinder,3 cone), fit_residual, and per-patch "
                "params in metadata. (Torus not yet supported -- curved-corner fillets stay split.)"
            ),
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("merge_tol", default=0.004, min=0.0002, max=0.1, step=0.0002, display_mode="number", tooltip=(
                    "Merge two adjacent patches if their COMBINED best-fit primitive has RMS "
                    "residual below this fraction of the bbox diagonal. HIGHER = more merging "
                    "(bigger patches, risk merging distinct primitives); LOWER = conservative. "
                    "~0.003-0.006 typical.")),
                io.Int.Input("samples_per_patch", default=400, min=50, max=5000, step=50, tooltip=(
                    "Max face-centroid samples used per patch/region for fitting (speed cap). "
                    "Higher = more accurate fits, slower merge.")),
                io.Float.Input("sigma_clip", default=2.5, min=1.0, max=6.0, step=0.1, tooltip=(
                    "Robust sigma-clip k for each fit: points with residual > k*sigma are dropped "
                    "and the primitive refit once (removes blend/neighbour outliers). ~2.5.")),
                io.Float.Input("inlier_thresh", default=0.9, min=0.5, max=1.0, step=0.01, tooltip=(
                    "Merge two patches only if this FRACTION of their combined points fits a "
                    "single primitive within merge_tol. Higher = stricter (won't merge a region "
                    "where even a small part is a different primitive). ~0.9-0.95.")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="primitive_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, merge_tol=0.004, samples_per_patch=400, sigma_clip=2.5,
                inlier_thresh=0.9):
        mesh = trimesh.copy()
        fa = getattr(mesh, "face_attributes", {})
        if "patch_id" not in fa:
            raise ValueError("Fit Primitives needs a face `patch_id` field -- run Variational "
                             "Shape Approximation first.")
        F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        nF = len(F)
        C = np.asarray(mesh.triangles_center, dtype=np.float64)
        Nf = np.asarray(mesh.face_normals, dtype=np.float64)
        diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        tol = merge_tol * diag
        labels = np.asarray(fa["patch_id"], dtype=np.int64).copy()
        uq, labels = np.unique(labels, return_inverse=True)
        npatch0 = labels.max() + 1
        rng = np.random.default_rng(0)

        # per-patch sampled points + normals (capped)
        faces_of = _group_by_label(labels, npatch0)
        def sample(faces):
            if len(faces) > samples_per_patch:
                faces = rng.choice(faces, samples_per_patch, replace=False)
            return C[faces], Nf[faces]
        pts = {}; nrm = {}
        for r in range(npatch0):
            pts[r], nrm[r] = sample(faces_of[r])

        # patch adjacency
        adj = np.asarray(mesh.face_adjacency)
        pl, pr_ = labels[adj[:, 0]], labels[adj[:, 1]]
        mask = pl != pr_
        pairs = set(map(tuple, np.sort(np.vstack([pl[mask], pr_[mask]]).T, axis=1)))
        nbrs = {r: set() for r in range(npatch0)}
        for a, b in pairs:
            nbrs[a].add(b); nbrs[b].add(a)

        parent = np.arange(npatch0)
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        def merged_score(a, b):
            """(rms, inlier_fraction) of the combined patches' best single primitive."""
            P = np.vstack([pts[a], pts[b]]); Nr = np.vstack([nrm[a], nrm[b]])
            if len(P) > samples_per_patch:
                idx = rng.choice(len(P), samples_per_patch, replace=False); P, Nr = P[idx], Nr[idx]
            res = _best_primitive(P, Nr, sigma_clip, do_clip=False)   # no clip in merge search (speed)
            if res is None:
                return 1e30, 0.0
            r = res[2]
            return float(np.sqrt((r ** 2).mean())), float(np.mean(r < tol))

        heap = []
        for a, b in pairs:
            rms, frac = merged_score(a, b)
            if frac >= inlier_thresh:                  # only merge if ~all points fit ONE primitive
                heapq.heappush(heap, (rms, a, b))

        # greedy agglomerative merge (gate on inlier fraction, re-verify on pop)
        while heap:
            rms, a, b = heapq.heappop(heap)
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            rms2, frac2 = merged_score(ra, rb)         # regions changed -> re-verify
            if frac2 < inlier_thresh:
                continue
            parent[rb] = ra
            P = np.vstack([pts[ra], pts[rb]]); Nr = np.vstack([nrm[ra], nrm[rb]])
            if len(P) > samples_per_patch:
                idx = rng.choice(len(P), samples_per_patch, replace=False); P, Nr = P[idx], Nr[idx]
            pts[ra], nrm[ra] = P, Nr
            nbrs[ra] |= nbrs[rb]
            for x in list(nbrs[ra]):
                rx = find(x)
                if rx != ra:
                    rms3, frac3 = merged_score(ra, rx)
                    if frac3 >= inlier_thresh:
                        heapq.heappush(heap, (rms3, ra, rx))

        roots = np.array([find(r) for r in range(npatch0)])
        face_root = roots[labels]
        uq, new_labels = np.unique(face_root, return_inverse=True)
        n_final = new_labels.max() + 1

        # final per-region primitive type/params on ALL region faces
        ptype = np.zeros(nF, dtype=np.int64)
        presid = np.zeros(nF, dtype=np.float32)
        params = {}
        type_count = {k: 0 for k in _TYPE}
        groups_final = _group_by_label(new_labels, n_final)
        for r in range(n_final):
            fr = groups_final[r]
            P, Nr = sample(fr)
            res = _best_primitive(P, Nr, sigma_clip)
            if res is None:
                continue
            t, pr_p, _resarr = res
            ptype[fr] = _TYPE[t]
            presid[fr] = _residual(t, pr_p, C[fr], Nf[fr]).astype(np.float32)
            type_count[t] += 1
            params[int(r + 1)] = {"type": t, **{k: (v.tolist() if hasattr(v, "tolist") else v)
                                                 for k, v in pr_p.items()}}

        patch_id = (new_labels + 1).astype(np.int64)
        mesh.face_attributes["patch_id"] = patch_id
        mesh.face_attributes["patch_type"] = ptype
        mesh.face_attributes["fit_residual"] = presid
        mesh.metadata = (mesh.metadata.copy() if mesh.metadata else {})
        mesh.metadata["primitives"] = params

        info = (
            f"Fit Primitives:\n\n"
            f"patches: {npatch0:,} -> {n_final:,} (merged)\n"
            f"types: plane {type_count['plane']}, sphere {type_count['sphere']}, "
            f"cylinder {type_count['cylinder']}, cone {type_count['cone']}\n"
            f"merge_tol: {merge_tol} ({tol:.4f}) | mean fit residual: {presid.mean():.5f}\n\n"
            f"Fields: patch_id, patch_type (0 plane/1 sphere/2 cyl/3 cone), fit_residual; "
            f"params in metadata['primitives']"
        )
        log.info("Fit Primitives: %d -> %d patches | types %s", npatch0, n_final, type_count)
        return io.NodeOutput(mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFitPrimitives": FitPrimitivesNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFitPrimitives": "Fit Primitives"}
