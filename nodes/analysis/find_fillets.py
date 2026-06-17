# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Find fillet / rounded-edge regions on a CAD mesh.

A fillet is the surface swept by a ball of fixed radius r rolling along an edge, so
every point on it has the SAME small rolling-ball radius and is DEVELOPABLE (one
principal curvature ~1/r across the round, the other ~0 along it). We detect exactly
that signature from the principal curvatures:

  - developable: |k_min| / |k_max| < developable_ratio   (cylindrical/conical, NOT a
    sphere/blob where the ratio ~1, and NOT a saddle)
  - small radius: min_radius < r = 1/|k_max| < max_radius (a flat face has r huge ->
    rejected; a big cylinder has r large -> rejected)
  - locally-constant radius across the strip (low relative spread of r)

then connect the qualifying faces into fillet STRIPS and report each strip's radius.
(This is the curvature/rolling-ball signature used by DeFillet, Jiang et al. SIGGRAPH
2025 -- their full method also uses a Voronoi/medial-axis radius estimate. Note: cleanly
separating a true small CYLINDER from a fillet is genuinely ambiguous from geometry alone
-- DeFillet leaves that to semantics too -- so a small isolated cylinder may read as a
fillet; the radius cap and the 'bounded by two larger patches' shape help but aren't
perfect.)

Outputs vertex fields: fillet (1 on fillets), fillet_radius (estimated r), fillet_id."""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _local_edge_length(mesh, n):
    ev = np.asarray(mesh.edges_unique)
    el = np.asarray(mesh.edges_unique_length, dtype=np.float64)
    loc = np.zeros(n); cnt = np.zeros(n)
    np.add.at(loc, ev[:, 0], el); np.add.at(loc, ev[:, 1], el)
    np.add.at(cnt, ev[:, 0], 1.0); np.add.at(cnt, ev[:, 1], 1.0)
    loc = loc / np.maximum(cnt, 1.0)
    loc[loc <= 0] = float(np.mean(el)) if len(el) else 1.0
    return loc, ev


class FindFilletsNode(io.ComfyNode):
    """Detect fillet / rounded-edge strips (developable, constant small radius)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFindFillets",
            display_name="Find Fillets",
            category="geompack/analysis",
            description=(
                "Detect FILLETS / rounded edges: developable strips of locally-constant small "
                "radius (the rolling-ball signature). Uses principal curvatures -- a fillet has "
                "one curvature ~1/r (across the round) and the other ~0 (along it), with r small "
                "and consistent. Outputs vertex fields fillet (1 on fillets), fillet_radius "
                "(estimated radius), fillet_id (strip). Caveat: a true small cylinder can read as "
                "a fillet -- separating them is semantically ambiguous from geometry alone."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("radius", default=5, min=1, max=12, step=1, tooltip=(
                    "k-ring neighborhood for the libigl quadric principal-curvature fit. Set near "
                    "the fillet width in edges. Larger = smoother/robust, blurs tiny fillets.")),
                io.Float.Input("max_radius_frac", default=0.02, min=0.002, max=0.5, step=0.002, display_mode="number", tooltip=(
                    "Max fillet radius as a FRACTION of the model's bounding-box diagonal -- the key "
                    "knob that separates fillets from the body. Surfaces whose radius of curvature "
                    "exceeds this are too big to be a fillet (flats, the main cylinder/barrel) and "
                    "are rejected; only tighter rounds (hole rims, edge rounds) survive. LOWER = "
                    "only tighter rounds. ~0.015-0.025 typical; 0.02 default.")),
                io.Float.Input("min_radius_frac", default=0.002, min=0.0001, max=0.2, step=0.0005, display_mode="number", tooltip=(
                    "Min fillet radius (fraction of bbox diagonal) -- rejects sub-this curvature "
                    "noise / sharp creases. Default 0.002.")),
                io.Float.Input("developable_ratio", default=0.4, min=0.02, max=1.0, step=0.02, tooltip=(
                    "A point counts as developable (cylindrical/conical = fillet-like) when "
                    "|k_min|/|k_max| < this. SMALLER = stricter (only near-perfect cylinders, "
                    "rejects spheres/blobs harder). ~0.3-0.5 typical.")),
                io.Float.Input("radius_consistency", default=0.5, min=0.05, max=5.0, step=0.05, tooltip=(
                    "Reject points whose rolling-ball radius differs too much from the local median "
                    "(relative): keep if |r - r_med|/r_med < this. SMALLER = require very constant "
                    "radius (true fillets). Default 0.5. Set high to disable.")),
                io.Int.Input("min_strip_faces", default=30, min=1, max=1000000, step=1, tooltip=(
                    "Drop fillet strips with fewer faces than this (de-speckle).")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh_with_fillets"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, radius=5, max_radius_frac=0.06, min_radius_frac=0.002,
                developable_ratio=0.4, radius_consistency=0.5, min_strip_faces=30):
        import igl
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components

        mesh = trimesh.copy()
        try:
            mesh.merge_vertices(); mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
        except Exception as e:
            log.debug("preclean skipped: %s", e)

        V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        n = len(V)
        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        rmax = max_radius_frac * diag
        rmin = min_radius_frac * diag

        out = igl.principal_curvature(V, F, int(max(1, radius)))
        k1 = np.asarray(out[2], dtype=np.float64)
        k2 = np.asarray(out[3], dtype=np.float64)
        kmax = np.maximum(np.abs(k1), np.abs(k2))
        kmin = np.minimum(np.abs(k1), np.abs(k2))
        r = 1.0 / np.maximum(kmax, 1e-12)                  # rolling-ball radius estimate
        ratio = kmin / np.maximum(kmax, 1e-12)

        cand = (r < rmax) & (r > rmin) & (ratio < float(developable_ratio))

        # local-radius consistency: compare r to its neighborhood median (1-ring smoothed)
        loc, ev = _local_edge_length(mesh, n)
        rmed = r.copy()
        for _ in range(2):                                 # couple of median-ish smoothing passes
            acc = np.zeros(n); cnt = np.zeros(n)
            np.add.at(acc, ev[:, 0], r[ev[:, 1]]); np.add.at(acc, ev[:, 1], r[ev[:, 0]])
            np.add.at(cnt, ev[:, 0], 1.0); np.add.at(cnt, ev[:, 1], 1.0)
            rmed = acc / np.maximum(cnt, 1.0)
        consistent = np.abs(r - rmed) / np.maximum(rmed, 1e-12) < float(radius_consistency)
        cand = cand & consistent

        # connect candidate vertices into strips (subgraph of candidate-candidate edges)
        both = cand[ev[:, 0]] & cand[ev[:, 1]]
        e = ev[both]
        rows = np.concatenate([e[:, 0], e[:, 1]]); cols = np.concatenate([e[:, 1], e[:, 0]])
        A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
        ncomp, comp = connected_components(A, directed=False)
        comp_size = np.bincount(comp)
        keep = cand & (comp_size[comp] >= int(min_strip_faces))

        fillet_id = np.zeros(n, dtype=np.int64)
        uids = np.unique(comp[keep])
        remap = {c: i + 1 for i, c in enumerate(uids)}
        for v in np.where(keep)[0]:
            fillet_id[v] = remap[comp[v]]
        n_strips = len(uids)

        mesh.vertex_attributes["fillet"] = keep.astype(np.float32)
        mesh.vertex_attributes["fillet_radius"] = np.where(keep, r, 0.0).astype(np.float32)
        mesh.vertex_attributes["fillet_id"] = fillet_id.astype(np.float32)
        mesh.vertex_attributes["kappa_max"] = kmax.astype(np.float32)

        radii = [float(np.median(r[fillet_id == i + 1])) for i in range(n_strips)]
        info = (
            f"Find Fillets:\n\n"
            f"Vertices: {n:,} | fillet verts: {int(keep.sum()):,} ({100*keep.mean():.1f}%)\n"
            f"fillet strips: {n_strips}\n"
            f"bbox diag {diag:.3f} | radius window [{rmin:.4f}, {rmax:.4f}] | developable<{developable_ratio}\n"
        )
        if radii:
            rr = np.array(radii)
            info += (f"strip radii (bbox-frac): min {rr.min()/diag:.4f}, median {np.median(rr)/diag:.4f}, "
                     f"max {rr.max()/diag:.4f}\n")
        info += "\nFields: fillet (1 on fillets), fillet_radius, fillet_id, kappa_max"
        log.info("Find Fillets: %d strips, %d fillet verts", n_strips, int(keep.sum()))
        return io.NodeOutput(mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFindFillets": FindFilletsNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFindFillets": "Find Fillets"}
