# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Detect TANGENT / FILLET boundary edges -- the line where a flat region rolls
into a fillet/round.

That seam is a SECOND-ORDER (G2) feature: the surface is tangent-continuous there
(normals match across it), so the dihedral angle is ~0 and a normal-jump / crease
detector misses it entirely. What changes is the CURVATURE: ~0 on the plane,
~1/r on the fillet. So we detect the curvature STEP:

  1. principal curvatures by quadric fit (libigl, k-ring `radius`) -> kappa_max =
     max(|k1|,|k2|). (Max-principal localises the across-fillet step better than mean H.)
  2. RESOLUTION-AWARE flat test: a vertex is 'flat' iff kappa_max * local_edge <
     1/flat_radius_edges (radius of curvature bigger than that many edges) -- so fine
     tessellation noise isn't called a feature.
  3. method:
       - region_boundary : de-speckle the flat/curved regions, then mark the edges
         that straddle the flat<->curved boundary = the tangent line.
       - curvature_jump  : mark edges whose kappa_max*edge STEP exceeds jump_threshold
         (more direct, no region clean-up).

Outputs vertex fields kappa_max, flat_region (1 flat / 0 curved) and tangent_edge
(1 on the boundary). Colour tangent_edge, or feed flat_region to Split By Field."""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _local_edge_length(mesh, n):
    ev = np.asarray(mesh.edges_unique)
    el = np.asarray(mesh.edges_unique_length, dtype=np.float64)
    loc = np.zeros(n)
    cnt = np.zeros(n)
    np.add.at(loc, ev[:, 0], el)
    np.add.at(loc, ev[:, 1], el)
    np.add.at(cnt, ev[:, 0], 1.0)
    np.add.at(cnt, ev[:, 1], 1.0)
    loc = loc / np.maximum(cnt, 1.0)
    loc[loc <= 0] = float(np.mean(el)) if len(el) else 1.0
    return loc, ev


def _despeckle(region, ev, n, min_size):
    """Relabel connected same-region components smaller than min_size to their
    surrounding region (flip the label of tiny islands). region is a 0/1 array."""
    if min_size <= 1:
        return region
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    same = region[ev[:, 0]] == region[ev[:, 1]]
    e = ev[same]
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    ncomp, comp = connected_components(A, directed=False)
    counts = np.bincount(comp, minlength=ncomp)
    small = counts[comp] < int(min_size)
    out = region.copy()
    out[small] = 1 - out[small]            # flip tiny islands to the other class
    return out


class DetectTangentEdgesNode(io.ComfyNode):
    """Detect flat<->fillet tangent boundary edges (a G2 / curvature-step feature)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackDetectTangentEdges",
            display_name="Detect Tangent / Fillet Edges",
            category="geompack/analysis",
            description=(
                "Find the tangent line where a FLAT region meets a FILLET/round. That seam is "
                "tangent-continuous (normals match, dihedral ~0) so a crease/dihedral detector "
                "misses it -- it is a CURVATURE step (flat ~0 -> fillet ~1/r), not a normal jump. "
                "This thresholds the max principal curvature (resolution-aware) and marks the "
                "flat<->curved boundary. Outputs vertex fields kappa_max, flat_region, "
                "tangent_edge (1 on the seam)."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Combo.Input("method", options=["region_boundary", "curvature_jump"],
                               default="region_boundary", tooltip=(
                    "region_boundary = classify flat vs curved (resolution-aware), de-speckle, "
                    "mark the boundary between them (clean tangent lines). curvature_jump = mark "
                    "edges whose curvature STEP exceeds jump_threshold (direct, no clean-up).")),
                io.Int.Input("radius", default=5, min=1, max=12, step=1, tooltip=(
                    "k-ring neighborhood for the libigl quadric principal-curvature fit. Larger "
                    "= smoother / noise-robust (set near the fillet radius in edges); too large "
                    "blurs the seam. ~3 crisp, ~5 default.")),
                io.Float.Input("flat_radius_edges", default=8.0, min=1.0, max=100.0, step=1.0, tooltip=(
                    "(region_boundary) A vertex counts as FLAT when its curvature radius exceeds "
                    "this many LOCAL edge lengths (kappa_max*edge < 1/this). HIGHER = stricter "
                    "'flat' (more of the fillet onset counts as curved -> seam sits earlier). "
                    "~5-15 typical.")),
                io.Float.Input("jump_threshold", default=0.15, min=0.001, max=2.0, step=0.005, display_mode="number", tooltip=(
                    "(curvature_jump) Mark an edge when the dimensionless curvature step "
                    "|kappa_max_i - kappa_max_j| * local_edge across it exceeds this. LOWER = "
                    "more sensitive (more edges). ~0.1-0.3 typical.")),
                io.Int.Input("min_region_size", default=20, min=1, max=1000000, step=1, tooltip=(
                    "(region_boundary) Flat/curved islands smaller than this many vertices are "
                    "flipped into their surroundings before the boundary is taken (de-speckle).")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh_with_edges"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, method="region_boundary", radius=5, flat_radius_edges=8.0,
                jump_threshold=0.15, min_region_size=20):
        import igl

        mesh = trimesh.copy()
        try:
            mesh.merge_vertices()
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
        except Exception as e:
            log.debug("preclean skipped: %s", e)

        V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        n = len(V)

        out = igl.principal_curvature(V, F, int(max(1, radius)))
        k1 = np.asarray(out[2], dtype=np.float64)
        k2 = np.asarray(out[3], dtype=np.float64)
        kmax = np.maximum(np.abs(k1), np.abs(k2))          # max principal curvature

        loc, ev = _local_edge_length(mesh, n)
        score = kmax * loc                                  # dimensionless curvature-per-edge

        if method == "curvature_jump":
            d = np.abs(score[ev[:, 0]] - score[ev[:, 1]])
            feat_e = ev[d > float(jump_threshold)]
            flat_region = (score < (1.0 / float(max(1.0, flat_radius_edges)))).astype(np.int64)
        else:  # region_boundary
            tau = 1.0 / float(max(1.0, flat_radius_edges))
            flat_region = (score < tau).astype(np.int64)
            flat_region = _despeckle(flat_region, ev, n, int(min_region_size))
            cross = flat_region[ev[:, 0]] != flat_region[ev[:, 1]]
            feat_e = ev[cross]

        tangent = np.zeros(n, dtype=np.float32)
        if len(feat_e):
            tangent[feat_e.ravel()] = 1.0

        mesh.vertex_attributes["kappa_max"] = kmax.astype(np.float32)
        mesh.vertex_attributes["flat_region"] = flat_region.astype(np.float32)
        mesh.vertex_attributes["tangent_edge"] = tangent

        # validation: dihedral at the detected seam SHOULD be low (it's G2, not G1)
        fa = np.asarray(mesh.face_adjacency)
        ang = np.degrees(np.asarray(mesh.face_adjacency_angles))
        # an adjacency edge is "on the seam" if both its shared verts are tangent
        fae = np.asarray(mesh.face_adjacency_edges)
        on_seam = (tangent[fae[:, 0]] > 0) & (tangent[fae[:, 1]] > 0)
        seam_dih = float(np.median(ang[on_seam])) if on_seam.any() else float("nan")
        sharp_dih = float(np.percentile(ang, 99))

        n_feat = int((tangent > 0).sum())
        info = (
            f"Detect Tangent / Fillet Edges ({method}):\n\n"
            f"Vertices: {n:,} | tangent-edge verts: {n_feat:,} ({100*n_feat/max(1,n):.1f}%)\n"
            f"flat: {100*float((flat_region==1).mean()):.1f}% | curved: {100*float((flat_region==0).mean()):.1f}%\n\n"
            f"kappa_max: median {np.median(kmax):.2f}, p99 {np.percentile(kmax,99):.2f}\n"
            f"median dihedral AT the seam: {seam_dih:.2f} deg  (vs sharp-edge p99 {sharp_dih:.2f} deg)\n"
            f"  -> low seam-dihedral confirms these are TANGENT (G2) lines a crease detector misses.\n\n"
            f"Fields: tangent_edge (1 on seam), flat_region (1 flat/0 curved), kappa_max"
        )
        log.info("Detect Tangent Edges: %d seam verts (%s), median seam dihedral %.2f deg",
                 n_feat, method, seam_dih)
        return io.NodeOutput(mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackDetectTangentEdges": DetectTangentEdgesNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackDetectTangentEdges": "Detect Tangent / Fillet Edges"}
