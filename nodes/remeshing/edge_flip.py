# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Standalone edge-flip / Delaunay-flip node (libigl edge_flaps + vectorized flips).

Flips interior edges (swaps the shared diagonal of two triangles) to improve the
triangulation, keeping the vertex set fixed -- an EXTRINSIC flip, so the output is
a normal renderable mesh (unlike igl/geometry-central intrinsic Delaunay, which
flips geodesically and keeps F unchanged).

Two criteria:
  - delaunay: flip if the empirical Delaunay condition is violated, i.e. the two
    angles opposite the edge sum to > 180 deg. Maximizes the minimum angle /
    triangle quality.
  - valence: Botsch-Kobbelt valence equalization -- flip if it lowers the total
    deviation of the four involved vertices from their ideal valence (6 interior,
    4 boundary). Regularizes connectivity.

The criterion is evaluated for all edges at once (batched numpy), and conflicting
flips (two flips sharing a triangle) are resolved per pass by a Luby-style
maximal-independent-set step (random priority + scatter-max over faces) -- no
Python loop over edges. igl.edge_flaps (C++) is the only library call per pass.
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _angles(U, V):
    """Row-wise angle between vectors U and V."""
    dot = np.sum(U * V, axis=1)
    denom = np.linalg.norm(U, axis=1) * np.linalg.norm(V, axis=1) + 1e-20
    return np.arccos(np.clip(dot / denom, -1.0, 1.0))


def _tri_normals(V, T):
    """Unnormalized face normals for triangle index array T (n,3)."""
    return np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])


def _edge_flip(V, F, criterion, iterations, feature_angle=30.0, seed=0):
    """Vectorized edge flipping. Returns (new_F, total_flips)."""
    import igl

    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int64).copy()
    N = len(V)
    rng = np.random.default_rng(seed)
    total = 0

    for _ in range(int(iterations)):
        E, EMAP, EF, EI = igl.edge_flaps(F)
        E = np.asarray(E, dtype=np.int64)
        EF = np.asarray(EF, dtype=np.int64)
        EI = np.asarray(EI, dtype=np.int64)

        intr = EF[:, 1] >= 0                       # interior edges only
        if not intr.any():
            break
        f0 = EF[intr, 0]; f1 = EF[intr, 1]
        i0 = EI[intr, 0]; i1 = EI[intr, 1]
        a = E[intr, 0]; b = E[intr, 1]
        c = F[f0, i0]                              # apex of f0 (opposite the edge)
        d = F[f1, i1]                              # apex of f1
        p = F[f0, (i0 + 1) % 3]
        q = F[f0, (i0 + 2) % 3]

        # boundary vertices: RXMesh excludes any flip touching one of them.
        bvert = np.zeros(N, dtype=bool)
        bv = np.unique(E[~intr])
        if bv.size:
            bvert[bv] = True
        touches_boundary = bvert[a] | bvert[b] | bvert[c] | bvert[d]

        # --- criterion ---
        if criterion == "delaunay":
            ang = _angles(V[a] - V[c], V[b] - V[c]) + _angles(V[a] - V[d], V[b] - V[d])
            crit = ang > (np.pi + 1e-6)
        else:  # valence (Botsch-Kobbelt): minimize SUM of squared valence deviation
            deg = np.bincount(E.reshape(-1), minlength=N)
            ideal = np.where(bvert, 4, 6).astype(np.int64)   # interior 6, boundary 4
            da = deg[a] - ideal[a]; db = deg[b] - ideal[b]
            dc = deg[c] - ideal[c]; dd = deg[d] - ideal[d]
            bef = da * da + db * db + dc * dc + dd * dd
            aft = ((da - 1) ** 2 + (db - 1) ** 2 + (dc + 1) ** 2 + (dd + 1) ** 2)
            crit = aft < bef

        # --- guards (all vectorized) ---
        new0 = np.stack([c, p, d], axis=1)
        new1 = np.stack([c, d, q], axis=1)
        n0 = _tri_normals(V, new0); n1 = _tri_normals(V, new1)
        area_ok = (np.linalg.norm(n0, axis=1) > 1e-12) & (np.linalg.norm(n1, axis=1) > 1e-12)
        nf0 = _tri_normals(V, F[f0]); nf1 = _tri_normals(V, F[f1])
        no_fold = np.sum((nf0 + nf1) * (n0 + n1), axis=1) >= 0.0   # don't create a fold

        # feature-edge lock: never flip an edge whose dihedral (angle between the
        # two adjacent face normals) exceeds feature_angle -- that's a crease that
        # must be preserved. feature_angle >= 180 disables the lock.
        nf0u = nf0 / (np.linalg.norm(nf0, axis=1, keepdims=True) + 1e-20)
        nf1u = nf1 / (np.linalg.norm(nf1, axis=1, keepdims=True) + 1e-20)
        dihedral = np.degrees(np.arccos(np.clip(np.sum(nf0u * nf1u, axis=1), -1.0, 1.0)))
        not_feature = dihedral <= feature_angle

        # would the new edge (c,d) already exist? -> non-manifold; skip
        ek = (np.minimum(E[:, 0], E[:, 1]).astype(np.int64) * N
              + np.maximum(E[:, 0], E[:, 1]))
        cd = (np.minimum(c, d).astype(np.int64) * N + np.maximum(c, d))
        dup = np.isin(cd, ek)

        ok = crit & area_ok & no_fold & not_feature & (c != d) & ~dup & ~touches_boundary
        cand = np.flatnonzero(ok)
        if cand.size == 0:
            break

        # --- Luby maximal independent set, VERTEX-disjoint: a flip wins iff it
        #     out-ranks every other candidate sharing any of its 4 vertices
        #     (a,b,c,d). Vertex-disjoint subsumes face-disjoint (so topology is
        #     safe) AND keeps each flip's valence change independent -- without
        #     this, two flips sharing only a vertex are scored on stale valence
        #     and the valence criterion oscillates. ---
        prio = rng.permutation(cand.size)
        vmax = np.full(N, -1, dtype=np.int64)
        for varr in (a, b, c, d):
            np.maximum.at(vmax, varr[cand], prio)
        win = ((prio == vmax[a[cand]]) & (prio == vmax[b[cand]])
               & (prio == vmax[c[cand]]) & (prio == vmax[d[cand]]))
        W = cand[win]
        # also ensure no two winners produce the same new edge
        _, uidx = np.unique(cd[W], return_index=True)
        W = W[np.sort(uidx)]
        if W.size == 0:
            break

        F[f0[W]] = new0[W]
        F[f1[W]] = new1[W]
        total += int(W.size)

    return F, total


class EdgeFlipNode(io.ComfyNode):
    """Standalone edge-flip / Delaunay-flip (extrinsic, keeps vertices)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackEdgeFlip",
            display_name="Edge Flip",
            category="geompack/remeshing",
            description=(
                "Flip interior edges (swap the shared diagonal of two triangles) to "
                "improve the triangulation, keeping the vertices fixed. Extrinsic, so the "
                "output is a normal mesh.\n"
                "\n"
                "criterion=delaunay: flip when the two angles opposite an edge sum to > "
                "180 deg (maximizes the minimum angle / triangle quality). "
                "criterion=valence: Botsch-Kobbelt valence equalization -- flip when it "
                "lowers the total deviation from ideal valence (6 interior / 4 boundary), "
                "regularizing the connectivity (the flip step of isotropic remeshing).\n"
                "\n"
                "Vectorized (batched criterion + Luby independent-set per pass), so it is "
                "fast even on million-edge meshes."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Combo.Input("criterion", options=["delaunay", "valence"], default="delaunay", tooltip=(
                    "delaunay = maximize min angle (flip when opposite angles sum > 180 deg). "
                    "valence = Botsch-Kobbelt connectivity regularization (flip toward ideal "
                    "valence 6/4).")),
                io.Float.Input("feature_angle", default=30.0, min=0.0, max=180.0, step=1.0, tooltip=(
                    "Preserve sharp edges: any edge whose dihedral (angle between the two "
                    "adjacent face normals) exceeds this is LOCKED and never flipped, so "
                    "creases/feature lines survive. Lower = protect more edges (CAD: ~20-45). "
                    "Set 180 to disable protection and flip everything.")),
                io.Int.Input("iterations", default=10, min=1, max=100, step=1, tooltip=(
                    "Max flip passes. Each pass flips a conflict-free set and re-evaluates; "
                    "stops early once no edge wants to flip (converged).")),
                io.Int.Input("seed", default=0, min=0, max=2_000_000_000, step=1, tooltip=(
                    "Seed for the independent-set tie-breaking (reproducible results).")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="flipped_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, criterion="delaunay", feature_angle=30.0, iterations=10, seed=0):
        import time

        V = np.asarray(trimesh.vertices, dtype=np.float64)
        F = np.asarray(trimesh.faces, dtype=np.int64)
        nv, nf = len(V), len(F)
        log.info("Edge Flip: %s | %d verts %d faces | feature=%.0fdeg iters=%d",
                 criterion, nv, nf, feature_angle, iterations)

        min_ang_before = float(np.degrees(np.min(trimesh.face_angles))) if nf else 0.0
        # how many edges are locked as features (dihedral > feature_angle)
        try:
            n_features = int((np.degrees(trimesh.face_adjacency_angles) > feature_angle).sum())
        except Exception:
            n_features = 0

        t0 = time.perf_counter()
        F_new, flips = _edge_flip(V, F, criterion, iterations, feature_angle, seed)
        elapsed = time.perf_counter() - t0

        result = trimesh_module.Trimesh(vertices=V, faces=F_new, process=False)
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            result.metadata = trimesh.metadata.copy()
        # carry over per-vertex fields (vertices are unchanged); face fields are not valid
        if getattr(trimesh, "vertex_attributes", None):
            for k, vals in trimesh.vertex_attributes.items():
                try:
                    result.vertex_attributes[k] = vals
                except Exception:
                    pass
        result.metadata["edge_flip"] = {"criterion": criterion, "feature_angle": feature_angle,
                                        "iterations": iterations, "flips": flips}

        min_ang_after = float(np.degrees(np.min(result.face_angles))) if nf else 0.0

        info = (
            f"Edge Flip ({criterion}):\n"
            f"\n"
            f"Flips: {flips:,}\n"
            f"Feature edges preserved: {n_features:,} (dihedral > {feature_angle:.0f} deg)\n"
            f"Vertices: {nv:,} (unchanged)\n"
            f"Faces: {nf:,} (unchanged)\n"
            f"Min angle: {min_ang_before:.2f} deg -> {min_ang_after:.2f} deg\n"
            f"Time: {elapsed:.2f}s"
        )
        log.info("Edge Flip done: %d flips, min-angle %.2f -> %.2f, %.2fs",
                 flips, min_ang_before, min_ang_after, elapsed)
        return io.NodeOutput(result, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackEdgeFlip": EdgeFlipNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackEdgeFlip": "Edge Flip"}
