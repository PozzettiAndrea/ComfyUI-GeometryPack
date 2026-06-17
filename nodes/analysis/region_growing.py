# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Region growing segmentation: seed faces and grow regions by SIMILARITY.

Unlike segment_patches (which cuts at sharp dihedral edges), this clusters faces
by a tunable similarity metric -- a face joins a region while it stays close to
the region's running representative in:
  - normal direction        (group co-planar / smoothly-varying faces)
  - + face area weighting    (large faces dominate the representative -> robust
                              region normals, the VSA-style area-weighted L2,1 metric)
  - + curvature              (also require similar mean curvature -> separate a
                              flat from a fillet even when their normals agree at
                              the seam)
Seeds are taken flattest-first (lowest curvature), so big planar regions form
first and creases/fillets fall out as their own regions. Each component term has
its own weight. Outputs per-face `region_id` (1..R) for colouring / Split By Field
/ per-region primitive fitting.
"""

import logging
from collections import deque

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")

_CRITERIA = {
    "normals": (True, False, False),
    "normals + area": (True, False, True),
    "curvature + normals": (True, True, False),
    "curvature": (False, True, False),
}


def _face_mean_curvature(V, F):
    """Per-face |mean curvature|: vertex mean-curvature (cotan Laplacian) averaged
    to faces. Robust, cheap, and only used for seed ordering + the curvature term."""
    import igl
    L = igl.cotmatrix(V, F)
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    Minv = 1.0 / np.clip(np.asarray(M.diagonal()), 1e-12, None)
    Hv = 0.5 * np.linalg.norm(Minv[:, None] * (L @ V), axis=1)        # per-vertex |H|
    return Hv[F].mean(axis=1)                                          # per-face |H|


def _build_face_neighbors_csr(adj_pairs, m):
    """CSR (indptr, indices) face-adjacency from the shared-edge face pairs."""
    a = adj_pairs[:, 0]
    b = adj_pairs[:, 1]
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    order = np.argsort(src, kind="stable")
    src_s = src[order]
    dst_s = dst[order]
    indptr = np.zeros(m + 1, dtype=np.int64)
    np.add.at(indptr, src_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, dst_s.astype(np.int64)


def _region_grow(N, areas, Hface, indptr, indices, use_normal, use_curv, use_area,
                 w_normal, w_curv, threshold):
    """Greedy flood-fill region growing. A face joins a region while its weighted
    dissimilarity to the region's running (area-weighted) representative stays
    below `threshold`. Seeds flattest-first. Returns per-face region id (1..R)."""
    m = len(N)
    cscale = float(Hface.std()) + 1e-9
    order = np.argsort(Hface, kind="stable")           # flattest faces seed first
    labels = np.full(m, -1, dtype=np.int64)
    rid = 0
    for seed in order:
        if labels[seed] >= 0:
            continue
        rid += 1
        labels[seed] = rid
        w0 = float(areas[seed]) if use_area else 1.0
        n_acc = N[seed] * w0                            # running (weighted) normal sum
        c_acc = float(Hface[seed]) * w0                 # running weighted curvature sum
        w_sum = w0
        q = deque((seed,))
        while q:
            g = q.popleft()
            n_rep = n_acc / (np.linalg.norm(n_acc) + 1e-12)
            c_rep = c_acc / w_sum
            for idx in range(indptr[g], indptr[g + 1]):
                h = indices[idx]
                if labels[h] >= 0:
                    continue
                d = 0.0
                if use_normal:
                    d += w_normal * (1.0 - float(np.dot(N[h], n_rep)))
                if use_curv:
                    d += w_curv * abs(float(Hface[h]) - c_rep) / cscale
                if d <= threshold:
                    labels[h] = rid
                    wh = float(areas[h]) if use_area else 1.0
                    n_acc = n_acc + N[h] * wh
                    c_acc += float(Hface[h]) * wh
                    w_sum += wh
                    q.append(h)
    return labels


def _merge_small(labels, indptr, indices, areas, min_faces):
    """Merge regions smaller than min_faces into the adjacent region they share the
    most faces with. Iterates to a fixed point; relabels 1..R contiguously."""
    if min_faces <= 1:
        uniq, inv = np.unique(labels, return_inverse=True)
        return (inv + 1).astype(np.int64)
    labels = labels.copy()
    for _ in range(1000):
        uniq, inv = np.unique(labels, return_inverse=True)
        labels = inv.astype(np.int64)
        counts = np.bincount(labels)
        small = np.where(counts < min_faces)[0]
        if len(small) == 0:
            break
        small_set = set(small.tolist())
        changed = False
        for r in small:
            members = np.where(labels == r)[0]
            # tally neighbour-region face counts
            nbr = {}
            for g in members:
                for idx in range(indptr[g], indptr[g + 1]):
                    rl = labels[indices[idx]]
                    if rl != r:
                        nbr[rl] = nbr.get(rl, 0) + 1
            if nbr:
                best = max(nbr, key=nbr.get)
                labels[members] = best
                changed = True
        if not changed:
            break
    uniq, inv = np.unique(labels, return_inverse=True)
    return (inv + 1).astype(np.int64)


class RegionGrowingNode(io.ComfyNode):
    """Seed + grow regions, clustering faces by a tunable similarity metric."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRegionGrowing",
            display_name="Region Growing",
            category="geompack/analysis",
            description=(
                "Segment a mesh by SEEDING faces (flattest first) and GROWING regions: a "
                "face joins while it stays similar to the region's running representative. "
                "Choose what 'similar' means in `criterion` (normals / normals+area / "
                "curvature+normals / curvature) and tune each component's weight + the "
                "`threshold` (lower = more, smaller regions). Outputs per-face region_id "
                "(1..R) -- colour it, Split By Field, or fit a primitive per region."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Combo.Input("criterion",
                               options=list(_CRITERIA.keys()), default="curvature + normals",
                               tooltip=(
                    "What makes two faces belong to the same region:\n"
                    " normals = co-planar / smoothly-varying normals only;\n"
                    " normals + area = same, but big faces dominate the region normal "
                    "(area-weighted, robust on uneven tessellation);\n"
                    " curvature + normals = also require similar mean curvature (separates a "
                    "flat from a fillet that meet tangentially);\n"
                    " curvature = group by curvature magnitude alone.")),
                io.Float.Input("threshold", default=0.05, min=0.0005, max=2.0, step=0.005, display_mode="number", tooltip=(
                    "Similarity tolerance to ADD a face to the growing region. LOWER = stricter "
                    "= more, smaller regions; HIGHER = looser = fewer, larger regions. The "
                    "normal term is 1-cos(angle) (0.015~10deg, 0.06~20deg, 0.13~30deg); the "
                    "curvature term is in units of the mesh's curvature std. Default 0.05.")),
                io.Float.Input("normal_weight", default=1.0, min=0.0, max=20.0, step=0.1, tooltip=(
                    "Weight of the NORMAL-difference term (1-cos angle between the face normal "
                    "and the region's representative normal). Higher = normals matter more. "
                    "Ignored unless the criterion includes normals.")),
                io.Float.Input("curvature_weight", default=1.0, min=0.0, max=20.0, step=0.1, tooltip=(
                    "Weight of the CURVATURE-difference term (|H_face - H_region| / curvature "
                    "std). Higher = curvature matters more, so a fillet won't merge into the "
                    "abutting flat. Ignored unless the criterion includes curvature.")),
                io.Int.Input("min_region_size", default=10, min=1, max=1000000, step=1, tooltip=(
                    "Regions with fewer faces than this are merged into the neighbour they "
                    "share the most boundary with (de-speckle). 1 = keep every region.")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="segmented_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, criterion="curvature + normals", threshold=0.05,
                normal_weight=1.0, curvature_weight=1.0, min_region_size=10):
        mesh = trimesh.copy()
        try:
            mesh.merge_vertices()
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
        except Exception as e:
            log.debug("preclean skipped: %s", e)

        V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        m = len(F)
        use_normal, use_curv, use_area = _CRITERIA[criterion]

        N = np.asarray(mesh.face_normals, dtype=np.float64)
        areas = np.asarray(mesh.area_faces, dtype=np.float64)
        Hface = _face_mean_curvature(V, F)            # also used for seed ordering

        adj = np.asarray(mesh.face_adjacency)
        if len(adj) == 0:
            raise ValueError("Mesh has no face adjacency (disconnected or degenerate).")
        indptr, indices = _build_face_neighbors_csr(adj, m)

        labels = _region_grow(N, areas, Hface, indptr, indices,
                              use_normal, use_curv, use_area,
                              float(normal_weight), float(curvature_weight), float(threshold))
        labels = _merge_small(labels, indptr, indices, areas, int(min_region_size))
        n_reg = int(labels.max())

        mesh.face_attributes["region_id"] = labels.astype(np.int64)
        # also a per-vertex copy (majority face label) for vertex-field consumers
        vlab = np.zeros(len(V), dtype=np.float32)
        cnt = np.zeros(len(V))
        for k in range(3):
            np.add.at(vlab, F[:, k], labels)
            np.add.at(cnt, F[:, k], 1.0)
        mesh.vertex_attributes["region_id"] = (vlab / np.maximum(cnt, 1.0)).astype(np.float32)

        sizes = np.bincount(labels)[1:]
        info = (
            f"Region Growing:\n\n"
            f"Faces: {m:,} | regions: {n_reg:,}\n"
            f"criterion: {criterion} (n_w={normal_weight}, c_w={curvature_weight})\n"
            f"threshold: {threshold} | min_region_size: {min_region_size}\n\n"
            f"region face-count: min {int(sizes.min())}, median {int(np.median(sizes))}, "
            f"max {int(sizes.max())}\n"
            f"Field: face_attributes['region_id'] (1..{n_reg})"
        )
        log.info("Region Growing: %d regions (%s, thr=%.3f)", n_reg, criterion, threshold)
        return io.NodeOutput(mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRegionGrowing": RegionGrowingNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRegionGrowing": "Region Growing"}
