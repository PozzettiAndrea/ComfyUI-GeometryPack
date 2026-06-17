# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Segment a mesh into regions by curvature TYPE (the (H,K) / shape-operator classes).

CAD primitives are points/curves in (H,K)-space (H=(k1+k2)/2, K=k1*k2):
  - PLANAR     : k1~0, k2~0           (H~0, K~0)
  - CYLINDRICAL: one principal ~0     (K~0, H!=0)  -- developable (cylinders/cones)
  - SPHERICAL  : k1~k2, same sign     (umbilic; K~H^2 > 0)
  - SADDLE     : k1,k2 opposite signs (K < 0)
  - GENERIC    : everything else

Classification is RESOLUTION-AWARE: a principal curvature counts as "zero" only when
its radius exceeds `flat_radius_edges` x the LOCAL edge length (sampling theory -- a
fillet finer than the local triangles isn't real). Output is per-vertex class + a
connected-region label, ready to colour or split.
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")

_CLASS = {"planar": 0, "cylindrical": 1, "spherical": 2, "saddle": 3, "generic": 4}
_CLASS_NAME = {v: k for k, v in _CLASS.items()}


class SegmentByCurvatureNode(io.ComfyNode):
    """Segment a mesh into curvature-type regions (planar/cylindrical/spherical/...)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSegmentByCurvature",
            display_name="Segment by Curvature",
            category="geompack/analysis",
            description=(
                "Classify every vertex by curvature TYPE from the principal curvatures "
                "(k1,k2): planar (both ~0), cylindrical/conical (one ~0, developable), "
                "spherical (k1~k2 same sign, umbilic), saddle (opposite signs), or generic. "
                "Then label connected regions of the same class. The 'zero curvature' test "
                "is resolution-aware (a principal curvature is flat only if its radius > "
                "flat_radius_edges x the LOCAL edge length). Outputs vertex fields "
                "segment_class (0 planar,1 cyl,2 sphere,3 saddle,4 generic) and "
                "segment_region (connected-component id) -- colour or Split By Field on them."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("radius", default=5, min=1, max=12, step=1, tooltip=(
                    "Neighborhood (k-ring) for the libigl quadric principal-curvature fit. "
                    "Larger = smoother/noise-robust, blurs features. ~3 crisp CAD, ~5 default.")),
                io.Float.Input("flat_radius_edges", default=8.0, min=1.0, max=100.0, step=1.0, tooltip=(
                    "A principal curvature counts as ZERO (flat / developable direction) when "
                    "its radius of curvature exceeds this many LOCAL edge lengths. Higher = "
                    "stricter 'flat' (more vertices called planar/cylindrical). ~5-15 typical.")),
                io.Float.Input("umbilic_tolerance", default=0.3, min=0.0, max=1.0, step=0.05, tooltip=(
                    "How close k1 and k2 must be (relative) to call a point SPHERICAL rather "
                    "than generic: |k1-k2| / (|k1|+|k2|) < this. Larger = more spherical.")),
                io.Int.Input("min_region_size", default=10, min=0, max=1000000, step=1, tooltip=(
                    "Connected regions with fewer vertices than this are relabeled 'generic' "
                    "(class 4) -- removes speckle from noise. 0 = keep all regions.")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="segmented_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, radius=5, flat_radius_edges=8.0, umbilic_tolerance=0.3,
                min_region_size=10):
        import igl
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components

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

        # per-vertex local edge length
        ev = np.asarray(mesh.edges_unique)
        el = np.asarray(mesh.edges_unique_length, dtype=np.float64)
        loc = np.zeros(n); cnt = np.zeros(n)
        np.add.at(loc, ev[:, 0], el); np.add.at(loc, ev[:, 1], el)
        np.add.at(cnt, ev[:, 0], 1.0); np.add.at(cnt, ev[:, 1], 1.0)
        loc = loc / np.maximum(cnt, 1.0)
        loc[loc <= 0] = float(np.mean(el)) if len(el) else 1.0

        # "flat" if radius of curvature > flat_radius_edges * local edge length,
        # i.e. |kappa| * local_edge < 1 / flat_radius_edges.
        tau = 1.0 / float(max(1.0, flat_radius_edges))
        flat1 = np.abs(k1) * loc < tau
        flat2 = np.abs(k2) * loc < tau

        cls_arr = np.full(n, _CLASS["generic"], dtype=np.int64)
        cls_arr[flat1 & flat2] = _CLASS["planar"]
        cls_arr[flat1 ^ flat2] = _CLASS["cylindrical"]                 # exactly one flat
        both = ~flat1 & ~flat2
        saddle = both & (k1 * k2 < 0)
        cls_arr[saddle] = _CLASS["saddle"]
        denom = np.abs(k1) + np.abs(k2) + 1e-12
        umbilic = both & (~saddle) & (np.abs(k1 - k2) / denom < float(umbilic_tolerance))
        cls_arr[umbilic] = _CLASS["spherical"]
        # remaining `both & same-sign & not-umbilic` stay generic

        # connected regions: components of the subgraph of same-class edges
        same = cls_arr[ev[:, 0]] == cls_arr[ev[:, 1]]
        e = ev[same]
        rows = np.concatenate([e[:, 0], e[:, 1]])
        cols = np.concatenate([e[:, 1], e[:, 0]])
        A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
        n_reg, region = connected_components(A, directed=False)

        # drop tiny regions -> generic
        if min_region_size > 0:
            counts = np.bincount(region, minlength=n_reg)
            small = counts[region] < int(min_region_size)
            cls_arr[small] = _CLASS["generic"]

        mesh.vertex_attributes["segment_class"] = cls_arr.astype(np.float32)
        mesh.vertex_attributes["segment_region"] = region.astype(np.float32)
        mesh.metadata = (mesh.metadata.copy() if mesh.metadata else {})
        mesh.metadata["segmentation"] = {
            "radius": int(radius), "flat_radius_edges": float(flat_radius_edges),
            "umbilic_tolerance": float(umbilic_tolerance), "n_regions": int(n_reg),
        }

        counts = {name: int((cls_arr == idx).sum()) for name, idx in _CLASS.items()}
        pct = {k: 100.0 * v / max(1, n) for k, v in counts.items()}
        info = (
            f"Segment by Curvature:\n"
            f"\n"
            f"Vertices: {n:,} | regions: {n_reg:,}\n"
            f"\n"
            f"planar:      {counts['planar']:,} ({pct['planar']:.1f}%)\n"
            f"cylindrical: {counts['cylindrical']:,} ({pct['cylindrical']:.1f}%)\n"
            f"spherical:   {counts['spherical']:,} ({pct['spherical']:.1f}%)\n"
            f"saddle:      {counts['saddle']:,} ({pct['saddle']:.1f}%)\n"
            f"generic:     {counts['generic']:,} ({pct['generic']:.1f}%)\n"
            f"\n"
            f"Fields: segment_class (0 planar,1 cyl,2 sphere,3 saddle,4 generic), segment_region"
        )
        log.info("Segment by Curvature: %d regions | %s", n_reg, counts)
        return io.NodeOutput(mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSegmentByCurvature": SegmentByCurvatureNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSegmentByCurvature": "Segment by Curvature"}
