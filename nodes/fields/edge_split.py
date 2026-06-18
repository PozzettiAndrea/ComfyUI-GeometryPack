# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors
"""Edge Split — refine a mesh until every edge is <= max_edge_length, carrying fields exactly.

This is the "split" operator of isotropic remeshing in isolation. Because splitting only
refines (it never moves or removes geometry), field transport is EXACT and search-free —
the operations themselves give the correspondence (cf. Progressive Meshes / AMR prolongation):

  - face fields  -> each child face inherits its parent face's value (parent index is returned
    by the subdivider). Exact for labels (cad_face_id) AND continuous per-face data.
  - vertex fields -> new vertices lie exactly inside their parent ORIGINAL triangle (planar),
    so we barycentric-interpolate the original per-vertex field there. Original vertices keep
    their values verbatim. Integer/label vertex fields take the dominant-weight original vertex
    (never averaged into nonexistent labels).

No BVH / closest-point query is needed — the parent triangle is known from the subdivision.
"""
import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("GeometryPack")


def _is_discrete(arr):
    return np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_


def split_to_max_edge(mesh, max_edge, max_iter=10):
    """Return (new_mesh, info_dict). Refines until max edge <= max_edge, carrying all
    vertex_attributes + face_attributes exactly via parent-triangle provenance."""
    V0 = np.asarray(mesh.vertices, dtype=np.float64)
    F0 = np.asarray(mesh.faces)
    n0 = len(V0)

    V2, F2, parent = trimesh_module.remesh.subdivide_to_size(
        V0, F0, max_edge=float(max_edge), max_iter=int(max_iter), return_index=True)
    parent = np.asarray(parent)                       # (n_newfaces,) -> original face id

    new = trimesh_module.Trimesh(vertices=V2, faces=F2, process=False)
    new.metadata = dict(getattr(mesh, "metadata", {}) or {})

    # --- vertex fields: original verts keep values; new verts barycentric in parent tri ---
    vattr = dict(getattr(mesh, "vertex_attributes", {}) or {})
    n_new_v = len(V2) - n0
    bary = pf = None
    if n_new_v > 0 and vattr:
        # parent original face for each NEW vertex (any incident new face works)
        pf_of_v = np.full(len(V2), -1, dtype=np.int64)
        for fi in range(len(F2)):
            for v in F2[fi]:
                if v >= n0 and pf_of_v[v] < 0:
                    pf_of_v[v] = parent[fi]
        new_ids = np.arange(n0, len(V2))
        pf = pf_of_v[new_ids]
        tri = F0[pf]                                   # (k,3) original vertex ids
        bary = trimesh_module.triangles.points_to_barycentric(
            V0[tri], V2[new_ids])                      # (k,3), exact (planar)

    for name, arr in vattr.items():
        arr = np.asarray(arr)
        out = np.empty((len(V2),) + arr.shape[1:], dtype=arr.dtype)
        out[:n0] = arr                                 # originals verbatim
        if n_new_v > 0:
            tri = F0[pf]
            f = arr[tri]                               # (k,3,...)
            if _is_discrete(arr):
                dom = tri[np.arange(len(pf)), np.argmax(bary, axis=1)]
                out[n0:] = arr[dom]
            else:
                w = bary.reshape(bary.shape + (1,) * (f.ndim - 2))
                out[n0:] = (w * f).sum(axis=1).astype(arr.dtype, copy=False)
        new.vertex_attributes[name] = out

    # --- face fields: each child inherits its parent (exact, any dtype) ---
    fattr = dict(getattr(mesh, "face_attributes", {}) or {})
    for name, arr in fattr.items():
        new.face_attributes[name] = np.asarray(arr)[parent]

    e = np.vstack([F2[:, [0, 1]], F2[:, [1, 2]], F2[:, [2, 0]]])
    max_after = float(np.linalg.norm(V2[e[:, 0]] - V2[e[:, 1]], axis=1).max()) if len(F2) else 0.0
    info = {"v0": n0, "f0": len(F0), "v1": len(V2), "f1": len(F2),
            "max_edge_after": max_after, "target": float(max_edge),
            "vfields": list(vattr.keys()), "ffields": list(fattr.keys())}
    return new, info


class EdgeSplitNode(io.ComfyNode):
    """Split long edges until every edge <= max length, carrying vertex & face fields exactly."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackEdgeSplit",
            display_name="Edge Split",
            category="geompack/remeshing",
            inputs=[
                io.Custom("TRIMESH").Input("mesh",
                    tooltip="Input mesh. Its vertex_attributes / face_attributes are carried through the split exactly."),
                io.Float.Input("max_edge_length", default=1.0, min=1e-6, max=1e9, step=0.001,
                    tooltip="Upper bound on edge length, in the mesh's world units. Every edge longer than this is split (recursively) by inserting midpoints. Smaller = finer mesh. This is a true cap on edge length — unlike OCC BRepMesh, which has no max-length knob."),
                io.Int.Input("max_iterations", default=10, min=1, max=50, step=1, advanced=True,
                    tooltip="Safety cap on subdivision passes. If the target needs more than this many halvings the result may still have a few longer edges (see the reported max_edge_after)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, max_edge_length=1.0, max_iterations=10):
        new, d = split_to_max_edge(mesh, max_edge_length, max_iterations)
        lines = [
            f"Edge Split: {d['v0']}v/{d['f0']}f -> {d['v1']}v/{d['f1']}f | "
            f"max edge {d['max_edge_after']:.4g} (target {d['target']:.4g})"
            f"{'  [hit max_iterations]' if d['max_edge_after'] > d['target'] * 1.0001 else ''}",
        ]
        if d["vfields"]:
            lines.append(f"  vertex fields carried: {d['vfields']}")
        if d["ffields"]:
            lines.append(f"  face fields carried (exact, per parent face): {d['ffields']}")
        if not d["vfields"] and not d["ffields"]:
            lines.append("  (no fields on input mesh)")
        info = "\n".join(lines)
        log.info("[EdgeSplit] %s", info)
        return io.NodeOutput(new, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackEdgeSplit": EdgeSplitNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackEdgeSplit": "Edge Split"}
