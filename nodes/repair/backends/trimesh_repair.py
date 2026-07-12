# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Trimesh mesh-repair backend node (merge duplicate vertices + drop degenerate/cap faces)."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geompack")


def _face_max_angle_apex(V, F):
    """Per-face: (max interior angle [rad], local apex index 0/1/2 of that angle)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    def ang(u, w):
        un = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
        wn = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)
        return np.arccos(np.clip(np.sum(un * wn, axis=1), -1.0, 1.0))
    A = np.stack([ang(b - a, c - a), ang(a - b, c - b), ang(a - c, b - c)], axis=1)
    return A.max(axis=1), A.argmax(axis=1)


def _collapse_caps(V, F, vattrs, fattrs, max_angle_rad):
    """Collapse cap faces (max interior angle >= threshold) by merging each apex onto
    the nearer of its two base vertices. Returns (V, F, vattrs, fattrs, n_caps)."""
    maxang, apex_local = _face_max_angle_apex(V, F)
    cap = maxang >= max_angle_rad
    n_caps = int(cap.sum())
    if n_caps == 0:
        return V, F, vattrs, fattrs, 0

    ci = np.flatnonzero(cap)
    al = apex_local[ci]
    apex_v = F[ci, al]
    b1 = F[ci, (al + 1) % 3]
    b2 = F[ci, (al + 2) % 3]
    len1 = np.linalg.norm(V[apex_v] - V[b1], axis=1)
    len2 = np.linalg.norm(V[apex_v] - V[b2], axis=1)
    base_v = np.where(len1 <= len2, b1, b2)          # collapse the shorter apex edge

    # union-find: attach apex under base so the surviving position is the base's.
    parent = np.arange(len(V))
    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r
    for a_v, bv in zip(apex_v.tolist(), base_v.tolist()):
        ra, rb = find(int(a_v)), find(int(bv))
        if ra != rb:
            parent[ra] = rb                          # base side becomes the root

    roots = np.array([find(i) for i in range(len(V))])
    keep = np.unique(roots)                           # surviving (representative) vertices
    newidx = np.full(len(V), -1, dtype=np.int64)
    newidx[keep] = np.arange(len(keep))
    vmap = newidx[roots]                              # old vertex -> compact new index

    V2 = V[keep]
    vattrs2 = {k: v[keep] for k, v in vattrs.items()}

    Fr = vmap[F]
    good = (Fr[:, 0] != Fr[:, 1]) & (Fr[:, 1] != Fr[:, 2]) & (Fr[:, 0] != Fr[:, 2])
    F2 = Fr[good]
    fattrs2 = {k: v[good] for k, v in fattrs.items()}
    return V2, F2, vattrs2, fattrs2, n_caps


class MeshRepairTrimeshNode(io.ComfyNode):
    """Trimesh cleanup backend: merges duplicate vertices, then drops degenerate/cap faces."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackMeshRepair_Trimesh",
            display_name="Mesh Repair Trimesh (backend)",
            category="geompack/repair",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.Float.Input("tolerance", default=1e-5, min=1e-8, max=1e-2, step=1e-6,
                               tooltip="Distance tolerance for merging duplicate vertices (1e-5 "
                                       "recommended for CAD meshes). Runs first -- merging can "
                                       "create degenerate faces, which the next step then cleans up."),
                io.Float.Input("min_area", default=1e-10, min=0.0, max=1.0, step=1e-10,
                               optional=True,
                               tooltip="Faces with area below this are DELETED (zero/near-zero "
                                       "area slivers). 0 disables this test."),
                io.Float.Input("max_angle_deg", default=180.0, min=90.0, max=180.0, step=0.5,
                               optional=True,
                               tooltip="CAP-sliver removal by COLLAPSE. A face whose LARGEST "
                                       "interior angle is >= this is a cap -- its apex is "
                                       "collapsed onto the nearer base vertex. 180 = OFF; set "
                                       "~175-179 to clean caps from OCC/image-derived meshes."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="repaired_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, tolerance=1e-5, min_area=1e-10, max_angle_deg=180.0):
        verts_before, faces_before = len(mesh.vertices), len(mesh.faces)

        # Step 1: merge duplicate/near-duplicate vertices.
        merged = mesh.copy()
        digits = max(0, -int(np.floor(np.log10(tolerance))))
        merged.merge_vertices(digits_vertex=digits)
        verts_after_merge, faces_after_merge = len(merged.vertices), len(merged.faces)

        # Step 2: drop degenerate/cap faces (may have been created by the merge above).
        cleaned = merged
        n_caps = 0
        if 0.0 < float(max_angle_deg) < 180.0:
            V = np.asarray(cleaned.vertices, dtype=np.float64)
            F = np.asarray(cleaned.faces, dtype=np.int64)
            vattrs = {k: np.asarray(v) for k, v in dict(cleaned.vertex_attributes).items()}
            fattrs = {k: np.asarray(v) for k, v in dict(cleaned.face_attributes).items()}
            V2, F2, va2, fa2, n_caps = _collapse_caps(V, F, vattrs, fattrs,
                                                      np.radians(float(max_angle_deg)))
            if n_caps > 0:
                cleaned = trimesh_module.Trimesh(vertices=V2, faces=F2.astype(np.int32), process=False)
                for k, v in va2.items():
                    cleaned.vertex_attributes[k] = v
                for k, v in fa2.items():
                    cleaned.face_attributes[k] = v

        duplicate_mask = np.array([len(set(f)) == 3 for f in cleaned.faces])
        if np.any(~duplicate_mask):
            cleaned.update_faces(duplicate_mask)

        if hasattr(cleaned, 'nondegenerate_faces'):
            area_mask = cleaned.nondegenerate_faces()
            if np.any(~area_mask):
                cleaned.update_faces(area_mask)

        if min_area > 0 and len(cleaned.faces):
            area_mask = cleaned.area_faces >= min_area
            if np.any(~area_mask):
                cleaned.update_faces(area_mask)

        cleaned.remove_unreferenced_vertices()

        verts_after, faces_after = len(cleaned.vertices), len(cleaned.faces)
        merge_verts_removed = verts_before - verts_after_merge
        degen_faces_removed = faces_after_merge - faces_after

        info = f"""Mesh Repair (trimesh):

Merge Vertices (tolerance={tolerance:.2e}, {digits} decimal places):
  Vertices: {verts_before:,} -> {verts_after_merge:,} ({-merge_verts_removed:+,})
  Faces: {faces_before:,} -> {faces_after_merge:,} ({faces_after_merge - faces_before:+,})

Remove Degenerate Faces (cap angle >= {max_angle_deg:g} deg, {n_caps:,} collapsed):
  Vertices: {verts_after_merge:,} -> {verts_after:,} ({verts_after - verts_after_merge:+,})
  Faces: {faces_after_merge:,} -> {faces_after:,} ({-degen_faces_removed:+,})

Overall:
  Vertices: {verts_before:,} -> {verts_after:,} ({verts_after - verts_before:+,})
  Faces: {faces_before:,} -> {faces_after:,} ({faces_after - faces_before:+,})
"""
        log.info("Mesh Repair (trimesh): %dv/%df -> %dv/%df", verts_before, faces_before, verts_after, faces_after)
        return io.NodeOutput(cleaned, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackMeshRepair_Trimesh": MeshRepairTrimeshNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackMeshRepair_Trimesh": "Mesh Repair Trimesh (backend)"}
