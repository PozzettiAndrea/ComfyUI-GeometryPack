# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remove degenerate faces (zero area / duplicate indices) and optionally CAP slivers
(very obtuse triangles) from a mesh.

Degenerate faces can be created by:
- Vertex merging when two vertices of a triangle merge to the same index
- OCC meshing creating sliver triangles at CAD face boundaries
- Import from poorly-constructed mesh files

A CAP sliver is a triangle with one interior angle near 180 deg (its apex sits almost
on the opposite edge). It has NON-zero area, so the area tests miss it -- and you
cannot just DELETE it (that leaves a slit). It is removed by COLLAPSING the apex onto
its nearest base vertex (the shortest of the apex's two edges), which closes the gap
and keeps the mesh connected. Topology only loses the sliver, so vertex/face
attributes (e.g. cad_face_id) are carried through.
"""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


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


class RemoveDegenerateFacesNode(io.ComfyNode):
    """Remove degenerate faces (zero area / duplicate indices) and optional cap slivers."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemoveDegenerateFaces",
            display_name="Remove Degenerate Faces",
            category="geompack/repair",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.Float.Input("min_area", default=1e-10, min=0.0, max=1.0, step=1e-10,
                               optional=True,
                               tooltip="Faces with area below this are DELETED (zero/near-zero "
                                       "area slivers). 0 disables this test."),
                io.Float.Input("max_angle_deg", default=180.0, min=90.0, max=180.0, step=0.5,
                               optional=True, tooltip=(
                    "CAP-sliver removal by COLLAPSE. A face whose LARGEST interior angle is >= this "
                    "is a cap (apex nearly on the opposite edge) -- it has non-zero area so the "
                    "area tests miss it, and deleting it would leave a slit. Instead its apex is "
                    "collapsed onto the nearer base vertex (shortest apex edge), closing the gap "
                    "and keeping the mesh connected. 180 = OFF (no triangle reaches 180); set ~175-"
                    "179 to clean caps from OCC/image-derived meshes. LOWER = more aggressive "
                    "(catches fatter triangles). Runs BEFORE the area/duplicate deletes. Vertex/"
                    "face attributes (e.g. cad_face_id) are preserved.")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="cleaned_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, min_area=1e-10, max_angle_deg=180.0):
        log.info("Input: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
        faces_before = len(mesh.faces)
        verts_before = len(mesh.vertices)

        cleaned_mesh = mesh.copy()
        n_caps = 0

        # Method 0 (NEW): collapse cap slivers by largest-angle threshold.
        if 0.0 < float(max_angle_deg) < 180.0:
            V = np.asarray(cleaned_mesh.vertices, dtype=np.float64)
            F = np.asarray(cleaned_mesh.faces, dtype=np.int64)
            vattrs = {k: np.asarray(v) for k, v in dict(cleaned_mesh.vertex_attributes).items()}
            fattrs = {k: np.asarray(v) for k, v in dict(cleaned_mesh.face_attributes).items()}
            V2, F2, va2, fa2, n_caps = _collapse_caps(V, F, vattrs, fattrs,
                                                      np.radians(float(max_angle_deg)))
            if n_caps > 0:
                log.info("Collapsed %d cap faces (max-angle >= %.1f deg)", n_caps, max_angle_deg)
                cleaned_mesh = trimesh_module.Trimesh(vertices=V2, faces=F2.astype(np.int32),
                                                      process=False)
                for k, v in va2.items():
                    cleaned_mesh.vertex_attributes[k] = v
                for k, v in fa2.items():
                    cleaned_mesh.face_attributes[k] = v

        # Method 1: drop faces with duplicate vertex indices (e.g. [0,1,1]).
        duplicate_mask = np.array([len(set(f)) == 3 for f in cleaned_mesh.faces])
        if np.any(~duplicate_mask):
            cleaned_mesh.update_faces(duplicate_mask)

        # Method 2: trimesh's zero-area test.
        if hasattr(cleaned_mesh, 'nondegenerate_faces'):
            area_mask = cleaned_mesh.nondegenerate_faces()
            if np.any(~area_mask):
                cleaned_mesh.update_faces(area_mask)

        # Method 3: explicit min_area threshold.
        if min_area > 0 and len(cleaned_mesh.faces):
            area_mask = cleaned_mesh.area_faces >= min_area
            if np.any(~area_mask):
                cleaned_mesh.update_faces(area_mask)

        cleaned_mesh.remove_unreferenced_vertices()

        faces_after = len(cleaned_mesh.faces)
        verts_after = len(cleaned_mesh.vertices)
        faces_removed = faces_before - faces_after
        verts_removed = verts_before - verts_after

        cap_line = (f"  Cap faces collapsed (angle >= {max_angle_deg:g} deg): {n_caps:,}\n"
                    if 0.0 < float(max_angle_deg) < 180.0 else "")
        info = f"""Degenerate Face Removal Results:

Before:  {verts_before:,} verts, {faces_before:,} faces
{cap_line}After:   {verts_after:,} verts ({-verts_removed:+,}), {faces_after:,} faces ({-faces_removed:+,})

{'[OK] Removed ' + format(faces_removed, ',') + ' faces' if faces_removed > 0 else '[INFO] Nothing removed'}
"""
        log.info("Removed %d faces (%d caps collapsed), %d unreferenced verts",
                 faces_removed, n_caps, verts_removed)
        return io.NodeOutput(cleaned_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {
    "GeomPackRemoveDegenerateFaces": RemoveDegenerateFacesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemoveDegenerateFaces": "Remove Degenerate Faces",
}
