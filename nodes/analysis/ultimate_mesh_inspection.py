# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Ultimate Mesh Inspection - the comprehensive mesh quality / validity analyser.

Computes a full statistics suite (counts, topology, element quality, regularity,
numerics) reported both as worst-case AND mean +/- std, and BAKES every
highlightable defect as a scalar FIELD onto the mesh so it can be lit up in the
VTK viewer by simply selecting that field in the dropdown:

  face fields:   component_id, tri_quality, min_angle_deg, aspect_ratio,
                 is_degenerate, is_sliver, is_needle, is_cap, is_worst_quality,
                 touches_nonmanifold_edge, touches_boundary, is_inverted
  vertex fields: boundary_vertex, nonmanifold_vertex, valence

Outputs the passthrough TRIMESH (with the baked fields -> pipe into Preview Mesh
and pick a field to highlight it), a text report, and a JSON stats string.

Tiers (cheap -> expensive):
  0 numerics/sanity  1 element geometry  2 local topology
  3 global topology (Euler/genus)  4 global geometry (self-intersection, gated)

Note: trimesh.load(process=True) silently merges duplicate verts / drops
degenerate+duplicate faces, so "duplicate/degenerate" counts are only truthful on
a raw (process=False) load. Enable `reload_raw` to re-read the source file (from
metadata['file_path']) without processing for those Tier-0 counts.
"""

import json
import logging

import numpy as np
import trimesh as trimesh_module

from comfy_api.latest import io

log = logging.getLogger("geometrypack")

RAD2DEG = 180.0 / np.pi


def _stats(a):
    """Return (min, max, mean, std) of a 1-D array, NaN-safe; zeros if empty."""
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(a.min()), float(a.max()), float(a.mean()), float(a.std())


def _nonmanifold_and_boundary_edges(mesh):
    """Vectorized via edge grouping. Returns (nm_edge_verts, boundary_edge_verts)."""
    from trimesh.grouping import group_rows
    es = mesh.edges_sorted
    groups = group_rows(es)  # list of arrays of row-indices that are the same edge
    nm_edges = []      # >2 faces
    bnd_edges = []     # ==1 face
    for g in groups:
        if len(g) == 1:
            bnd_edges.append(es[g[0]])
        elif len(g) > 2:
            nm_edges.append(es[g[0]])
    nm_edges = np.array(nm_edges).reshape(-1, 2) if nm_edges else np.zeros((0, 2), int)
    bnd_edges = np.array(bnd_edges).reshape(-1, 2) if bnd_edges else np.zeros((0, 2), int)
    return nm_edges, bnd_edges


def _boundary_loops(bnd_edges):
    """Count boundary loops (connected components of the boundary-edge graph) via union-find."""
    if len(bnd_edges) == 0:
        return 0
    verts = np.unique(bnd_edges)
    idx = {v: i for i, v in enumerate(verts)}
    parent = list(range(len(verts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in bnd_edges:
        ra, rb = find(idx[a]), find(idx[b])
        if ra != rb:
            parent[ra] = rb
    roots = {find(i) for i in range(len(verts))}
    return len(roots)


def _nonmanifold_vertices(mesh, max_faces=400000):
    """Bowtie / non-manifold vertex detection: a vertex whose incident faces don't form a
    single edge-connected fan. Gated by face count (per-vertex grouping is O(sum valence))."""
    nf = len(mesh.faces)
    if nf > max_faces:
        return None  # too big; skip (caller reports "skipped")
    from collections import defaultdict
    faces = mesh.faces
    edge_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        a, b, c = f
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces[(u, v) if u < v else (v, u)].append(fi)
    vert_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        for v in f:
            vert_faces[v].append(fi)
    nm = np.zeros(len(mesh.vertices), dtype=bool)
    for v, flist in vert_faces.items():
        if len(flist) < 2:
            continue
        fidx = {f: i for i, f in enumerate(flist)}
        parent = list(range(len(flist)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for f in flist:
            a, b, c = faces[f]
            for u, w in ((a, b), (b, c), (c, a)):
                if u != v and w != v:
                    continue
                key = (u, w) if u < w else (w, u)
                for f2 in edge_faces[key]:
                    if f2 in fidx and f2 != f:
                        ra, rb = find(fidx[f]), find(fidx[f2])
                        if ra != rb:
                            parent[ra] = rb
        roots = {find(i) for i in range(len(flist))}
        if len(roots) > 1:
            nm[v] = True
    return nm


class UltimateMeshInspection(io.ComfyNode):
    """Comprehensive mesh quality + validity analyser with baked highlight fields."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackUltimateMeshInspection",
            display_name="Ultimate Mesh Inspection",
            category="geompack/analysis",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Boolean.Input("reload_raw", default=False, optional=True,
                    tooltip="Re-read the source file (metadata['file_path']) with process=False so "
                            "duplicate/degenerate counts reflect the file, not trimesh's auto-clean."),
                io.Boolean.Input("check_nonmanifold_vertices", default=True, optional=True,
                    tooltip="Detect bowtie / non-manifold vertices (per-vertex; skipped above ~400k faces)."),
                io.Boolean.Input("check_self_intersection", default=False, optional=True,
                    tooltip="Detect self-intersecting face pairs (expensive; needs pymeshlab)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="report"),
                io.String.Output(display_name="stats_json"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, reload_raw=False, check_nonmanifold_vertices=True,
                check_self_intersection=False):
        mesh = trimesh
        raw_note = ""
        if reload_raw and hasattr(mesh, "metadata") and mesh.metadata.get("file_path"):
            try:
                fp = mesh.metadata["file_path"]
                raw = trimesh_module.load(fp, process=False, maintain_order=True, force="mesh")
                if hasattr(raw, "faces"):
                    mesh = raw
                    raw_note = f" (raw reload of {mesh.metadata.get('file_name', fp)})"
            except Exception as e:
                raw_note = f" (raw reload failed: {e})"

        V = len(mesh.vertices)
        F = int(len(mesh.faces)) if hasattr(mesh, "faces") and mesh.faces is not None else 0
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        S = {}     # structured stats
        rows = []  # (group, label, value, field_or_None)

        def row(group, label, value, field=None):
            rows.append((group, label, value, field))

        # ---- Tier 0: numerics / sanity ----
        nonfinite = int(np.count_nonzero(~np.isfinite(verts)))
        bbox_min = verts.min(0) if V else np.zeros(3)
        bbox_max = verts.max(0) if V else np.zeros(3)
        extents = bbox_max - bbox_min
        diag = float(np.linalg.norm(extents))
        S["counts"] = {"vertices": V, "faces": F, "edges_unique": int(len(mesh.edges_unique)) if F else 0}
        S["bbox"] = {"min": bbox_min.tolist(), "max": bbox_max.tolist(),
                     "extents": extents.tolist(), "diagonal": diag}
        S["numerics"] = {"nonfinite_coords": nonfinite}
        row("Counts", "vertices", V)
        row("Counts", "faces", F)
        row("Counts", "unique edges", int(len(mesh.edges_unique)) if F else 0)
        row("Size", "bbox extents", f"{extents[0]:.4g} x {extents[1]:.4g} x {extents[2]:.4g}")
        row("Size", "bbox diagonal", f"{diag:.4g}")
        row("Numerics", "non-finite coords (NaN/Inf)", nonfinite)

        if F == 0:
            report = cls._format(rows, raw_note)
            return io.NodeOutput(mesh, report, json.dumps(S))

        faces = np.asarray(mesh.faces)

        # ---- duplicates (only truthful on raw load) ----
        uniq_v = np.unique(np.round(verts, 8), axis=0)
        coincident = V - len(uniq_v)
        face_keys = np.sort(faces, axis=1)
        _, fcounts = np.unique(face_keys, axis=0, return_counts=True)
        dup_faces = int(np.sum(fcounts - 1))
        referenced = np.zeros(V, bool); referenced[faces.reshape(-1)] = True
        orphan = int(np.count_nonzero(~referenced))
        S["duplicates"] = {"coincident_vertices": int(coincident), "duplicate_faces": dup_faces,
                           "orphan_vertices": orphan}
        row("Duplicates*", "coincident vertices", int(coincident))
        row("Duplicates*", "duplicate faces", dup_faces)
        row("Duplicates*", "orphan vertices", orphan)

        # ---- Tier 1: element geometry (vectorized) ----
        areas = np.asarray(mesh.area_faces, dtype=np.float64)
        angles = np.asarray(mesh.face_angles, dtype=np.float64) * RAD2DEG   # (F,3) degrees
        face_min_ang = angles.min(1)
        face_max_ang = angles.max(1)
        tv = verts[faces]
        e0 = tv[:, 1] - tv[:, 0]; e1 = tv[:, 2] - tv[:, 1]; e2 = tv[:, 0] - tv[:, 2]
        sum_l2 = (e0 ** 2).sum(1) + (e1 ** 2).sum(1) + (e2 ** 2).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            Q = np.where(sum_l2 > 0, 4.0 * np.sqrt(3.0) * areas / sum_l2, 0.0)
        edge_lens = np.sqrt(np.stack([(e0 ** 2).sum(1), (e1 ** 2).sum(1), (e2 ** 2).sum(1)], 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            aspect = edge_lens.max(1) / np.maximum(edge_lens.min(1), 1e-30)

        is_degenerate = areas <= 1e-14
        is_sliver = (face_min_ang < 10.0) & ~is_degenerate
        is_needle = (aspect > 20.0) & ~is_degenerate
        is_cap = (face_max_ang > 150.0) & ~is_degenerate
        worst_q_face = int(np.argmin(Q))
        is_worst_quality = np.zeros(F, np.float32); is_worst_quality[worst_q_face] = 1.0

        amn, amx, ame, ast = _stats(face_min_ang)
        qmn, qmx, qme, qst = _stats(Q)
        armn, armx, arme, arst = _stats(aspect)
        pct_lt30 = float(100.0 * np.count_nonzero(face_min_ang < 30.0) / F)
        pct_q_lt01 = float(100.0 * np.count_nonzero(Q < 0.1) / F)
        S["element_quality"] = {
            "min_angle_deg": {"worst": amn, "mean": ame, "std": ast, "pct_below_30": pct_lt30},
            "quality_Q": {"worst": qmn, "mean": qme, "std": qst, "pct_below_0.1": pct_q_lt01},
            "aspect_ratio": {"worst": armx, "mean": arme, "std": arst},
            "degenerate": int(is_degenerate.sum()), "sliver": int(is_sliver.sum()),
            "needle": int(is_needle.sum()), "cap": int(is_cap.sum()),
        }
        row("Element quality", "min angle (deg)", f"worst {amn:.2f} | mean {ame:.2f} +/- {ast:.2f}", "min_angle_deg")
        row("Element quality", "% faces min-angle < 30", f"{pct_lt30:.2f}%", "min_angle_deg")
        row("Element quality", "triangle Q (1=ideal)", f"worst {qmn:.3f} | mean {qme:.3f} +/- {qst:.3f}", "tri_quality")
        row("Element quality", "% faces Q < 0.1", f"{pct_q_lt01:.2f}%", "tri_quality")
        row("Element quality", "aspect ratio", f"worst {armx:.2f} | mean {arme:.2f} +/- {arst:.2f}", "aspect_ratio")
        row("Element quality", "worst triangle (by Q)", f"face {worst_q_face} (Q={Q[worst_q_face]:.3g})", "is_worst_quality")
        row("Degenerate", "zero-area", int(is_degenerate.sum()), "is_degenerate")
        row("Degenerate", "slivers (min-ang<10)", int(is_sliver.sum()), "is_sliver")
        row("Degenerate", "needles (aspect>20)", int(is_needle.sum()), "is_needle")
        row("Degenerate", "caps (max-ang>150)", int(is_cap.sum()), "is_cap")

        a_mn, a_mx, a_me, a_st = _stats(areas)
        el = mesh.edges_unique_length
        e_mn, e_mx, e_me, e_st = _stats(el)
        cov = (e_st / e_me) if e_me > 0 else 0.0
        S["distributions"] = {"face_area": {"min": a_mn, "max": a_mx, "mean": a_me, "std": a_st},
                              "edge_length": {"min": e_mn, "max": e_mx, "mean": e_me, "std": e_st, "cov": cov}}
        row("Distributions", "face area", f"min {a_mn:.3g} | max {a_mx:.3g} | mean {a_me:.3g}")
        row("Distributions", "edge length", f"min {e_mn:.3g} | max {e_mx:.3g} | mean {e_me:.3g}")
        row("Distributions", "edge length CoV (uniformity)", f"{cov:.3f}")

        # ---- Tier 2: local topology ----
        nm_edges, bnd_edges = _nonmanifold_and_boundary_edges(mesh)
        n_boundary = len(bnd_edges)
        n_loops = _boundary_loops(bnd_edges)
        n_nm_edges = len(nm_edges)
        try:
            winding_ok = bool(mesh.is_winding_consistent)
        except Exception:
            winding_ok = None
        try:
            watertight = bool(mesh.is_watertight)
        except Exception:
            watertight = None
        n_comp = None
        comp_field = np.zeros(F, np.float32)
        try:
            import trimesh.graph as tg
            comp_lists = tg.connected_components(mesh.face_adjacency, nodes=np.arange(F))
            for ci, fl in enumerate(comp_lists):
                comp_field[fl] = ci
            n_comp = len(comp_lists)
        except Exception:
            try:
                n_comp = int(mesh.body_count)
            except Exception:
                pass

        loop_len_note = ""
        if n_boundary:
            blen = np.linalg.norm(verts[bnd_edges[:, 0]] - verts[bnd_edges[:, 1]], axis=1)
            loop_len_note = f"total {blen.sum():.3g}, per-edge mean {blen.mean():.3g}"

        S["topology"] = {"watertight": watertight, "winding_consistent": winding_ok,
                         "boundary_edges": n_boundary, "boundary_loops": n_loops,
                         "nonmanifold_edges": n_nm_edges, "components": n_comp}
        row("Topology", "watertight", watertight, "touches_boundary")
        row("Topology", "winding consistent", winding_ok)
        row("Topology", "boundary edges", n_boundary, "touches_boundary")
        row("Topology", "boundary loops (holes)", n_loops, "touches_boundary")
        if loop_len_note:
            row("Topology", "boundary length", loop_len_note, "touches_boundary")
        row("Topology", "non-manifold edges (>2 faces)", n_nm_edges, "touches_nonmanifold_edge")
        row("Topology", "connected components", n_comp, "component_id")

        # ---- Tier 3: global topology (Euler / genus) ----
        Eu = int(len(mesh.edges_unique))
        euler = V - Eu + F
        S["topology"]["euler_characteristic"] = euler
        row("Global topology", "Euler characteristic (V-E+F)", euler)
        if watertight and winding_ok and (n_comp == 1):
            genus = (2 - euler) // 2
            S["topology"]["genus"] = int(genus)
            row("Global topology", "genus (handles/tunnels)", int(genus))
        else:
            S["topology"]["genus"] = None
            row("Global topology", "genus", "n/a (needs closed+orientable+1 component)")

        # ---- regularity (valence) ----
        valence = np.bincount(mesh.edges_unique.reshape(-1), minlength=V)
        bnd_v = np.zeros(V, bool)
        if n_boundary:
            bnd_v[np.unique(bnd_edges)] = True
        interior = ~bnd_v
        if interior.any():
            pct_v6 = float(100.0 * np.count_nonzero((valence == 6) & interior) / max(np.count_nonzero(interior), 1))
            irregular = int(np.count_nonzero((valence != 6) & interior))
        else:
            pct_v6, irregular = 0.0, 0
        S["regularity"] = {"pct_interior_valence6": pct_v6, "irregular_interior_vertices": irregular}
        row("Regularity", "% interior verts valence 6", f"{pct_v6:.1f}%", "valence")
        row("Regularity", "irregular interior verts", irregular, "valence")

        # ---- dihedral angles ----
        try:
            dih = np.asarray(mesh.face_adjacency_angles) * RAD2DEG
            d_mn, d_mx, d_me, d_st = _stats(dih)
            sharp = int(np.count_nonzero(dih > 60.0))
            S["dihedral"] = {"mean": d_me, "std": d_st, "max": d_mx, "sharp_gt60": sharp}
            row("Dihedral", "angle (deg)", f"mean {d_me:.2f} +/- {d_st:.2f} | max {d_mx:.2f}")
            row("Dihedral", "sharp edges (>60deg)", sharp)
        except Exception:
            pass

        # ---- inverted faces (heuristic, watertight only) ----
        is_inverted = np.zeros(F, np.float32)
        try:
            if watertight:
                fn = np.asarray(mesh.face_normals)
                fc = verts[faces].mean(1)
                outward = fc - verts.mean(0)
                dots = (fn * outward).sum(1)
                flipped = dots < 0
                if flipped.mean() > 0.5:
                    flipped = ~flipped
                is_inverted = flipped.astype(np.float32)
                row("Geometry", "possibly-inverted faces", int(is_inverted.sum()), "is_inverted")
        except Exception:
            pass

        # ---- Tier 0 expensive: non-manifold vertices ----
        nm_vert_field = np.zeros(V, np.float32)
        if check_nonmanifold_vertices:
            nmv = _nonmanifold_vertices(mesh)
            if nmv is None:
                row("Topology", "non-manifold (bowtie) verts", "skipped (>400k faces)", None)
                S["topology"]["nonmanifold_vertices"] = "skipped"
            else:
                nm_vert_field = nmv.astype(np.float32)
                row("Topology", "non-manifold (bowtie) verts", int(nmv.sum()), "nonmanifold_vertex")
                S["topology"]["nonmanifold_vertices"] = int(nmv.sum())

        # ---- Tier 4: self-intersection (gated) ----
        if check_self_intersection:
            n_si = cls._self_intersections(mesh)
            row("Geometry", "self-intersecting face pairs", n_si if n_si is not None else "n/a (need pymeshlab)")
            S["geometry"] = {"self_intersecting_pairs": n_si}

        # ---- bake highlight fields onto the mesh ----
        try:
            # vectorized face<->edge incidence: encode each undirected edge as one int64 key
            def _enc(a, b):
                a = a.astype(np.int64); b = b.astype(np.int64)
                lo = np.minimum(a, b); hi = np.maximum(a, b)
                return lo * np.int64(V) + hi

            fe_a = _enc(faces[:, 0], faces[:, 1])
            fe_b = _enc(faces[:, 1], faces[:, 2])
            fe_c = _enc(faces[:, 2], faces[:, 0])

            def faces_touching(edge_verts):
                if len(edge_verts) == 0:
                    return np.zeros(F, np.float32)
                targ = _enc(edge_verts[:, 0], edge_verts[:, 1])
                m = np.isin(fe_a, targ) | np.isin(fe_b, targ) | np.isin(fe_c, targ)
                return m.astype(np.float32)

            boundary_face = faces_touching(bnd_edges)
            nm_face = faces_touching(nm_edges)

            mesh.face_attributes["component_id"] = comp_field
            mesh.face_attributes["tri_quality"] = Q.astype(np.float32)
            mesh.face_attributes["min_angle_deg"] = face_min_ang.astype(np.float32)
            mesh.face_attributes["aspect_ratio"] = aspect.astype(np.float32)
            mesh.face_attributes["is_degenerate"] = is_degenerate.astype(np.float32)
            mesh.face_attributes["is_sliver"] = is_sliver.astype(np.float32)
            mesh.face_attributes["is_needle"] = is_needle.astype(np.float32)
            mesh.face_attributes["is_cap"] = is_cap.astype(np.float32)
            mesh.face_attributes["is_worst_quality"] = is_worst_quality
            mesh.face_attributes["touches_boundary"] = boundary_face
            mesh.face_attributes["touches_nonmanifold_edge"] = nm_face
            mesh.face_attributes["is_inverted"] = is_inverted
            mesh.vertex_attributes["boundary_vertex"] = bnd_v.astype(np.float32)
            mesh.vertex_attributes["nonmanifold_vertex"] = nm_vert_field
            mesh.vertex_attributes["valence"] = valence.astype(np.float32)
        except Exception as e:
            log.warning("Could not bake all fields: %s", e)

        report = cls._format(rows, raw_note)
        log.info("[UltimateMeshInspection] V=%d F=%d watertight=%s genus=%s", V, F, watertight,
                 S["topology"].get("genus"))
        return io.NodeOutput(mesh, report, json.dumps(S))

    @staticmethod
    def _self_intersections(mesh):
        try:
            import pymeshlab as ml
            ms = ml.MeshSet()
            ms.add_mesh(ml.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)))
            ms.compute_selection_by_self_intersections_per_face()
            m = ms.current_mesh()
            return int(np.count_nonzero(m.face_selection_array()))
        except Exception as e:
            log.info("self-intersection check unavailable: %s", e)
            return None

    @staticmethod
    def _format(rows, note):
        lines = [f"=== Ultimate Mesh Inspection{note} ===", ""]
        cur = None
        for group, label, value, field in rows:
            if group != cur:
                lines.append(f"[{group}]")
                cur = group
            tag = f"   <- field: {field}" if field else ""
            lines.append(f"  {label:<34} {value}{tag}")
        lines.append("")
        lines.append("* duplicate/orphan counts are only truthful with reload_raw=True")
        lines.append("Fields baked onto mesh -> pipe to Preview Mesh and pick a field to highlight it.")
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "GeomPackUltimateMeshInspection": UltimateMeshInspection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackUltimateMeshInspection": "Ultimate Mesh Inspection",
}
