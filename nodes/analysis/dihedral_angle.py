# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Dihedral Angle Field -- a per-vertex crease / feature measure.

The dihedral angle is intrinsically a PER-EDGE quantity: the angle between the two
faces that share an edge. trimesh's `face_adjacency_angles` gives the DEVIATION from
flat (0 = coplanar, larger = sharper crease; a 90-degree fold = 90 degrees). To turn
that into a VERTEX field we reduce the incident interior edges at each vertex --
default MAX, i.e. the sharpest crease touching the vertex, which is the right signal
for feature / edge-loop detection (a vertex sitting on a crease has at least one
high-dihedral incident edge).

Boundary vertices (only on open edges, which have no second face) and vertices in
flat interiors get 0. Threshold the field to mark crease vertices, colour it in
Preview Mesh, or feed it to Split By Field to isolate flat regions for primitive
fitting / edge-loop extraction."""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _dihedral_vertex_field(mesh, reduction, degrees, signed):
    """Per-vertex reduction of incident face-adjacency (dihedral) angles.

    Returns (values[n], stats_dict). values carry the chosen units/sign."""
    n = len(mesh.vertices)
    v = np.zeros(n, dtype=np.float64)

    fae = np.asarray(mesh.face_adjacency_edges)          # (E,2) shared-edge vertex ids
    ang = np.asarray(mesh.face_adjacency_angles, float)  # (E,) deviation-from-flat, radians
    if fae.size == 0 or ang.size == 0:
        return v, {"n_edges": 0}

    if degrees:
        ang = np.degrees(ang)
    mag = np.abs(ang)                                     # magnitude (>=0)
    if signed:
        conv = np.asarray(mesh.face_adjacency_convex, bool)  # True = convex edge
        val = np.where(conv, mag, -mag)                  # convex +, concave -
    else:
        val = mag

    if reduction == "mean":
        s = np.zeros(n); c = np.zeros(n)
        for col in (0, 1):
            np.add.at(s, fae[:, col], val)
            np.add.at(c, fae[:, col], 1.0)
        v = s / np.maximum(c, 1.0)
    else:  # max (by magnitude, carrying sign)
        best = np.zeros(n)
        for col in (0, 1):
            np.maximum.at(best, fae[:, col], mag)
        for col in (0, 1):
            idx = fae[:, col]
            sel = np.isclose(mag, best[idx])             # this edge is the vertex's sharpest
            v[idx[sel]] = val[sel]

    stats = {
        "n_edges": int(len(ang)),
        "edge_max": float(mag.max()),
        "edge_median": float(np.median(mag)),
        "edge_p99": float(np.percentile(mag, 99)),
    }
    return v, stats


class DihedralAngleFieldNode(io.ComfyNode):
    """Add a per-vertex dihedral-angle (crease) field to a mesh."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackDihedralAngleField",
            display_name="Dihedral Angle Field",
            category="geompack/analysis",
            description=(
                "Add a per-VERTEX dihedral-angle field. Dihedral angle is a per-EDGE quantity "
                "(angle between the two faces on an edge; 0 = flat, 90 = right-angle fold); this "
                "reduces the incident edges at each vertex (MAX = sharpest incident crease, the "
                "signal for feature/edge-loop detection). Boundary + flat-interior vertices = 0. "
                "Colour it in Preview Mesh, threshold it for crease vertices, or feed Split By Field."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Combo.Input("reduction", options=["max", "mean"], default="max", tooltip=(
                    "How to collapse a vertex's incident EDGE dihedrals to one value. max = the "
                    "sharpest crease touching the vertex (best for feature/edge-loop detection; a "
                    "crease vertex lights up even if its other edges are flat). mean = average over "
                    "incident edges (smoother, dilutes thin creases -- more of a 'roughness').")),
                io.Combo.Input("units", options=["degrees", "radians"], default="degrees", tooltip=(
                    "Units of the stored field. degrees is the intuitive crease angle (a sharp box "
                    "edge ~= 90). The value is the DEVIATION from flat (angle between face normals), "
                    "so coplanar = 0 regardless of units.")),
                io.Boolean.Input("signed", default=False, tooltip=(
                    "If on, convex edges store +angle and concave edges -angle (via trimesh "
                    "face_adjacency_convex), so you can tell an outer edge from an inner pocket "
                    "edge. With reduction=max the vertex takes its largest-MAGNITUDE incident edge, "
                    "carrying that edge's sign. Off = unsigned magnitude (0..180), simplest for "
                    "thresholding crease-ness.")),
                io.String.Input("field_name", default="dihedral_angle", tooltip=(
                    "Name of the vertex attribute to write (shows up as this name in Preview Mesh "
                    "and Split By Field).")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh_with_field"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, reduction="max", units="degrees", signed=False,
                field_name="dihedral_angle"):
        mesh = trimesh.copy()
        name = (field_name or "dihedral_angle").strip() or "dihedral_angle"
        n = len(mesh.vertices)
        degrees = units == "degrees"

        if len(mesh.faces) == 0:
            v = np.zeros(n, dtype=np.float32)
            mesh.vertex_attributes[name] = v
            info = f"Dihedral Angle Field: mesh has no faces (point cloud?); wrote zeros to '{name}'."
            log.warning(info)
            return io.NodeOutput(mesh, info, ui={"text": [info]})

        v, stats = _dihedral_vertex_field(mesh, reduction, degrees, bool(signed))
        mesh.vertex_attributes[name] = v.astype(np.float32)

        unit = "deg" if degrees else "rad"
        # crease coverage at a couple of intuitive thresholds (only meaningful unsigned/abs)
        a = np.abs(v)
        t1 = 20.0 if degrees else np.radians(20.0)
        t2 = 45.0 if degrees else np.radians(45.0)
        cov1 = 100.0 * float((a > t1).mean())
        cov2 = 100.0 * float((a > t2).mean())

        info = (
            f"Dihedral Angle Field ({reduction}, {unit}, signed={bool(signed)}):\n\n"
            f"Vertices: {n:,} | interior edges: {stats.get('n_edges', 0):,}\n"
            f"Field '{name}': min {float(v.min()):.2f}, max {float(v.max()):.2f} {unit}\n"
            f"Per-edge dihedral: median {stats.get('edge_median', float('nan')):.2f}, "
            f"p99 {stats.get('edge_p99', float('nan')):.2f}, max {stats.get('edge_max', float('nan')):.2f} {unit}\n\n"
            f"Crease coverage: {cov1:.1f}% of verts > {(20.0 if degrees else 0.35):.0f}{unit}, "
            f"{cov2:.1f}% > {(45.0 if degrees else 0.79):.0f}{unit}\n"
            f"  -> threshold '{name}' to mark crease vertices, then trace them into feature loops."
        )
        log.info("Dihedral Angle Field: '%s' (%s,%s) max %.2f %s over %d verts",
                 name, reduction, unit, float(v.max()), unit, n)
        return io.NodeOutput(mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackDihedralAngleField": DihedralAngleFieldNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackDihedralAngleField": "Dihedral Angle Field"}
