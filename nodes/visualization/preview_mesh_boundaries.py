# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Preview Mesh Boundaries - interactive VTK.js preview of a mesh whose edges are
thresholded by a per-edge value derived from the TWO faces each edge is shared by:

    edge_value = reduce(face_field[A], face_field[B])

  * face_field = "face_normals" + reduction "angle"  -> dihedral angle (degrees)
  * a scalar face_attributes field + "abs_diff"      -> e.g. PartField difference
  * a vector field + "l2"                            -> Euclidean difference

Edges whose value passes the threshold are written (with the surface) to a .vtp
and shown in the shared VTK.js viewer (viewer_vtk.html). No offscreen rendering.
"""

import logging
import os
import uuid

from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _edge_values(mesh, field_name, reduction):
    """Return (adj_edges[M,2] vertex idx, values[M], used_field_name, available_fields)."""
    import numpy as np

    adj = getattr(mesh, "face_adjacency", None)
    adj_edges = getattr(mesh, "face_adjacency_edges", None)
    avail = sorted((getattr(mesh, "face_attributes", {}) or {}).keys())
    if adj is None or adj_edges is None or len(adj) == 0:
        return np.zeros((0, 2), int), np.zeros(0), field_name, avail

    fattr = getattr(mesh, "face_attributes", {}) or {}
    if field_name in (None, "", "face_normals"):
        field = np.asarray(mesh.face_normals, dtype=np.float64)
        used = "face_normals"
    elif field_name in fattr:
        field = np.asarray(fattr[field_name], dtype=np.float64)
        used = field_name
    else:
        field = np.asarray(mesh.face_normals, dtype=np.float64)
        used = f"face_normals (']{field_name}' not found)"

    A = field[adj[:, 0]]
    B = field[adj[:, 1]]
    red = reduction
    if red == "auto":
        red = "angle" if (A.ndim == 2 and A.shape[1] > 1) else "abs_diff"

    if red == "angle":
        A2, B2 = np.atleast_2d(A), np.atleast_2d(B)
        dot = np.sum(A2 * B2, axis=1)
        na = np.linalg.norm(A2, axis=1)
        nb = np.linalg.norm(B2, axis=1)
        cos = np.clip(dot / (na * nb + 1e-12), -1.0, 1.0)
        vals = np.degrees(np.arccos(cos))
    elif red == "l2":
        vals = np.linalg.norm(np.atleast_2d(A) - np.atleast_2d(B), axis=1)
    else:  # abs_diff (first component if vector)
        a = A.reshape(len(A), -1)[:, 0]
        b = B.reshape(len(B), -1)[:, 0]
        vals = np.abs(a - b)

    return np.asarray(adj_edges, dtype=np.int64), np.asarray(vals, dtype=np.float64), used, avail


class PreviewMeshBoundaries(io.ComfyNode):
    """Threshold mesh edges by an adjacent-face metric (dihedral, etc.) and preview in VTK.js."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackPreviewMeshBoundaries",
            display_name="Preview Mesh Boundaries",
            category="geompack/visualization",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh", tooltip="Mesh to analyze."),
                io.String.Input("face_field", default="face_normals", multiline=False,
                    tooltip="Per-FACE field to compare across each edge. 'face_normals' (+angle) "
                            "gives the dihedral angle. Or any face_attributes key (listed in the "
                            "summary)."),
                io.Combo.Input("reduction", options=["auto", "angle", "abs_diff", "l2"], default="auto",
                    tooltip="How to combine the two faces' values into one per-edge number. "
                            "angle = degrees between vectors (dihedral for normals)."),
                io.Float.Input("threshold", default=30.0, min=0.0, max=100000.0, step=1.0,
                    tooltip="Edges pass when their value meets the threshold (dihedral in degrees)."),
                io.Combo.Input("comparison", options=[">=", "<="], default=">=",
                    tooltip="Show edges whose value is >= (sharp) or <= the threshold."),
                io.Boolean.Input("show_surface", default=True,
                    tooltip="Include the mesh surface (boundary edges drawn on top)."),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, mesh, face_field="face_normals", reduction="auto",
                threshold=30.0, comparison=">=", show_surface=True):
        import numpy as np
        import pyvista as pv
        import folder_paths

        adj_edges, vals, used_field, avail = _edge_values(mesh, face_field, reduction)
        if comparison == "<=":
            passing = vals <= threshold
        else:
            passing = vals >= threshold
        edges = adj_edges[passing] if len(adj_edges) else np.zeros((0, 2), np.int64)
        ev = vals[passing] if len(vals) else np.zeros(0)
        K = int(len(edges))

        points = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(getattr(mesh, "faces", np.zeros((0, 3), int)), dtype=np.int64)
        F = int(len(faces)) if show_surface else 0

        combined = pv.PolyData(points)
        if F:
            combined.faces = np.hstack([np.full((F, 1), 3, np.int64), faces]).ravel()
        if K:
            combined.lines = np.hstack([np.full((K, 1), 2, np.int64), edges]).ravel()

        # Cell data follows VTK's order: verts, LINES, then POLYS.
        if K or F:
            boundary, edge_value = [], []
            if K:
                boundary.append(np.ones(K)); edge_value.append(ev)
            if F:
                boundary.append(np.zeros(F)); edge_value.append(np.zeros(F))
            combined.cell_data["boundary"] = np.concatenate(boundary).astype(np.float32)
            combined.cell_data["edge_value"] = np.concatenate(edge_value).astype(np.float32)

        tmp = folder_paths.get_temp_directory()
        os.makedirs(tmp, exist_ok=True)
        filename = f"gp_boundaries_{uuid.uuid4().hex[:8]}.vtp"
        # ASCII VTP -> reliably parsed by the VTK.js XMLPolyDataReader (incl. Lines).
        combined.save(os.path.join(tmp, filename), binary=False)

        summary = (f"boundaries: {K} edge(s) {comparison} {threshold:g} via "
                   f"reduce={reduction} on '{used_field}'. "
                   f"available face fields: {avail or ['(none)']}")
        log.info("[PreviewMeshBoundaries] %s", summary)
        return io.NodeOutput(ui={
            "mesh_file": [filename],
            "boundary_edges": [K],
            "field_names": [["boundary", "edge_value"]],
            "summary": [summary],
        })


NODE_CLASS_MAPPINGS = {"GeomPackPreviewMeshBoundaries": PreviewMeshBoundaries}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackPreviewMeshBoundaries": "Preview Mesh Boundaries"}
