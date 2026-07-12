# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Threshold Mesh by Field node using PyVista.

Deletes the faces/points of a mesh whose scalar field value falls outside (or
inside) a chosen range -- e.g. keep only faces where 'viewed' >= 0.5. Operates on
vertex_attributes (point data) or face_attributes (cell data) via pyvista's
threshold filter, then returns the surviving surface as a trimesh.
"""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _trimesh_to_pyvista(mesh):
    """Convert trimesh.Trimesh to pyvista.PolyData."""
    import pyvista as pv

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    faces_pv = np.column_stack([np.full(len(faces), 3), faces])
    pv_mesh = pv.PolyData(vertices, faces_pv)

    # Transfer vertex_attributes -> point_data
    if hasattr(mesh, 'vertex_attributes'):
        for name, data in mesh.vertex_attributes.items():
            arr = np.array(data)
            if arr.ndim == 1 and len(arr) == len(vertices):
                pv_mesh.point_data[name] = arr.astype(np.float32)

    # Transfer face_attributes -> cell_data
    if hasattr(mesh, 'face_attributes'):
        for name, data in mesh.face_attributes.items():
            arr = np.array(data)
            if arr.ndim == 1 and len(arr) == len(faces):
                pv_mesh.cell_data[name] = arr.astype(np.float32)

    return pv_mesh


def _pyvista_to_trimesh(pv_mesh):
    """Convert pyvista.PolyData back to trimesh.Trimesh."""
    vertices = np.array(pv_mesh.points)

    # Parse pyvista face format: [n, v0, v1, ..., n, v0, v1, ...]
    faces = []
    if pv_mesh.n_faces > 0:
        faces_flat = np.array(pv_mesh.faces)
        i = 0
        while i < len(faces_flat):
            n = faces_flat[i]
            if n == 3:
                faces.append(faces_flat[i + 1:i + 4])
            elif n == 4:
                # Triangulate quads
                faces.append([faces_flat[i + 1], faces_flat[i + 2], faces_flat[i + 3]])
                faces.append([faces_flat[i + 1], faces_flat[i + 3], faces_flat[i + 4]])
            i += n + 1

    if faces:
        faces = np.array(faces, dtype=np.int32)
    else:
        faces = np.zeros((0, 3), dtype=np.int32)

    result = trimesh_module.Trimesh(vertices=vertices, faces=faces, process=False)

    # Transfer point_data -> vertex_attributes
    for name in pv_mesh.point_data.keys():
        data = np.array(pv_mesh.point_data[name])
        if data.ndim == 1 and len(data) == len(vertices):
            result.vertex_attributes[name] = data.astype(np.float32)

    # Transfer cell_data -> face_attributes
    for name in pv_mesh.cell_data.keys():
        data = np.array(pv_mesh.cell_data[name])
        if data.ndim == 1 and len(data) == len(faces):
            result.face_attributes[name] = data.astype(np.float32)

    return result


class ThresholdMeshByFieldNode(io.ComfyNode):
    """Keep/drop mesh elements by a scalar field value (pyvista threshold)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackThresholdByField",
            display_name="Threshold Mesh by Field",
            category="geompack/paraview",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh",
                    tooltip="Mesh carrying a scalar field in vertex_attributes (point) or "
                            "face_attributes (cell) -- e.g. 'viewed', 'curvature'."),
                io.String.Input("field_name", default="viewed",
                    tooltip="Name of the scalar field to threshold on. For a face field added as "
                            "'face.xxx' in viewers, use just 'xxx' here."),
                io.Combo.Input("keep", options=["above", "below", "between", "equal"], default="above",
                    tooltip="Which elements to KEEP: above/below `value`, between `value`..`value_upper`, "
                            "or approximately equal to `value`. The rest are deleted."),
                io.Float.Input("value", default=0.5, min=-1e9, max=1e9, step=0.01,
                    tooltip="Threshold value (lower bound for 'between'). For a 0/1 field like 'viewed', "
                            "keep='above', value=0.5 keeps the viewed faces."),
                io.Float.Input("value_upper", default=1.0, min=-1e9, max=1e9, step=0.01, optional=True,
                    tooltip="Upper bound (only used by keep='between')."),
                io.Combo.Input("association", options=["auto", "cell", "point"], default="auto",
                    tooltip="Which data the field lives on. auto = prefer face/cell data then vertex/point. "
                            "cell = delete faces; point = delete vertices."),
                io.Boolean.Input("all_scalars", default=False, optional=True,
                    tooltip="For point fields: require ALL of a face's vertices to pass (else any one). "
                            "Ignored for cell/face fields."),
                io.Float.Input("equal_tolerance", default=1e-4, min=0.0, max=1e6, step=1e-4, optional=True,
                    tooltip="Half-width of the keep window for keep='equal'."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="summary"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, field_name, keep="above", value=0.5, value_upper=1.0,
                association="auto", all_scalars=False, equal_tolerance=1e-4):
        try:
            import pyvista as pv  # noqa: F401
        except (ImportError, OSError):
            raise ImportError("Threshold Mesh by Field requires pyvista. Install with: pip install pyvista")

        if not hasattr(trimesh, "faces") or len(trimesh.faces) == 0:
            raise ValueError("Threshold Mesh by Field requires a mesh with faces, not a point cloud.")

        field_name = (field_name or "").strip()
        # allow 'face.xxx' field names from the viewers
        if field_name.startswith("face."):
            field_name = field_name[len("face."):]

        pv_mesh = _trimesh_to_pyvista(trimesh)
        in_cell = field_name in pv_mesh.cell_data
        in_point = field_name in pv_mesh.point_data
        if not in_cell and not in_point:
            raise ValueError(
                f"Field '{field_name}' not found. "
                f"point fields={list(pv_mesh.point_data.keys())}, "
                f"face fields={list(pv_mesh.cell_data.keys())}"
            )

        if association == "cell":
            preference = "cell"
        elif association == "point":
            preference = "point"
        else:  # auto: prefer cell/face data (deleting faces is the usual intent)
            preference = "cell" if in_cell else "point"

        n_faces_in = len(trimesh.faces)

        # Build the threshold spec. pyvista keeps cells INSIDE [lo, hi]; invert keeps outside.
        invert = False
        if keep == "above":
            spec = float(value)                       # keeps >= value
        elif keep == "below":
            spec = float(value)
            invert = True                              # keeps < value
        elif keep == "between":
            lo, hi = sorted((float(value), float(value_upper)))
            spec = (lo, hi)
        elif keep == "equal":
            t = float(equal_tolerance)
            spec = (float(value) - t, float(value) + t)
        else:
            raise ValueError(f"Unknown keep mode: {keep}")

        log.info("Threshold '%s' (%s data): keep %s value=%s upper=%s invert=%s",
                 field_name, preference, keep, value, value_upper, invert)

        result_ug = pv_mesh.threshold(
            spec, scalars=field_name, preference=preference,
            invert=invert, all_scalars=bool(all_scalars),
        )
        surf = result_ug.extract_surface()
        result = _pyvista_to_trimesh(surf)

        n_faces_out = len(result.faces)
        summary = (f"Threshold '{field_name}' keep={keep} value={value}"
                   f"{f'..{value_upper}' if keep == 'between' else ''} ({preference}): "
                   f"{n_faces_out}/{n_faces_in} faces kept, {n_faces_in - n_faces_out} deleted.")
        log.info(summary)
        return io.NodeOutput(result, summary)


NODE_CLASS_MAPPINGS = {
    "GeomPackThresholdByField": ThresholdMeshByFieldNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackThresholdByField": "Threshold Mesh by Field",
}
