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
from comfy_api.latest import io

from .pv_filter import _trimesh_to_pyvista, _pyvista_to_trimesh

log = logging.getLogger("geometrypack")


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
