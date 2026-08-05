# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Interpolate Field - nearest backend. Surface-aware nearest: closest point on a
source triangle, then the value of the dominant-barycentric corner vertex. The
correct choice for integer / label fields (segmentation, cad_face_id, materials)
-- and what every other backend silently falls back to for such fields."""

import logging

from comfy_api.latest import io

from ._interpolate_field_common import nearest_vertex_interp, run_interpolation, parse_field_names

log = logging.getLogger("GeometryPack")


class InterpolateFieldNearestNode(io.ComfyNode):
    """Nearest (surface-aware) backend for Interpolate Field."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackInterpolateField_Nearest",
            display_name="Interpolate Field Nearest (backend)",
            category="geompack/fields",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("field_providing_mesh"),
                io.Custom("TRIMESH").Input("field_target_mesh"),
                io.Boolean.Input("interpolate_all_fields", default=True, optional=True),
                io.String.Input("field_names", default="", optional=True),
                io.Float.Input("max_distance", default=0.0, min=0.0, max=1e9, step=0.001, optional=True),
                io.Float.Input("fill_value", default=0.0, min=-1e12, max=1e12, step=0.1, optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, field_providing_mesh=None, field_target_mesh=None, interpolate_all_fields=True, field_names="",
                max_distance=0.0, fill_value=0.0):
        tgt, info = run_interpolation(field_providing_mesh, field_target_mesh,
                                      parse_field_names(interpolate_all_fields, field_names),
                                      "nearest", nearest_vertex_interp,
                                      max_distance, fill_value)
        return io.NodeOutput(tgt, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackInterpolateField_Nearest": InterpolateFieldNearestNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackInterpolateField_Nearest": "Interpolate Field Nearest (backend)"}
