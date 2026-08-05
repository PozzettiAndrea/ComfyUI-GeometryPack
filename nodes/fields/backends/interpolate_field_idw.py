# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Interpolate Field - IDW backend. Inverse-distance weighting over the k nearest
source vertices (meshless -- also fine when the source is effectively a point
cloud). Smooths; locality controlled by k and the distance exponent."""

import logging

import numpy as np
from comfy_api.latest import io

from ._interpolate_field_common import run_interpolation, parse_field_names

log = logging.getLogger("GeometryPack")


def make_idw_interp(idw_k, idw_power):
    def idw_interp(src, P, F):
        from scipy.spatial import cKDTree
        Vsrc = np.asarray(src.vertices, dtype=np.float64)
        tree = cKDTree(Vsrc)
        k = int(max(1, min(idw_k, len(Vsrc))))
        d, idx = tree.query(P, k=k)
        d = np.atleast_2d(d).reshape(len(P), k)
        idx = np.atleast_2d(idx).reshape(len(P), k)
        w = 1.0 / (np.power(d, idw_power) + 1e-12)             # (n,k)
        w /= w.sum(axis=1, keepdims=True)
        out = (w[:, :, None] * F[idx].astype(np.float64)).sum(axis=1)
        return out, d[:, 0]
    return idw_interp


class InterpolateFieldIDWNode(io.ComfyNode):
    """IDW (k-nearest inverse-distance) backend for Interpolate Field."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackInterpolateField_IDW",
            display_name="Interpolate Field IDW (backend)",
            category="geompack/fields",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("field_providing_mesh"),
                io.Custom("TRIMESH").Input("field_target_mesh"),
                io.Boolean.Input("interpolate_all_fields", default=True, optional=True),
                io.String.Input("field_names", default="", optional=True),
                io.Int.Input("idw_k", default=8, min=1, max=128, step=1, optional=True),
                io.Float.Input("idw_power", default=2.0, min=0.1, max=8.0, step=0.1, optional=True),
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
                idw_k=8, idw_power=2.0, max_distance=0.0, fill_value=0.0):
        tgt, info = run_interpolation(field_providing_mesh, field_target_mesh,
                                      parse_field_names(interpolate_all_fields, field_names),
                                      "idw", make_idw_interp(int(idw_k), float(idw_power)),
                                      max_distance, fill_value)
        return io.NodeOutput(tgt, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackInterpolateField_IDW": InterpolateFieldIDWNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackInterpolateField_IDW": "Interpolate Field IDW (backend)"}
