# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Interpolate Field - RBF backend. scipy thin-plate-spline RBFInterpolator over
the source vertices (local, using `rbf_neighbors` nearest centers). Smoothest
result; good for sparse or noisy sources."""

import logging

import numpy as np
from comfy_api.latest import io

from ._interpolate_field_common import run_interpolation, parse_field_names

log = logging.getLogger("GeometryPack")


def make_rbf_interp(rbf_neighbors):
    def rbf_interp(src, P, F):
        from scipy.spatial import cKDTree
        from scipy.interpolate import RBFInterpolator
        Vsrc = np.asarray(src.vertices, dtype=np.float64)
        nbr = int(min(rbf_neighbors, len(Vsrc)))
        rbf = RBFInterpolator(Vsrc, F.astype(np.float64),
                              neighbors=nbr, kernel="thin_plate_spline")
        out = rbf(P)
        dist, _ = cKDTree(Vsrc).query(P, k=1)
        return out, dist
    return rbf_interp


class InterpolateFieldRBFNode(io.ComfyNode):
    """RBF (thin-plate spline) backend for Interpolate Field."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackInterpolateField_RBF",
            display_name="Interpolate Field RBF (backend)",
            category="geompack/fields",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("field_providing_mesh"),
                io.Custom("TRIMESH").Input("field_target_mesh"),
                io.Boolean.Input("interpolate_all_fields", default=True, optional=True),
                io.String.Input("field_names", default="", optional=True),
                io.Int.Input("rbf_neighbors", default=32, min=4, max=256, step=4, optional=True),
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
                rbf_neighbors=32, max_distance=0.0, fill_value=0.0):
        tgt, info = run_interpolation(field_providing_mesh, field_target_mesh,
                                      parse_field_names(interpolate_all_fields, field_names),
                                      "rbf", make_rbf_interp(int(rbf_neighbors)),
                                      max_distance, fill_value)
        return io.NodeOutput(tgt, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackInterpolateField_RBF": InterpolateFieldRBFNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackInterpolateField_RBF": "Interpolate Field RBF (backend)"}
