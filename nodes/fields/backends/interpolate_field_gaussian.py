# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Interpolate Field - Gaussian backend. VTK's vtkPointInterpolator with a Gaussian
kernel (via pyvista): each target vertex averages nearby source values weighted by
exp(-(sharpness*d/R)^2). Smoothing, radius-controlled -- the pyvista `interpolate()`
method. Discrete/label fields never pass through this kernel (shared driver routes
them to surface-aware nearest)."""

import logging

import numpy as np
from comfy_api.latest import io

from ._interpolate_field_common import run_interpolation, parse_field_names

log = logging.getLogger("GeometryPack")


def make_gaussian_interp(radius, sharpness):
    def gaussian_interp(src, P, F):
        import pyvista as pv
        from scipy.spatial import cKDTree

        Vsrc = np.asarray(src.vertices, dtype=np.float64)
        r = float(radius)
        if r <= 0:                                # auto: ~5% of source bbox diagonal
            r = 0.05 * float(np.linalg.norm(Vsrc.max(axis=0) - Vsrc.min(axis=0)))

        src_pd = pv.PolyData(Vsrc)
        for c in range(F.shape[1]):
            src_pd.point_data[f"__f{c}"] = F[:, c].astype(np.float64)

        tgt_pd = pv.PolyData(np.asarray(P, dtype=np.float64))
        res = tgt_pd.interpolate(src_pd, radius=r, sharpness=float(sharpness),
                                 strategy="closest_point")
        out = np.stack([np.asarray(res.point_data[f"__f{c}"]) for c in range(F.shape[1])], axis=1)
        dist, _ = cKDTree(Vsrc).query(P, k=1)
        return out, dist
    return gaussian_interp


class InterpolateFieldGaussianNode(io.ComfyNode):
    """Gaussian-kernel (VTK vtkPointInterpolator) backend for Interpolate Field."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackInterpolateField_Gaussian",
            display_name="Interpolate Field Gaussian (backend)",
            category="geompack/fields",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("field_providing_mesh"),
                io.Custom("TRIMESH").Input("field_target_mesh"),
                io.Boolean.Input("interpolate_all_fields", default=True, optional=True),
                io.String.Input("field_names", default="", optional=True),
                io.Float.Input("radius", default=0.0, min=0.0, max=1e9, step=0.001, optional=True),
                io.Float.Input("sharpness", default=2.0, min=0.1, max=10.0, step=0.1, optional=True),
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
                radius=0.0, sharpness=2.0, max_distance=0.0, fill_value=0.0):
        tgt, info = run_interpolation(field_providing_mesh, field_target_mesh,
                                      parse_field_names(interpolate_all_fields, field_names),
                                      "gaussian", make_gaussian_interp(float(radius), float(sharpness)),
                                      max_distance, fill_value)
        return io.NodeOutput(tgt, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackInterpolateField_Gaussian": InterpolateFieldGaussianNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackInterpolateField_Gaussian": "Interpolate Field Gaussian (backend)"}
