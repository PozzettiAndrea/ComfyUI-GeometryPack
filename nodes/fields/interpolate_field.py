# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors
"""Interpolate Field - unified frontend with backend selection.

Transfers per-vertex / per-face attributes from one mesh onto another (e.g. after
remeshing), each algorithm a separate hidden node dispatched through GraphBuilder:
- barycentric: closest point on a source triangle + barycentric blend (second-order;
  exact when the meshes coincide -- the right default after remeshing the same object)
- nearest: surface-aware nearest (dominant corner of the closest triangle)
- idw: inverse-distance weighting over k nearest source vertices (meshless)
- rbf: scipy thin-plate-spline RBF (smoothest; sparse/noisy sources)
- gaussian: VTK vtkPointInterpolator Gaussian kernel via pyvista (radius-controlled
  smoothing -- pyvista's `interpolate()`)

Hard rules (enforced in the shared backend driver, regardless of selection):
integer/boolean fields always transfer by nearest -- blending labels produces
nonexistent ids; per-face fields always transfer by nearest source face.
"""
from comfy_api.latest import io


class InterpolateFieldNode(io.ComfyNode):
    """Transfer scalar/vector fields from one mesh onto another -- pick a backend."""

    BACKEND_MAP = {
        "barycentric": "GeomPackInterpolateField_Barycentric",
        "nearest": "GeomPackInterpolateField_Nearest",
        "idw": "GeomPackInterpolateField_IDW",
        "rbf": "GeomPackInterpolateField_RBF",
        "gaussian": "GeomPackInterpolateField_Gaussian",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackInterpolateField",
            display_name="Interpolate Field",
            category="geompack/fields",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.DynamicCombo.Input("backend",
                    tooltip="Which interpolation algorithm transfers the field:\n"
                            "- barycentric: closest point on a source TRIANGLE + barycentric "
                            "blend of its corners. Second-order accurate, exact when the meshes "
                            "coincide -- the default for resampling a continuous field after "
                            "remeshing the same object.\n"
                            "- nearest: surface-aware nearest (dominant corner of the closest "
                            "triangle). The right call for labels -- and what EVERY backend "
                            "automatically falls back to for integer/boolean fields, since "
                            "blending label ids is meaningless.\n"
                            "- idw: inverse-distance weighting over the k nearest source "
                            "vertices. Meshless (works on point-cloud-like sources), smooths.\n"
                            "- rbf: thin-plate-spline radial basis functions. Smoothest; best "
                            "for sparse or noisy sources.\n"
                            "- gaussian: VTK point interpolator with a Gaussian kernel "
                            "(pyvista's interpolate()). Radius-controlled smoothing.",
                    options=[
                        io.DynamicCombo.Option("barycentric", []),
                        io.DynamicCombo.Option("nearest", []),
                        io.DynamicCombo.Option("idw", [
                            io.Int.Input("idw_k", default=8, min=1, max=128, step=1,
                                tooltip="Number of nearest source vertices averaged per target vertex."),
                            io.Float.Input("idw_power", default=2.0, min=0.1, max=8.0, step=0.1,
                                tooltip="Inverse-distance exponent: higher = more local (less smoothing)."),
                        ]),
                        io.DynamicCombo.Option("rbf", [
                            io.Int.Input("rbf_neighbors", default=32, min=4, max=256, step=4,
                                tooltip="Number of nearest RBF centers used per evaluation "
                                        "(scipy RBFInterpolator neighbors)."),
                        ]),
                        io.DynamicCombo.Option("gaussian", [
                            io.Float.Input("radius", default=0.0, min=0.0, max=1e9, step=0.001,
                                tooltip="Gaussian kernel radius in world units. 0 = automatic "
                                        "(~5% of the source bounding-box diagonal)."),
                            io.Float.Input("sharpness", default=2.0, min=0.1, max=10.0, step=0.1,
                                tooltip="Gaussian falloff within the radius: higher = tighter "
                                        "kernel (less smoothing)."),
                        ]),
                    ]),
                io.Custom("TRIMESH").Input("field_providing_mesh",
                    tooltip="Source mesh carrying the field(s) to transfer (in its "
                            "vertex_attributes / face_attributes)."),
                io.Custom("TRIMESH").Input("field_target_mesh",
                    tooltip="Target mesh to receive the interpolated field(s). Its geometry is "
                            "unchanged; only attributes are added."),
                io.Boolean.Input("interpolate_all_fields", default=True,
                    tooltip="ON (default): transfer EVERY vertex and face attribute found on the "
                            "source mesh -- field_names is ignored. Turn OFF to transfer only "
                            "the fields you list in field_names."),
                io.String.Input("field_names", default="", optional=True,
                    tooltip="Only used when interpolate_all_fields is OFF. Comma-separated list "
                            "of field names to transfer, e.g.:\n"
                            "    pressure, length\n"
                            "or quoted (quotes are stripped, both styles work):\n"
                            "    \"pressure\",\"length\"\n"
                            "Names are matched against the source mesh's vertex_attributes "
                            "first, then face_attributes. To address a FACE field explicitly "
                            "(e.g. when a vertex field shares its name, or as shown in viewer "
                            "dropdowns), prefix it with 'face.':\n"
                            "    \"pressure\",\"face.part_id\"\n"
                            "Names not present on the source are reported as NOT FOUND in the "
                            "info output (with the list of available fields) and skipped."),
                io.Float.Input("max_distance", default=0.0, min=0.0, max=1e9, step=0.001,
                    tooltip="World-unit cap on the source->target projection distance. Target "
                            "points whose closest source point is farther than this are "
                            "out-of-range and set to fill_value (prevents grabbing values "
                            "across holes / off the surface). 0 = no limit."),
                io.Float.Input("fill_value", default=0.0, min=-1e12, max=1e12, step=0.1, advanced=True,
                    tooltip="Value written where a target point exceeds max_distance (no nearby source)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, backend=None, field_providing_mesh=None, field_target_mesh=None,
                interpolate_all_fields=True, field_names="", max_distance=0.0, fill_value=0.0):
        from comfy_execution.graph_utils import GraphBuilder
        if cls.SCHEMA is None:
            cls.GET_SCHEMA()
        if backend is None:
            backend = {"backend": "barycentric"}
        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]
        kwargs = {
            "field_providing_mesh": field_providing_mesh,
            "field_target_mesh": field_target_mesh,
            "interpolate_all_fields": interpolate_all_fields,
            "field_names": field_names,
            "max_distance": max_distance,
            "fill_value": fill_value,
        }
        for k, v in backend.items():
            if k == "backend":
                continue
            kwargs[k] = v
        graph = GraphBuilder()
        backend_node = graph.node(node_id, **kwargs)
        return {
            "result": (backend_node.out(0), backend_node.out(1)),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {"GeomPackInterpolateField": InterpolateFieldNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackInterpolateField": "Interpolate Field"}
