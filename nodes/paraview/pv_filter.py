# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
ParaView/VTK filter node using PyVista.

Applies VTK analysis filters to meshes, producing scalar fields
for visualization in the Preview Mesh (VTK) viewer.
"""

import logging
import numpy as np
import trimesh
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

    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

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


class ParaViewFilterNode(io.ComfyNode):
    """
    Apply VTK/ParaView analysis filters to meshes using PyVista.

    Produces scalar fields for visualization in Preview Mesh (fields mode).
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackParaViewFilter",
            display_name="ParaView Filter",
            category="geompack/paraview",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Combo.Input("filter_type", options=[
                    "curvature_gaussian",
                    "curvature_mean",
                    "cell_sizes",
                    "elevation",
                    "feature_edges",
                    "warp_by_scalar",
                ], default="curvature_gaussian"),
                io.Combo.Input("axis", options=["X", "Y", "Z"], default="Z", tooltip="Axis for elevation or warp direction", optional=True),
                io.Float.Input("factor", default=1.0, min=-100.0, max=100.0, step=0.1, tooltip="Scale factor for warp_by_scalar", optional=True),
                io.String.Input("scalar_field", default="", tooltip="Vertex attribute name for warp_by_scalar", optional=True),
                io.Float.Input("angle", default=30.0, min=0.0, max=180.0, step=1.0, tooltip="Feature edge angle threshold (degrees)", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="filtered_mesh"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, filter_type, axis="Z", factor=1.0, scalar_field="", angle=30.0):
        """
        Apply a VTK filter via PyVista.

        Args:
            trimesh: Input trimesh.Trimesh object
            filter_type: Which filter to apply
            axis: Axis for elevation/warp
            factor: Scale factor for warp
            scalar_field: Vertex attribute name for warp_by_scalar
            angle: Feature edge angle threshold

        Returns:
            tuple: (filtered_mesh,)
        """
        try:
            import pyvista as pv
        except (ImportError, OSError):
            raise ImportError(
                "ParaView Filter requires pyvista. Install with: pip install pyvista"
            )

        # Validate input
        if not hasattr(trimesh, 'faces') or len(trimesh.faces) == 0:
            raise ValueError("ParaView Filter requires a mesh with faces, not a point cloud")

        log.info("Applying '%s' to mesh: %d vertices, %d faces",
                 filter_type, len(trimesh.vertices), len(trimesh.faces))

        # Convert to pyvista
        pv_mesh = _trimesh_to_pyvista(trimesh)

        # Apply filter
        if filter_type == "curvature_gaussian":
            pv_result = pv_mesh.curvature('gaussian')
            # curvature() returns a new mesh with 'Gauss_Curvature' in point_data
            result = _pyvista_to_trimesh(pv_result)
            # Rename to friendlier name
            if 'Gauss_Curvature' in result.vertex_attributes:
                result.vertex_attributes['curvature'] = result.vertex_attributes.pop('Gauss_Curvature')
            log.info("Computed Gaussian curvature")

        elif filter_type == "curvature_mean":
            pv_result = pv_mesh.curvature('mean')
            result = _pyvista_to_trimesh(pv_result)
            if 'Mean_Curvature' in result.vertex_attributes:
                result.vertex_attributes['curvature'] = result.vertex_attributes.pop('Mean_Curvature')
            log.info("Computed mean curvature")

        elif filter_type == "cell_sizes":
            pv_result = pv_mesh.compute_cell_sizes(length=False, volume=False, area=True)
            result = _pyvista_to_trimesh(pv_result)
            log.info("Computed cell sizes (face areas)")

        elif filter_type == "elevation":
            axis_map = {"X": 0, "Y": 1, "Z": 2}
            ax = axis_map.get(axis, 2)
            bounds = pv_mesh.bounds
            low_point = [0.0, 0.0, 0.0]
            high_point = [0.0, 0.0, 0.0]
            low_point[ax] = bounds[ax * 2]
            high_point[ax] = bounds[ax * 2 + 1]
            pv_result = pv_mesh.elevation(
                low_point=low_point,
                high_point=high_point,
                set_active_scalars=True
            )
            result = _pyvista_to_trimesh(pv_result)
            log.info("Computed elevation along %s axis", axis)

        elif filter_type == "feature_edges":
            pv_result = pv_mesh.extract_feature_edges(
                feature_angle=angle,
                boundary_edges=True,
                non_manifold_edges=True,
                manifold_edges=False,
            )
            if pv_result.n_points == 0:
                log.warning("No feature edges found at angle=%s", angle)
                # Return original mesh with an empty marker attribute
                result = trimesh.copy()
                result.vertex_attributes['feature_edge'] = np.zeros(len(result.vertices), dtype=np.float32)
            else:
                result = _pyvista_to_trimesh(pv_result)
                log.info("Extracted feature edges: %d points, %d lines", pv_result.n_points, pv_result.n_lines)

        elif filter_type == "warp_by_scalar":
            if not scalar_field:
                available = list(trimesh.vertex_attributes.keys()) if hasattr(trimesh, 'vertex_attributes') else []
                raise ValueError(
                    f"warp_by_scalar requires a scalar_field name. "
                    f"Available vertex attributes: {available}"
                )
            if scalar_field not in pv_mesh.point_data:
                available = list(pv_mesh.point_data.keys())
                raise ValueError(
                    f"scalar_field '{scalar_field}' not found. "
                    f"Available: {available}"
                )
            pv_result = pv_mesh.warp_by_scalar(
                scalars=scalar_field,
                factor=factor,
            )
            result = _pyvista_to_trimesh(pv_result)
            log.info("Warped by '%s' with factor=%s", scalar_field, factor)

        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        log.info("Result: %d vertices, %d faces", len(result.vertices), len(result.faces))

        # List output fields
        v_attrs = list(result.vertex_attributes.keys()) if hasattr(result, 'vertex_attributes') else []
        f_attrs = list(result.face_attributes.keys()) if hasattr(result, 'face_attributes') else []
        if v_attrs:
            log.info("Vertex attributes: %s", v_attrs)
        if f_attrs:
            log.info("Face attributes: %s", f_attrs)

        return io.NodeOutput(result)


NODE_CLASS_MAPPINGS = {
    "GeomPackParaViewFilter": ParaViewFilterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackParaViewFilter": "ParaView Filter",
}
