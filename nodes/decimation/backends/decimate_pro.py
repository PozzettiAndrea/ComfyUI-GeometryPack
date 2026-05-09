# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""PyVista/VTK DecimatePro backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _pyvista_decimate_pro(mesh, target_reduction, preserve_topology, feature_angle):
    """VTK DecimatePro via PyVista."""
    import pyvista as pv

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces_raw = np.asarray(mesh.faces, dtype=np.int32)

    # PyVista face format: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    n_faces = len(faces_raw)
    pv_faces = np.empty((n_faces, 4), dtype=np.int32)
    pv_faces[:, 0] = 3
    pv_faces[:, 1:] = faces_raw
    pv_faces = pv_faces.ravel()

    pv_mesh = pv.PolyData(vertices, pv_faces)

    decimated_pv = pv_mesh.decimate_pro(
        target_reduction,
        preserve_topology=preserve_topology,
        feature_angle=feature_angle,
    )

    out_verts = np.array(decimated_pv.points, dtype=np.float64)
    out_faces_pv = decimated_pv.faces.reshape(-1, 4)[:, 1:]

    return trimesh_module.Trimesh(
        vertices=out_verts,
        faces=out_faces_pv,
        process=False,
    )


class DecimateProNode(io.ComfyNode):
    """PyVista/VTK DecimatePro backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackDecimate_DecimatePro",
            display_name="Decimate Pro (backend)",
            category="geompack/decimation",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("target_reduction", default=0.5, min=0.01, max=0.99, step=0.01, tooltip="Fraction of faces to REMOVE. 0.5 = reduce to ~50%% of original faces, 0.9 = reduce to ~10%% of original."),
                io.Float.Input("feature_angle", default=15.0, min=0.0, max=180.0, step=1.0, tooltip="Feature angle threshold (degrees). Edges with dihedral angle above this are preserved."),
                io.Combo.Input("preserve_topology", options=["true", "false"], default="true", tooltip="Preserve mesh topology (genus) during decimation."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="decimated_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, target_reduction=0.5, feature_angle=15.0, preserve_topology="true"):
        log.info("Backend: decimate_pro")
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)
        log.info("Input: %s vertices, %s faces", f"{initial_vertices:,}", f"{initial_faces:,}")

        decimated = _pyvista_decimate_pro(
            trimesh, target_reduction,
            preserve_topology == "true",
            feature_angle,
        )

        if hasattr(trimesh, "metadata") and trimesh.metadata:
            decimated.metadata = trimesh.metadata.copy()
        decimated.metadata["decimation"] = {
            "algorithm": "decimate_pro",
            "original_vertices": initial_vertices,
            "original_faces": initial_faces,
            "result_vertices": len(decimated.vertices),
            "result_faces": len(decimated.faces),
        }

        face_change = len(decimated.faces) - initial_faces
        face_pct = (face_change / initial_faces) * 100 if initial_faces > 0 else 0

        info = f"""Decimate Results (decimate_pro):

Target Reduction: {target_reduction:.0%}
Preserve Topology: {preserve_topology}
Feature Angle: {feature_angle}\u00b0

Before:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After:
  Vertices: {len(decimated.vertices):,}
  Faces: {len(decimated.faces):,}
  Reduction: {abs(face_pct):.1f}%
"""
        return io.NodeOutput(decimated, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackDecimate_DecimatePro": DecimateProNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackDecimate_DecimatePro": "Decimate Pro (backend)"}
