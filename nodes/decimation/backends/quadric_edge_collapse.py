# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""PyMeshLab quadric edge collapse decimation backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _pymeshlab_quadric_edge_collapse(mesh, target_face_count, quality_threshold,
                                     preserve_boundary, preserve_normal,
                                     preserve_topology, planar_quadric):
    """Quadric edge collapse decimation via PyMeshLab."""
    import pymeshlab

    ms = pymeshlab.MeshSet()
    pml_mesh = pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )
    ms.add_mesh(pml_mesh)

    kwargs = {
        "targetfacenum": target_face_count,
        "qualitythr": quality_threshold,
        "preserveboundary": preserve_boundary,
        "preservenormal": preserve_normal,
        "preservetopology": preserve_topology,
        "planarquadric": planar_quadric,
        "autoclean": True,
    }

    try:
        ms.meshing_decimation_quadric_edge_collapse(**kwargs)
    except AttributeError:
        try:
            ms.simplification_quadric_edge_collapse_decimation(**kwargs)
        except AttributeError:
            raise RuntimeError(
                "PyMeshLab quadric edge collapse filter not available. "
                "Ensure pymeshlab >= 2023.12 is installed."
            )

    out = ms.current_mesh()
    return trimesh_module.Trimesh(
        vertices=out.vertex_matrix(),
        faces=out.face_matrix(),
        process=False,
    )


class DecimateQuadricEdgeCollapseNode(io.ComfyNode):
    """PyMeshLab quadric edge collapse decimation backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackDecimate_QuadricEdgeCollapse",
            display_name="Decimate Quadric Edge Collapse (backend)",
            category="geompack/decimation",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("target_face_count", default=5000, min=4, max=10000000, step=100, tooltip="Target number of output faces."),
                io.Float.Input("quality_threshold", default=0.3, min=0.0, max=1.0, step=0.05, tooltip="Quality threshold for edge collapse. Higher = more conservative, better triangle quality."),
                io.Combo.Input("preserve_boundary", options=["true", "false"], default="true", tooltip="Preserve mesh boundary edges during decimation."),
                io.Combo.Input("preserve_normal", options=["true", "false"], default="true", tooltip="Prevent face normal flips during decimation."),
                io.Combo.Input("preserve_topology", options=["true", "false"], default="true", tooltip="Preserve mesh topology (genus) during decimation."),
                io.Combo.Input("planar_quadric", options=["true", "false"], default="false", tooltip="Add penalty for non-planar faces. Helps preserve flat regions."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="decimated_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, target_face_count=5000, quality_threshold=0.3,
                preserve_boundary="true", preserve_normal="true",
                preserve_topology="true", planar_quadric="false"):
        log.info("Backend: quadric_edge_collapse")
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)
        log.info("Input: %s vertices, %s faces", f"{initial_vertices:,}", f"{initial_faces:,}")

        decimated = _pymeshlab_quadric_edge_collapse(
            trimesh, target_face_count, quality_threshold,
            preserve_boundary == "true",
            preserve_normal == "true",
            preserve_topology == "true",
            planar_quadric == "true",
        )

        if hasattr(trimesh, "metadata") and trimesh.metadata:
            decimated.metadata = trimesh.metadata.copy()
        decimated.metadata["decimation"] = {
            "algorithm": "quadric_edge_collapse",
            "original_vertices": initial_vertices,
            "original_faces": initial_faces,
            "result_vertices": len(decimated.vertices),
            "result_faces": len(decimated.faces),
        }

        face_change = len(decimated.faces) - initial_faces
        face_pct = (face_change / initial_faces) * 100 if initial_faces > 0 else 0

        info = f"""Decimate Results (quadric_edge_collapse):

Target Face Count: {target_face_count:,}
Quality Threshold: {quality_threshold}
Preserve Boundary: {preserve_boundary}
Preserve Normal: {preserve_normal}
Preserve Topology: {preserve_topology}
Planar Quadric: {planar_quadric}

Before:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After:
  Vertices: {len(decimated.vertices):,}
  Faces: {len(decimated.faces):,}
  Reduction: {abs(face_pct):.1f}%
"""
        return io.NodeOutput(decimated, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackDecimate_QuadricEdgeCollapse": DecimateQuadricEdgeCollapseNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackDecimate_QuadricEdgeCollapse": "Decimate Quadric Edge Collapse (backend)"}
