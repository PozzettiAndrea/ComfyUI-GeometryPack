# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""PyMeshLab vertex clustering decimation backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _pymeshlab_vertex_clustering(mesh, threshold_percentage):
    """Vertex clustering decimation via PyMeshLab."""
    import pymeshlab

    ms = pymeshlab.MeshSet()
    pml_mesh = pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )
    ms.add_mesh(pml_mesh)

    try:
        ms.meshing_decimation_clustering(
            threshold=pymeshlab.PercentageValue(threshold_percentage),
        )
    except AttributeError:
        try:
            ms.simplification_clustering_decimation(
                threshold=pymeshlab.PercentageValue(threshold_percentage),
            )
        except AttributeError:
            raise RuntimeError(
                "PyMeshLab vertex clustering filter not available. "
                "Ensure pymeshlab >= 2023.12 is installed."
            )

    out = ms.current_mesh()
    return trimesh_module.Trimesh(
        vertices=out.vertex_matrix(),
        faces=out.face_matrix(),
        process=False,
    )


class DecimateVertexClusteringNode(io.ComfyNode):
    """PyMeshLab vertex clustering decimation backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackDecimate_VertexClustering",
            display_name="Decimate Vertex Clustering (backend)",
            category="geompack/decimation",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("cluster_threshold", default=1.0, min=0.1, max=10.0, step=0.1, tooltip="Clustering cell size as percentage of bounding box diagonal. Larger = more aggressive reduction."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="decimated_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, cluster_threshold=1.0):
        log.info("Backend: vertex_clustering")
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)
        log.info("Input: %s vertices, %s faces", f"{initial_vertices:,}", f"{initial_faces:,}")

        decimated = _pymeshlab_vertex_clustering(trimesh, cluster_threshold)

        if hasattr(trimesh, "metadata") and trimesh.metadata:
            decimated.metadata = trimesh.metadata.copy()
        decimated.metadata["decimation"] = {
            "algorithm": "vertex_clustering",
            "original_vertices": initial_vertices,
            "original_faces": initial_faces,
            "result_vertices": len(decimated.vertices),
            "result_faces": len(decimated.faces),
        }

        face_change = len(decimated.faces) - initial_faces
        face_pct = (face_change / initial_faces) * 100 if initial_faces > 0 else 0

        info = f"""Decimate Results (vertex_clustering):

Cluster Threshold: {cluster_threshold}%

Before:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After:
  Vertices: {len(decimated.vertices):,}
  Faces: {len(decimated.faces):,}
  Reduction: {abs(face_pct):.1f}%
"""
        return io.NodeOutput(decimated, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackDecimate_VertexClustering": DecimateVertexClusteringNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackDecimate_VertexClustering": "Decimate Vertex Clustering (backend)"}
