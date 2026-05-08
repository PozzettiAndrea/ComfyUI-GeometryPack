# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""PyMeshLab isotropic remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _pymeshlab_isotropic_remesh(mesh, target_edge_length, iterations=3, adaptive=False, feature_angle=30.0):
    """Apply isotropic remeshing using PyMeshLab."""
    import pymeshlab

    ms = pymeshlab.MeshSet()
    pml_mesh = pymeshlab.Mesh(
        vertex_matrix=mesh.vertices,
        face_matrix=mesh.faces
    )
    ms.add_mesh(pml_mesh)

    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    target_pct = (target_edge_length / bbox_diag) * 100.0

    try:
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=pymeshlab.PercentageValue(target_pct),
            iterations=iterations,
            adaptive=adaptive,
            featuredeg=feature_angle
        )
    except AttributeError:
        try:
            ms.remeshing_isotropic_explicit_remeshing(
                targetlen=pymeshlab.PercentageValue(target_pct),
                iterations=iterations,
                adaptive=adaptive,
                featuredeg=feature_angle
            )
        except AttributeError:
            raise RuntimeError(
                "PyMeshLab meshing filter not available. "
                "On Linux, install OpenGL libraries: sudo apt-get install libgl1-mesa-glx libglu1-mesa"
            )

    remeshed_pml = ms.current_mesh()
    return trimesh_module.Trimesh(
        vertices=remeshed_pml.vertex_matrix(),
        faces=remeshed_pml.face_matrix()
    )


class RemeshPyMeshLabNode(io.ComfyNode):
    """PyMeshLab isotropic remeshing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_PyMeshLab",
            display_name="Remesh PyMeshLab (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("target_edge_length", default=1.00, min=0.001, max=10.0, step=0.01, display_mode="number", tooltip="Target edge length for output triangles. Value is relative to mesh scale."),
                io.Int.Input("iterations", default=3, min=1, max=20, step=1, tooltip="Number of remeshing passes."),
                io.Float.Input("feature_angle", default=30.0, min=0.0, max=180.0, step=1.0, tooltip="Angle threshold (degrees) for feature edge detection."),
                io.Combo.Input("adaptive", options=["true", "false"], default="false", tooltip="Use curvature-adaptive edge lengths."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, target_edge_length=1.0, iterations=3, feature_angle=30.0, adaptive="false"):
        log.info("Backend: pymeshlab_isotropic")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: target_edge_length=%s, iterations=%s, feature_angle=%s, adaptive=%s",
                 target_edge_length, iterations, feature_angle, adaptive)

        remeshed_mesh = _pymeshlab_isotropic_remesh(
            trimesh, target_edge_length, iterations,
            adaptive=(adaptive == "true"), feature_angle=feature_angle
        )

        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'pymeshlab_isotropic',
            'target_edge_length': target_edge_length,
            'iterations': iterations,
            'feature_angle': feature_angle,
            'adaptive': adaptive == "true",
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (PyMeshLab Isotropic): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"edge={target_edge_length}, iter={iterations}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_PyMeshLab": RemeshPyMeshLabNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_PyMeshLab": "Remesh PyMeshLab (backend)"}
