# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""PyMeshLab isotropic remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _pymeshlab_isotropic_remesh(mesh, target_edge_length, iterations=3, adaptive=False,
                                feature_angle=30.0, reproject=True):
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
            featuredeg=feature_angle,
            reprojectflag=reproject
        )
    except AttributeError:
        try:
            ms.remeshing_isotropic_explicit_remeshing(
                targetlen=pymeshlab.PercentageValue(target_pct),
                iterations=iterations,
                adaptive=adaptive,
                featuredeg=feature_angle,
                reprojectflag=reproject
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
                io.Float.Input("target_edge_length", default=1.00, min=0.0001, max=10.0, step=0.0001, display_mode="number", tooltip="Target edge length for output triangles, in world units (relative to mesh scale). Used only when target_vertices and target_faces are both 0."),
                io.Int.Input("target_vertices", default=0, min=0, max=20000000, step=100, tooltip="Target output vertex count (0 = off). Back-solves the edge length from the mesh area; overrides target_edge_length. Approximate."),
                io.Int.Input("target_faces", default=0, min=0, max=40000000, step=100, tooltip="Target output face count (0 = off). Back-solves the edge length from the mesh area; overrides target_vertices and target_edge_length. Approximate."),
                io.Int.Input("iterations", default=3, min=1, max=20, step=1, tooltip="Number of remeshing passes."),
                io.Float.Input("feature_angle", default=30.0, min=0.0, max=180.0, step=1.0, tooltip="Angle threshold (degrees) for feature/crease edge detection -- edges sharper than this are preserved. Lower = preserve more edges; 180 = none."),
                io.Combo.Input("adaptive", options=["true", "false"], default="false", tooltip="Use curvature-adaptive edge lengths."),
                io.Combo.Input("reproject", options=["true", "false"], default="true", tooltip="Reproject vertices back onto the original surface after each iteration (Botsch back-projection). true = stay faithful to the input surface (recommended); false = pure tangential smoothing, which lets vertices drift off the surface."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, target_edge_length=1.0, target_vertices=0, target_faces=0,
                iterations=3, feature_angle=30.0, adaptive="false", reproject="true"):
        import math

        # Resolve edge length from a target vertex/face count if given (0 = off). For an
        # isotropic mesh of equilateral triangles (edge e, area A): #faces ~= 4A/(sqrt(3)e^2)
        # and #verts ~= #faces/2 ~= 2A/(sqrt(3)e^2). So e = sqrt(4A/(sqrt(3)*F)) for a target
        # face count, or sqrt(2A/(sqrt(3)*V)) for a target vertex count. Approximate.
        # Priority: faces > vertices > edge_length.
        edge = float(target_edge_length)
        note = f"edge={edge:.4g}"
        try:
            area = float(trimesh.area)
        except Exception:
            area = 0.0
        tf = int(target_faces or 0)
        tv = int(target_vertices or 0)
        if tf > 0 and area > 0.0:
            edge = math.sqrt(4.0 * area / (math.sqrt(3.0) * tf))
            note = f"target_faces={tf:,} -> edge~={edge:.4g} (area={area:.4g})"
        elif tv > 0 and area > 0.0:
            edge = math.sqrt(2.0 * area / (math.sqrt(3.0) * tv))
            note = f"target_vertices={tv:,} -> edge~={edge:.4g} (area={area:.4g})"

        log.info("Backend: pymeshlab_isotropic")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: %s, iterations=%s, feature_angle=%s, adaptive=%s, reproject=%s",
                 note, iterations, feature_angle, adaptive, reproject)

        remeshed_mesh = _pymeshlab_isotropic_remesh(
            trimesh, edge, iterations,
            adaptive=(adaptive == "true"), feature_angle=feature_angle,
            reproject=(reproject == "true")
        )

        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'pymeshlab_isotropic',
            'target_edge_length': edge,
            'target_vertices': tv,
            'target_faces': tf,
            'iterations': iterations,
            'feature_angle': feature_angle,
            'adaptive': adaptive == "true",
            'reproject': reproject == "true",
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (PyMeshLab Isotropic): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"{note}, iter={iterations}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_PyMeshLab": RemeshPyMeshLabNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_PyMeshLab": "Remesh PyMeshLab (backend)"}
