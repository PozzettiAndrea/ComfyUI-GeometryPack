# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remesh Node - Main backends (pymeshlab, instant_meshes)
"""

import numpy as np
import trimesh as trimesh_module
from typing import Tuple, Optional


def _pymeshlab_isotropic_remesh(
    mesh: trimesh_module.Trimesh,
    target_edge_length: float,
    iterations: int = 3,
    adaptive: bool = False,
    feature_angle: float = 30.0
) -> Tuple[Optional[trimesh_module.Trimesh], str]:
    """Apply isotropic remeshing using PyMeshLab."""
    print(f"[pymeshlab_isotropic_remesh] ===== Starting Isotropic Remeshing =====")
    print(f"[pymeshlab_isotropic_remesh] Input mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"[pymeshlab_isotropic_remesh] Parameters:")
    print(f"[pymeshlab_isotropic_remesh]   target_edge_length: {target_edge_length}")
    print(f"[pymeshlab_isotropic_remesh]   iterations: {iterations}")
    print(f"[pymeshlab_isotropic_remesh]   adaptive: {adaptive}")
    print(f"[pymeshlab_isotropic_remesh]   feature_angle: {feature_angle}")

    try:
        import pymeshlab
    except (ImportError, OSError):
        return None, "pymeshlab is not installed. Install with: pip install pymeshlab"

    if not isinstance(mesh, trimesh_module.Trimesh):
        return None, "Input must be a trimesh.Trimesh object"

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None, "Mesh is empty"

    if target_edge_length <= 0:
        return None, f"Target edge length must be positive, got {target_edge_length}"

    if iterations < 1:
        return None, f"Iterations must be at least 1, got {iterations}"

    try:
        print(f"[pymeshlab_isotropic_remesh] Converting to PyMeshLab format...")
        ms = pymeshlab.MeshSet()

        pml_mesh = pymeshlab.Mesh(
            vertex_matrix=mesh.vertices,
            face_matrix=mesh.faces
        )
        ms.add_mesh(pml_mesh)

        print(f"[pymeshlab_isotropic_remesh] Applying isotropic remeshing...")
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
                return None, (
                    "PyMeshLab meshing filter not available. "
                    "This usually means the libfilter_meshing.so plugin failed to load. "
                    "On Linux, install OpenGL libraries: sudo apt-get install libgl1-mesa-glx libglu1-mesa"
                )

        print(f"[pymeshlab_isotropic_remesh] Converting back to trimesh...")
        remeshed_pml = ms.current_mesh()
        remeshed_mesh = trimesh_module.Trimesh(
            vertices=remeshed_pml.vertex_matrix(),
            faces=remeshed_pml.face_matrix()
        )

        remeshed_mesh.metadata = mesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'pymeshlab_isotropic',
            'target_edge_length': target_edge_length,
            'target_percentage': target_pct,
            'iterations': iterations,
            'adaptive': adaptive,
            'feature_angle': feature_angle,
            'original_vertices': len(mesh.vertices),
            'original_faces': len(mesh.faces),
            'remeshed_vertices': len(remeshed_mesh.vertices),
            'remeshed_faces': len(remeshed_mesh.faces)
        }

        vertex_change = len(remeshed_mesh.vertices) - len(mesh.vertices)
        face_change = len(remeshed_mesh.faces) - len(mesh.faces)
        vertex_pct = (vertex_change / len(mesh.vertices)) * 100 if len(mesh.vertices) > 0 else 0
        face_pct = (face_change / len(mesh.faces)) * 100 if len(mesh.faces) > 0 else 0

        print(f"[pymeshlab_isotropic_remesh] ===== Remeshing Complete =====")
        print(f"[pymeshlab_isotropic_remesh] Results:")
        print(f"[pymeshlab_isotropic_remesh]   Vertices: {len(mesh.vertices)} -> {len(remeshed_mesh.vertices)} ({vertex_change:+d}, {vertex_pct:+.1f}%)")
        print(f"[pymeshlab_isotropic_remesh]   Faces:    {len(mesh.faces)} -> {len(remeshed_mesh.faces)} ({face_change:+d}, {face_pct:+.1f}%)")

        return remeshed_mesh, ""

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error during remeshing: {str(e)}"


class RemeshNode:
    """
    Remesh - Topology-changing remeshing operations (main backends).

    Available backends:
    - pymeshlab_isotropic: PyMeshLab isotropic remeshing (fast)
    - instant_meshes: Field-aligned quad remeshing

    For CGAL isotropic remeshing, use "Remesh CGAL" node.
    For Blender voxel/quadriflow remeshing, use "Remesh Blender" node.
    For GPU-accelerated remeshing, use "Remesh GPU" node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "backend": ([
                    "pymeshlab_isotropic",
                    "instant_meshes",
                ], {
                    "default": "pymeshlab_isotropic",
                    "tooltip": "Remeshing algorithm. pymeshlab=fast isotropic, instant_meshes=field-aligned quads"
                }),
            },
            "optional": {
                # Isotropic params (pymeshlab)
                "target_edge_length": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Target edge length for output triangles. Value is relative to mesh scale.",
                    "visible_when": {"backend": ["pymeshlab_isotropic"]},
                }),
                "iterations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of remeshing passes. More iterations = smoother result, slower processing.",
                    "visible_when": {"backend": ["pymeshlab_isotropic"]},
                }),
                # PyMeshLab-specific
                "feature_angle": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Angle threshold (degrees) for feature edge detection. Edges with dihedral angle greater than this are preserved as sharp creases.",
                    "visible_when": {"backend": ["pymeshlab_isotropic"]},
                }),
                "adaptive": (["true", "false"], {
                    "default": "false",
                    "tooltip": "Use curvature-adaptive edge lengths. Creates smaller triangles in high-curvature areas, larger triangles in flat areas.",
                    "visible_when": {"backend": ["pymeshlab_isotropic"]},
                }),
                # Instant Meshes specific
                "target_vertex_count": ("INT", {
                    "default": 5000,
                    "min": 100,
                    "max": 1000000,
                    "step": 100,
                    "tooltip": "Target vertex count for Instant Meshes output. Creates field-aligned quad-dominant mesh.",
                    "visible_when": {"backend": ["instant_meshes"]},
                }),
                "deterministic": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Use deterministic algorithm for reproducible results. Disable for potentially better quality but non-reproducible output.",
                    "visible_when": {"backend": ["instant_meshes"]},
                }),
                "crease_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Angle threshold (degrees) for preserving sharp/crease edges in Instant Meshes. 0 = no crease preservation.",
                    "visible_when": {"backend": ["instant_meshes"]},
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("remeshed_mesh", "info")
    FUNCTION = "remesh"
    CATEGORY = "geompack/remeshing"
    OUTPUT_NODE = True

    def remesh(self, trimesh, backend, target_edge_length=1.0, iterations=3,
               feature_angle=30.0, adaptive="false",
               target_vertex_count=5000, deterministic="true", crease_angle=0.0):
        """Apply remeshing based on selected backend."""
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        print(f"\n{'='*60}")
        print(f"[Remesh] Backend: {backend}")
        print(f"[Remesh] Input: {initial_vertices:,} vertices, {initial_faces:,} faces")

        if backend == "pymeshlab_isotropic":
            print(f"[Remesh] Parameters: target_edge_length={target_edge_length}, iterations={iterations}, feature_angle={feature_angle}, adaptive={adaptive}")
            remeshed_mesh, info = self._pymeshlab_isotropic(
                trimesh, target_edge_length, iterations, feature_angle, adaptive
            )
        elif backend == "instant_meshes":
            print(f"[Remesh] Parameters: target_vertex_count={target_vertex_count:,}, deterministic={deterministic}, crease_angle={crease_angle}")
            remeshed_mesh, info = self._instant_meshes(
                trimesh, target_vertex_count, deterministic, crease_angle
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        print(f"{'='*60}\n")

        vertex_change = len(remeshed_mesh.vertices) - initial_vertices
        face_change = len(remeshed_mesh.faces) - initial_faces

        print(f"[Remesh] Output: {len(remeshed_mesh.vertices)} vertices ({vertex_change:+d}), "
              f"{len(remeshed_mesh.faces)} faces ({face_change:+d})")

        return {"ui": {"text": [info]}, "result": (remeshed_mesh, info)}

    def _pymeshlab_isotropic(self, trimesh, target_edge_length, iterations, feature_angle, adaptive):
        """PyMeshLab isotropic remeshing."""
        adaptive_bool = (adaptive == "true")
        remeshed_mesh, error = _pymeshlab_isotropic_remesh(
            trimesh, target_edge_length, iterations,
            adaptive=adaptive_bool, feature_angle=feature_angle
        )
        if remeshed_mesh is None:
            raise ValueError(f"PyMeshLab remeshing failed: {error}")

        info = f"""Remesh Results (PyMeshLab Isotropic):

Target Edge Length: {target_edge_length}
Iterations: {iterations}
Feature Angle: {feature_angle}\u00b0
Adaptive: {adaptive}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}
"""
        return remeshed_mesh, info

    def _instant_meshes(self, trimesh, target_vertex_count, deterministic, crease_angle):
        """Instant Meshes field-aligned remeshing."""
        try:
            import pynanoinstantmeshes as pynano
        except ImportError:
            raise ImportError(
                "PyNanoInstantMeshes not installed. Install with: pip install PyNanoInstantMeshes"
            )

        V = trimesh.vertices.astype(np.float32)
        F = trimesh.faces.astype(np.uint32)

        V_out, F_out = pynano.remesh(
            V, F,
            vertex_count=target_vertex_count,
            deterministic=(deterministic == "true"),
            creaseAngle=crease_angle
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=V_out,
            faces=F_out,
            process=False
        )

        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'instant_meshes',
            'target_vertex_count': target_vertex_count,
            'deterministic': deterministic == "true",
            'crease_angle': crease_angle,
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces)
        }

        info = f"""Remesh Results (Instant Meshes):

Target Vertex Count: {target_vertex_count:,}
Deterministic: {deterministic}
Crease Angle: {crease_angle}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}

Instant Meshes creates flow-aligned quad meshes.
"""
        return remeshed_mesh, info


NODE_CLASS_MAPPINGS = {
    "GeomPackRemesh": RemeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemesh": "Remesh",
}
