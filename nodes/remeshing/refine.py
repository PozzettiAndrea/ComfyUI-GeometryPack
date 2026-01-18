# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Refine Mesh Node - Non-destructive mesh refinement operations
"""

import trimesh as trimesh_module


class RefineMeshNode:
    """
    Refine Mesh - Unified non-destructive mesh refinement operations.

    Consolidates mesh refinement operations:
    - decimation: Reduce face count (quadric error metrics)
    - subdivision_loop: Smooth subdivision (Loop algorithm)
    - subdivision_midpoint: Simple midpoint subdivision
    - laplacian_smoothing: Iterative Laplacian smoothing

    Parameters are operation-specific; unused parameters are ignored.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "operation": ([
                    "decimation",
                    "subdivision_loop",
                    "subdivision_midpoint",
                    "laplacian_smoothing"
                ], {"default": "decimation"}),
            },
            "optional": {
                # Decimation
                "target_face_count": ("INT", {
                    "default": 5000,
                    "min": 4,
                    "max": 10000000,
                    "step": 100
                }),
                "decimation_method": (["trimesh", "pymeshlab"], {"default": "trimesh"}),
                # Subdivision
                "subdivision_iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1
                }),
                # Smoothing
                "smoothing_iterations": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "lambda_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("refined_mesh", "info")
    OUTPUT_NODE = True
    FUNCTION = "refine"
    CATEGORY = "geompack/remeshing"

    def refine(self, trimesh, operation, target_face_count=5000, decimation_method="trimesh",
               subdivision_iterations=1, smoothing_iterations=5, lambda_factor=0.5):
        """
        Apply mesh refinement based on selected operation.

        Args:
            trimesh: Input trimesh.Trimesh object
            operation: Refinement operation to apply
            [other params]: Operation-specific parameters

        Returns:
            tuple: (refined_mesh, info_string)
        """
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        print(f"[RefineMesh] Input: {initial_vertices} vertices, {initial_faces} faces")
        print(f"[RefineMesh] Operation: {operation}")

        if operation == "decimation":
            refined_mesh, info = self._decimate(trimesh, target_face_count, decimation_method)
        elif operation == "subdivision_loop":
            refined_mesh, info = self._subdivide(trimesh, subdivision_iterations, "loop")
        elif operation == "subdivision_midpoint":
            refined_mesh, info = self._subdivide(trimesh, subdivision_iterations, "midpoint")
        elif operation == "laplacian_smoothing":
            refined_mesh, info = self._smooth(trimesh, smoothing_iterations, lambda_factor)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        vertex_change = len(refined_mesh.vertices) - initial_vertices
        face_change = len(refined_mesh.faces) - initial_faces

        print(f"[RefineMesh] Output: {len(refined_mesh.vertices)} vertices ({vertex_change:+d}), "
              f"{len(refined_mesh.faces)} faces ({face_change:+d})")

        return {
            "result": (refined_mesh, info),
            "ui": {"text": [info]}
        }

    def _decimate(self, trimesh, target_face_count, method):
        """Decimate mesh to target face count."""
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        if method == "trimesh":
            decimated = trimesh.simplify_quadric_decimation(face_count=target_face_count)
        elif method == "pymeshlab":
            try:
                import pymeshlab
            except ImportError:
                raise ImportError("pymeshlab not installed. Install with: pip install pymeshlab")

            ms = pymeshlab.MeshSet()
            pml_mesh = pymeshlab.Mesh(
                vertex_matrix=trimesh.vertices,
                face_matrix=trimesh.faces
            )
            ms.add_mesh(pml_mesh)

            ms.simplification_quadric_edge_collapse_decimation(
                targetfacenum=target_face_count,
                preserveboundary=True,
                preservenormal=True,
                preservetopology=False
            )

            decimated_pml = ms.current_mesh()
            decimated = trimesh_module.Trimesh(
                vertices=decimated_pml.vertex_matrix(),
                faces=decimated_pml.face_matrix()
            )
        else:
            raise ValueError(f"Unknown decimation method: {method}")

        # Preserve metadata
        decimated.metadata = trimesh.metadata.copy()
        decimated.metadata['decimation'] = {
            'method': method,
            'target_face_count': target_face_count,
            'original_vertices': initial_vertices,
            'original_faces': initial_faces,
            'reduction_ratio': len(decimated.faces) / initial_faces if initial_faces > 0 else 0
        }

        reduction_pct = 100.0 * (initial_faces - len(decimated.faces)) / initial_faces if initial_faces > 0 else 0

        info = f"""Refine Mesh Results (Decimation):

Method: {method}
Target Faces: {target_face_count:,}

Before:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After:
  Vertices: {len(decimated.vertices):,}
  Faces: {len(decimated.faces):,}

Reduction: {reduction_pct:.1f}%
"""
        return decimated, info

    def _subdivide(self, trimesh, iterations, method):
        """Subdivide mesh to increase resolution."""
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        subdivided = trimesh.copy()

        for i in range(iterations):
            if method == "loop":
                subdivided = subdivided.subdivide_loop(iterations=1)
            elif method == "midpoint":
                subdivided = subdivided.subdivide()
            else:
                raise ValueError(f"Unknown subdivision method: {method}")

            print(f"[RefineMesh] Subdivision iteration {i+1}/{iterations}: "
                  f"{len(subdivided.vertices)} vertices, {len(subdivided.faces)} faces")

        # Preserve metadata
        subdivided.metadata = trimesh.metadata.copy()
        subdivided.metadata['subdivision'] = {
            'method': method,
            'iterations': iterations,
            'original_vertices': initial_vertices,
            'original_faces': initial_faces,
            'multiplier': len(subdivided.faces) / initial_faces if initial_faces > 0 else 0
        }

        info = f"""Refine Mesh Results (Subdivision):

Method: {method}
Iterations: {iterations}

Before:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After:
  Vertices: {len(subdivided.vertices):,}
  Faces: {len(subdivided.faces):,}

Multiplier: {len(subdivided.faces) / initial_faces:.2f}x
"""
        return subdivided, info

    def _smooth(self, trimesh, iterations, lambda_factor):
        """Apply Laplacian smoothing."""
        smoothed = trimesh.copy()
        smoothed = trimesh_module.smoothing.filter_laplacian(
            smoothed,
            lamb=lambda_factor,
            iterations=iterations
        )

        # Preserve metadata
        smoothed.metadata = trimesh.metadata.copy()
        smoothed.metadata['smoothing'] = {
            'algorithm': 'laplacian',
            'iterations': iterations,
            'lambda': lambda_factor
        }

        info = f"""Refine Mesh Results (Laplacian Smoothing):

Iterations: {iterations}
Lambda Factor: {lambda_factor}

Vertices: {len(smoothed.vertices):,}
Faces: {len(smoothed.faces):,}

Smoothing reduces surface roughness and noise.
"""
        return smoothed, info
