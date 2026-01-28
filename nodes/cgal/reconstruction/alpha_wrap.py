# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Alpha Wrap Node - Shrink wrap mesh generation using CGAL's Alpha Wrap algorithm.

Creates a watertight mesh that tightly wraps around input geometry.
Useful for point clouds, non-manifold meshes, or polygon soups.
"""

import numpy as np
import trimesh


class AlphaWrapNode:
    """
    Alpha Wrap - Generate a watertight shrink-wrapped mesh.

    Uses CGAL's Alpha Wrap algorithm (via PyMeshLab) to create a
    tight-fitting watertight mesh around input geometry.

    Works with:
    - Point clouds
    - Non-manifold meshes
    - Meshes with holes
    - Polygon soups (overlapping/intersecting geometry)

    Parameters:
    - alpha: Controls wrap tightness. Smaller = tighter wrap, more detail.
             Relative to bounding box diagonal (0.01 = 1% of bbox diagonal).
    - offset: Surface offset distance. Smaller = closer to original surface.
              Relative to bounding box diagonal.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mesh": ("TRIMESH",),
            },
            "optional": {
                "alpha_percent": ("FLOAT", {
                    "default": 0.04,
                    "min": 0.001,
                    "max": 50.0,
                    "step": 0.01,
                    "tooltip": "Wrap tightness as % of bounding box diagonal. Smaller = tighter wrap with more detail, but slower."
                }),
                "offset_percent": ("FLOAT", {
                    "default": 1.1,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Surface offset as % of bounding box diagonal. Smaller = closer to original surface."
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("wrapped_mesh", "report")
    FUNCTION = "alpha_wrap"
    CATEGORY = "geompack/reconstruction"

    def alpha_wrap(self, input_mesh, alpha_percent=0.04, offset_percent=1.1):
        """
        Generate alpha-wrapped mesh.

        Args:
            input_mesh: Input trimesh.Trimesh object (mesh or point cloud)
            alpha_percent: Alpha value as percentage of bbox diagonal
            offset_percent: Offset value as percentage of bbox diagonal

        Returns:
            tuple: (wrapped_trimesh, report_string)
        """
        try:
            import pymeshlab
        except ImportError:
            raise ImportError(
                "Alpha Wrap requires PyMeshLab with CGAL support.\n"
                "Install with: pip install pymeshlab"
            )

        # Get input stats
        vertices = np.asarray(input_mesh.vertices, dtype=np.float64)
        input_vertex_count = len(vertices)

        # Check if input is point cloud or mesh
        is_point_cloud = (
            not hasattr(input_mesh, 'faces') or
            input_mesh.faces is None or
            len(input_mesh.faces) == 0
        )

        if is_point_cloud:
            raise ValueError(
                "Alpha Wrap requires a mesh with faces (triangle soup), not a point cloud.\n"
                "For point clouds, first use 'Reconstruct Surface' node (e.g., Poisson or Ball Pivoting) "
                "to create a mesh, then apply Alpha Wrap."
            )

        input_face_count = len(input_mesh.faces)
        input_type = "mesh"

        # Compute bounding box diagonal for relative parameters
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)

        # Convert percentages to absolute values
        alpha = (alpha_percent / 100.0) * bbox_diagonal
        offset = (offset_percent / 100.0) * bbox_diagonal

        print(f"\n{'='*60}")
        print(f"[AlphaWrap] Input: {input_vertex_count:,} vertices, {input_face_count:,} faces ({input_type})")
        print(f"[AlphaWrap] Bounding box diagonal: {bbox_diagonal:.4f}")
        print(f"[AlphaWrap] Alpha: {alpha:.6f} ({alpha_percent}% of bbox)")
        print(f"[AlphaWrap] Offset: {offset:.6f} ({offset_percent}% of bbox)")
        print(f"{'='*60}\n")

        # Create PyMeshLab MeshSet
        ms = pymeshlab.MeshSet()

        # Add mesh to MeshSet
        faces = np.asarray(input_mesh.faces, dtype=np.int32)
        pml_mesh = pymeshlab.Mesh(
            vertex_matrix=vertices,
            face_matrix=faces
        )
        ms.add_mesh(pml_mesh)

        # Run Alpha Wrap
        print("[AlphaWrap] Running CGAL Alpha Wrap... (this may take a while)")
        ms.generate_alpha_wrap(
            alpha=pymeshlab.PureValue(alpha),
            offset=pymeshlab.PureValue(offset)
        )

        # Get result
        result_pml = ms.current_mesh()
        result_vertices = result_pml.vertex_matrix()
        result_faces = result_pml.face_matrix()

        # Create trimesh result
        result_mesh = trimesh.Trimesh(
            vertices=result_vertices,
            faces=result_faces,
            process=False
        )

        # Copy metadata if present
        if hasattr(input_mesh, 'metadata') and input_mesh.metadata:
            result_mesh.metadata = input_mesh.metadata.copy()
        else:
            result_mesh.metadata = {}

        # Add alpha wrap info to metadata
        result_mesh.metadata['alpha_wrap'] = {
            'alpha': alpha,
            'alpha_percent': alpha_percent,
            'offset': offset,
            'offset_percent': offset_percent,
            'bbox_diagonal': bbox_diagonal,
            'input_type': input_type
        }

        # Compute stats
        output_vertex_count = len(result_mesh.vertices)
        output_face_count = len(result_mesh.faces)
        is_watertight = result_mesh.is_watertight

        print(f"[AlphaWrap] Result: {output_vertex_count:,} vertices, {output_face_count:,} faces")
        print(f"[AlphaWrap] Watertight: {is_watertight}")

        # Build report
        report = f"""Alpha Wrap Report
{'='*40}

Input:
  Type: {input_type}
  Vertices: {input_vertex_count:,}
  Faces: {input_face_count:,}

Parameters:
  Alpha: {alpha:.6f} ({alpha_percent}% of bbox diagonal)
  Offset: {offset:.6f} ({offset_percent}% of bbox diagonal)
  BBox Diagonal: {bbox_diagonal:.4f}

Output:
  Vertices: {output_vertex_count:,}
  Faces: {output_face_count:,}
  Watertight: {'Yes' if is_watertight else 'No'}

Tips:
  - Decrease alpha_percent for tighter wrap / more detail
  - Decrease offset_percent to get closer to original surface
  - Lower values = slower computation
"""

        return (result_mesh, report)


NODE_CLASS_MAPPINGS = {
    "GeomPackAlphaWrap": AlphaWrapNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackAlphaWrap": "Alpha Wrap (Shrink Wrap)",
}
