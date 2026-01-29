# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Split By Field Node - Split point cloud/mesh by discrete vertex attribute
"""

from typing import Tuple

import numpy as np
import trimesh


class SplitByFieldNode:
    """
    Split a point cloud or mesh by a discrete vertex attribute field.

    Useful for debugging segmentation results - extract each cluster separately.
    Works with any integer-valued vertex attribute (e.g., labels, primitive types).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "geometry": ("TRIMESH,POINT_CLOUD", {
                    "tooltip": "Input point cloud or mesh with vertex_attributes."
                }),
                "field_name": ("STRING", {
                    "default": "label",
                    "tooltip": "Name of the discrete field to split by (e.g., 'label', 'primitive_type', 'cluster')."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("geometries", "summary")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "split"
    CATEGORY = "geompack/combine"

    def split(self, geometry, field_name: str) -> Tuple:
        """Split geometry by a discrete field."""
        print(f"[SplitByField] Splitting by field: '{field_name}'")

        # Check field exists
        if not hasattr(geometry, 'vertex_attributes') or geometry.vertex_attributes is None:
            raise ValueError("Geometry has no vertex_attributes")

        if field_name not in geometry.vertex_attributes:
            available = list(geometry.vertex_attributes.keys())
            raise ValueError(f"Field '{field_name}' not found. Available: {available}")

        field = geometry.vertex_attributes[field_name]

        # Check discrete (integer)
        if not np.issubdtype(field.dtype, np.integer):
            raise ValueError(f"Field '{field_name}' is not discrete (dtype: {field.dtype}). Must be integer.")

        # Check < 100 unique values
        unique_values = np.unique(field)
        if len(unique_values) > 100:
            raise ValueError(f"Too many unique values ({len(unique_values)}). Maximum allowed: 100")

        print(f"   Found {len(unique_values)} unique values: {unique_values}")

        # Determine if input is a point cloud or mesh
        is_point_cloud = (
            isinstance(geometry, trimesh.PointCloud) or
            geometry.metadata.get('is_point_cloud', False) or
            not hasattr(geometry, 'faces') or
            len(geometry.faces) == 0
        )

        # Split into separate geometries
        result = []
        summary_lines = [f"Split by '{field_name}': {len(unique_values)} groups\n"]

        for val in unique_values:
            mask = field == val
            num_points = np.sum(mask)

            if is_point_cloud:
                # Create point cloud subset
                subset = trimesh.PointCloud(geometry.vertices[mask])
            else:
                # For meshes, extract submesh by vertex mask
                # This is more complex - need to handle faces
                vertex_indices = np.where(mask)[0]
                # Create index mapping
                index_map = {old: new for new, old in enumerate(vertex_indices)}
                # Find faces where all vertices are in the mask
                face_mask = np.all(np.isin(geometry.faces, vertex_indices), axis=1)
                if np.sum(face_mask) > 0:
                    new_faces = geometry.faces[face_mask]
                    # Remap face indices
                    new_faces = np.vectorize(index_map.get)(new_faces)
                    subset = trimesh.Trimesh(
                        vertices=geometry.vertices[mask],
                        faces=new_faces
                    )
                else:
                    # No valid faces, create point cloud
                    subset = trimesh.PointCloud(geometry.vertices[mask])

            # Copy vertex attributes
            if not hasattr(subset, 'vertex_attributes'):
                subset.vertex_attributes = {}
            for attr_name, attr_data in geometry.vertex_attributes.items():
                subset.vertex_attributes[attr_name] = attr_data[mask]

            # Copy normals if available
            if hasattr(geometry, 'vertex_normals') and geometry.vertex_normals is not None:
                if len(geometry.vertex_normals) == len(geometry.vertices):
                    # For point clouds, store in metadata since vertex_normals may not be settable
                    if is_point_cloud:
                        subset.metadata['vertex_normals'] = geometry.vertex_normals[mask]
                    else:
                        try:
                            subset.vertex_normals = geometry.vertex_normals[mask]
                        except Exception:
                            subset.metadata['vertex_normals'] = geometry.vertex_normals[mask]

            # Set metadata
            subset.metadata['split_field'] = field_name
            subset.metadata['split_value'] = int(val)
            subset.metadata['is_point_cloud'] = is_point_cloud

            result.append(subset)
            summary_lines.append(f"  {field_name}={val}: {num_points} points")
            print(f"   {field_name}={val}: {num_points} points")

        summary = "\n".join(summary_lines)
        return (result, summary)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackSplitByField": SplitByFieldNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackSplitByField": "Split By Field",
}
