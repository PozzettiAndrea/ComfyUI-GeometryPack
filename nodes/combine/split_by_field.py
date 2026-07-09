# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Split By Field Node - Split point cloud/mesh by discrete vertex attribute
"""

import logging
from typing import Tuple

import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class SplitByFieldNode(io.ComfyNode):
    """
    Split a point cloud or mesh by a discrete vertex attribute field.

    Useful for debugging segmentation results - extract each cluster separately.
    Works with any integer-valued vertex attribute (e.g., labels, primitive types).
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSplitByField",
            display_name="Split By Field",
            category="geompack/combine",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH,POINT_CLOUD").Input("geometry", tooltip="Input point cloud or mesh with vertex_attributes (point data) or face_attributes (cell data)."),
                io.String.Input("field_name", default="label", tooltip="Name of the discrete field to split by (e.g., 'label', 'part_id', 'cluster'). For a face field labeled 'face.xxx' in viewers, use just 'xxx' here."),
                io.Int.Input("max_geometries", default=-1, min=-1, max=10000, tooltip="Cap the number of split-out geometries returned (-1 = unlimited, 0-10000 = explicit cap). An explicit cap bypasses the too-many-unique-values safety check below."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="geometries", is_output_list=True),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, geometry, field_name: str, max_geometries: int = -1) -> Tuple:
        """Split geometry by a discrete field (vertex/point or face/cell attribute)."""
        log.info("Splitting by field: '%s'", field_name)

        field_name = (field_name or "").strip()
        # allow 'face.xxx' field names from the viewers
        if field_name.startswith("face."):
            field_name = field_name[len("face."):]

        vertex_attrs = getattr(geometry, 'vertex_attributes', None) or {}
        face_attrs = getattr(geometry, 'face_attributes', None) or {}

        in_face = field_name in face_attrs
        in_vertex = field_name in vertex_attrs
        if not in_face and not in_vertex:
            raise ValueError(
                f"Field '{field_name}' not found. "
                f"vertex fields={list(vertex_attrs.keys())}, face fields={list(face_attrs.keys())}"
            )

        # Prefer face/cell data when the field lives on both, matching
        # ThresholdMeshByField's 'auto' behavior.
        by_face = in_face
        if by_face and (not hasattr(geometry, 'faces') or len(geometry.faces) == 0):
            raise ValueError(f"Field '{field_name}' is a face attribute but geometry has no faces.")

        field = np.asarray(face_attrs[field_name] if by_face else vertex_attrs[field_name])

        # Check discrete (integer)
        if not np.issubdtype(field.dtype, np.integer):
            raise ValueError(f"Field '{field_name}' is not discrete (dtype: {field.dtype}). Must be integer.")

        unique_values = np.unique(field)
        log.info("Found %d unique values (%s data): %s", len(unique_values), "face" if by_face else "vertex", unique_values)

        if max_geometries is not None and max_geometries >= 0:
            # An explicit cap means the user has already bounded the output
            # themselves -- skip the automatic safety cap below.
            if len(unique_values) > max_geometries:
                log.info("Capping to first %d of %d unique values (max_geometries)", max_geometries, len(unique_values))
            unique_values = unique_values[:max_geometries]
        elif len(unique_values) > 100:
            # No explicit cap: guard against accidentally creating hundreds of
            # tiny mesh fragments.
            raise ValueError(f"Too many unique values ({len(unique_values)}). Maximum allowed: 100")

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
            if by_face:
                # Group by face attribute: extract the faces with this value
                # plus the vertices they reference.
                face_mask = field == val
                face_indices = np.where(face_mask)[0]
                num_elements = len(face_indices)

                selected_faces = geometry.faces[face_indices]
                vertex_indices = np.unique(selected_faces)
                index_map = {old: new for new, old in enumerate(vertex_indices)}
                new_faces = np.vectorize(index_map.get)(selected_faces)
                subset = trimesh.Trimesh(
                    vertices=geometry.vertices[vertex_indices],
                    faces=new_faces
                )

                if not hasattr(subset, 'vertex_attributes'):
                    subset.vertex_attributes = {}
                for attr_name, attr_data in vertex_attrs.items():
                    subset.vertex_attributes[attr_name] = np.asarray(attr_data)[vertex_indices]

                if not hasattr(subset, 'face_attributes'):
                    subset.face_attributes = {}
                for attr_name, attr_data in face_attrs.items():
                    subset.face_attributes[attr_name] = np.asarray(attr_data)[face_indices]

                if hasattr(geometry, 'vertex_normals') and geometry.vertex_normals is not None:
                    if len(geometry.vertex_normals) == len(geometry.vertices):
                        try:
                            subset.vertex_normals = geometry.vertex_normals[vertex_indices]
                        except Exception:
                            subset.metadata['vertex_normals'] = geometry.vertex_normals[vertex_indices]
            else:
                mask = field == val
                num_elements = np.sum(mask)

                if is_point_cloud:
                    # Create point cloud subset
                    subset = trimesh.Trimesh(vertices=geometry.vertices[mask])
                else:
                    # For meshes, extract submesh by vertex mask
                    vertex_indices = np.where(mask)[0]
                    index_map = {old: new for new, old in enumerate(vertex_indices)}
                    # Find faces where all vertices are in the mask
                    face_mask = np.all(np.isin(geometry.faces, vertex_indices), axis=1)
                    if np.sum(face_mask) > 0:
                        new_faces = geometry.faces[face_mask]
                        new_faces = np.vectorize(index_map.get)(new_faces)
                        subset = trimesh.Trimesh(
                            vertices=geometry.vertices[mask],
                            faces=new_faces
                        )
                        if not hasattr(subset, 'face_attributes'):
                            subset.face_attributes = {}
                        face_indices = np.where(face_mask)[0]
                        for attr_name, attr_data in face_attrs.items():
                            subset.face_attributes[attr_name] = np.asarray(attr_data)[face_indices]
                    else:
                        # No valid faces, create point cloud
                        subset = trimesh.Trimesh(vertices=geometry.vertices[mask])

                # Copy vertex attributes
                if not hasattr(subset, 'vertex_attributes'):
                    subset.vertex_attributes = {}
                for attr_name, attr_data in vertex_attrs.items():
                    subset.vertex_attributes[attr_name] = np.asarray(attr_data)[mask]

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
            subset.metadata['is_point_cloud'] = is_point_cloud and not by_face

            result.append(subset)
            unit = "faces" if by_face else "points"
            summary_lines.append(f"  {field_name}={val}: {num_elements} {unit}")
            log.info("%s=%s: %d %s", field_name, val, num_elements, unit)

        summary = "\n".join(summary_lines)
        return io.NodeOutput(result, summary, ui={"text": [summary]})


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackSplitByField": SplitByFieldNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackSplitByField": "Split By Field",
}
