# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Mesh from Skeleton Node - Convert skeleton to solid mesh
"""

import logging

import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class SkeletonToMesh(io.ComfyNode):
    """
    Convert skeleton to solid mesh with cylinders (bones) and spheres (joints).

    High-quality visualization with adjustable geometry.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackMeshFromSkeleton",
            display_name="Mesh from Skeleton",
            category="geompack/skeleton",
            inputs=[
                io.Custom("SKELETON").Input("skeleton"),
                io.Float.Input("joint_radius", default=0.02, min=0.001, max=0.1, step=0.001),
                io.Float.Input("bone_radius", default=0.01, min=0.001, max=0.05, step=0.001),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh"),
            ],
        )

    @classmethod
    def execute(cls, skeleton, joint_radius, bone_radius):
        """Convert skeleton to solid geometry."""
        vertices = skeleton["vertices"]
        edges = skeleton["edges"]

        # Read metadata (with fallback for old skeletons without metadata)
        scale = skeleton.get("scale", None)
        center = skeleton.get("center", None)
        is_normalized = skeleton.get("normalized", True)  # Assume old skeletons were normalized

        log.info("Creating solid mesh: %d joints, %d bones", len(vertices), len(edges))
        if scale is not None:
            log.info("Skeleton metadata: scale=%.3f, normalized=%s", scale, is_normalized)

        # Print input skeleton bounding box
        skel_min = vertices.min(axis=0)
        skel_max = vertices.max(axis=0)
        skel_size = skel_max - skel_min
        skel_center = (skel_min + skel_max) / 2
        log.info("Input skeleton bounding box:")
        log.info("  Min: [%.3f, %.3f, %.3f]", skel_min[0], skel_min[1], skel_min[2])
        log.info("  Max: [%.3f, %.3f, %.3f]", skel_max[0], skel_max[1], skel_max[2])
        log.info("  Size: [%.3f, %.3f, %.3f]", skel_size[0], skel_size[1], skel_size[2])
        log.info("  Center: [%.3f, %.3f, %.3f]", skel_center[0], skel_center[1], skel_center[2])

        meshes = []

        # Create joint spheres
        for vertex in vertices:
            sphere = trimesh.creation.uv_sphere(radius=joint_radius, count=[8, 8])
            sphere.apply_translation(vertex)
            meshes.append(sphere)

        # Create bone cylinders
        for edge in edges:
            start = vertices[edge[0]]
            end = vertices[edge[1]]

            # Calculate cylinder parameters
            direction = end - start
            length = np.linalg.norm(direction)

            if length < 1e-6:
                continue  # Skip degenerate bones

            # Create cylinder along Z-axis
            cylinder = trimesh.creation.cylinder(
                radius=bone_radius,
                height=length,
                sections=8
            )

            # Calculate rotation to align with bone direction
            z_axis = np.array([0, 0, 1])
            bone_direction = direction / length

            # Rotation axis and angle
            rotation_axis = np.cross(z_axis, bone_direction)
            rotation_axis_norm = np.linalg.norm(rotation_axis)

            if rotation_axis_norm > 1e-6:
                rotation_axis = rotation_axis / rotation_axis_norm
                rotation_angle = np.arccos(np.clip(np.dot(z_axis, bone_direction), -1.0, 1.0))

                # Create rotation matrix
                from trimesh.transformations import rotation_matrix
                rotation = rotation_matrix(rotation_angle, rotation_axis)
                cylinder.apply_transform(rotation)

            # Translate to midpoint
            midpoint = (start + end) / 2
            cylinder.apply_translation(midpoint)

            meshes.append(cylinder)

        # Combine all meshes
        if not meshes:
            raise ValueError("No geometry created from skeleton")

        combined_mesh = trimesh.util.concatenate(meshes)

        log.info("Created mesh: %d vertices, %d faces", len(combined_mesh.vertices), len(combined_mesh.faces))

        # Print output mesh bounding box
        mesh_min = combined_mesh.bounds[0]
        mesh_max = combined_mesh.bounds[1]
        mesh_size = mesh_max - mesh_min
        mesh_center = (mesh_min + mesh_max) / 2
        log.info("Output mesh bounding box:")
        log.info("  Min: [%.3f, %.3f, %.3f]", mesh_min[0], mesh_min[1], mesh_min[2])
        log.info("  Max: [%.3f, %.3f, %.3f]", mesh_max[0], mesh_max[1], mesh_max[2])
        log.info("  Size: [%.3f, %.3f, %.3f]", mesh_size[0], mesh_size[1], mesh_size[2])
        log.info("  Center: [%.3f, %.3f, %.3f]", mesh_center[0], mesh_center[1], mesh_center[2])

        return io.NodeOutput(combined_mesh)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackMeshFromSkeleton": SkeletonToMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackMeshFromSkeleton": "Mesh from Skeleton",
}
