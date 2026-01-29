# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Mesh from Skeleton Node - Convert skeleton to solid mesh
"""

import numpy as np
import trimesh


class SkeletonToMesh:
    """
    Convert skeleton to solid mesh with cylinders (bones) and spheres (joints).

    High-quality visualization with adjustable geometry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON",),
                "joint_radius": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 0.1, "step": 0.001}),
                "bone_radius": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.05, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "convert"
    CATEGORY = "geompack/skeleton"

    def convert(self, skeleton, joint_radius, bone_radius):
        """Convert skeleton to solid geometry."""
        vertices = skeleton["vertices"]
        edges = skeleton["edges"]

        # Read metadata (with fallback for old skeletons without metadata)
        scale = skeleton.get("scale", None)
        center = skeleton.get("center", None)
        is_normalized = skeleton.get("normalized", True)  # Assume old skeletons were normalized

        print(f"[SkeletonToMesh] Creating solid mesh: {len(vertices)} joints, {len(edges)} bones")
        if scale is not None:
            print(f"[SkeletonToMesh] Skeleton metadata: scale={scale:.3f}, normalized={is_normalized}")

        # Print input skeleton bounding box
        skel_min = vertices.min(axis=0)
        skel_max = vertices.max(axis=0)
        skel_size = skel_max - skel_min
        skel_center = (skel_min + skel_max) / 2
        print(f"[SkeletonToMesh] Input skeleton bounding box:")
        print(f"  Min: [{skel_min[0]:.3f}, {skel_min[1]:.3f}, {skel_min[2]:.3f}]")
        print(f"  Max: [{skel_max[0]:.3f}, {skel_max[1]:.3f}, {skel_max[2]:.3f}]")
        print(f"  Size: [{skel_size[0]:.3f}, {skel_size[1]:.3f}, {skel_size[2]:.3f}]")
        print(f"  Center: [{skel_center[0]:.3f}, {skel_center[1]:.3f}, {skel_center[2]:.3f}]")

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

        print(f"[SkeletonToMesh] Created mesh: {len(combined_mesh.vertices)} vertices, "
              f"{len(combined_mesh.faces)} faces")

        # Print output mesh bounding box
        mesh_min = combined_mesh.bounds[0]
        mesh_max = combined_mesh.bounds[1]
        mesh_size = mesh_max - mesh_min
        mesh_center = (mesh_min + mesh_max) / 2
        print(f"[SkeletonToMesh] Output mesh bounding box:")
        print(f"  Min: [{mesh_min[0]:.3f}, {mesh_min[1]:.3f}, {mesh_min[2]:.3f}]")
        print(f"  Max: [{mesh_max[0]:.3f}, {mesh_max[1]:.3f}, {mesh_max[2]:.3f}]")
        print(f"  Size: [{mesh_size[0]:.3f}, {mesh_size[1]:.3f}, {mesh_size[2]:.3f}]")
        print(f"  Center: [{mesh_center[0]:.3f}, {mesh_center[1]:.3f}, {mesh_center[2]:.3f}]")

        return (combined_mesh,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackMeshFromSkeleton": SkeletonToMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackMeshFromSkeleton": "Mesh from Skeleton",
}
