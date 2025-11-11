"""
Transform Nodes - Mesh transformations and positioning
"""


class CenterMeshNode:
    """
    Center mesh at origin (0, 0, 0).

    Uses bounding box center to translate the mesh so its center
    is at the world origin. Useful for preparing meshes for export
    or ensuring consistent positioning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("centered_mesh",)
    FUNCTION = "center_mesh"
    CATEGORY = "geompack/transforms"

    def center_mesh(self, trimesh):
        """
        Center mesh at origin using bounding box center.

        Args:
            trimesh: Input trimesh.Trimesh object

        Returns:
            tuple: (centered_trimesh.Trimesh,)
        """
        print(f"[CenterMesh] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Calculate bounding box center
        bounds_center = (trimesh.bounds[0] + trimesh.bounds[1]) / 2.0

        print(f"[CenterMesh] Original center: [{bounds_center[0]:.3f}, {bounds_center[1]:.3f}, {bounds_center[2]:.3f}]")

        # Apply translation to center at origin
        mesh_centered = trimesh.copy()
        mesh_centered.apply_translation(-bounds_center)

        # Verify centering
        new_center = (mesh_centered.bounds[0] + mesh_centered.bounds[1]) / 2.0
        print(f"[CenterMesh] New center: [{new_center[0]:.3f}, {new_center[1]:.3f}, {new_center[2]:.3f}]")

        # Preserve metadata
        mesh_centered.metadata = trimesh.metadata.copy()
        mesh_centered.metadata['centered'] = True
        mesh_centered.metadata['original_center'] = bounds_center.tolist()

        return (mesh_centered,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackCenterMesh": CenterMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackCenterMesh": "Center Mesh",
}
