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
                "mesh": ("MESH",),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("centered_mesh",)
    FUNCTION = "center_mesh"
    CATEGORY = "geompack/transforms"

    def center_mesh(self, mesh):
        """
        Center mesh at origin using bounding box center.

        Args:
            mesh: Input trimesh.Trimesh object

        Returns:
            tuple: (centered_trimesh.Trimesh,)
        """
        print(f"[CenterMesh] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Calculate bounding box center
        bounds_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0

        print(f"[CenterMesh] Original center: [{bounds_center[0]:.3f}, {bounds_center[1]:.3f}, {bounds_center[2]:.3f}]")

        # Apply translation to center at origin
        mesh_centered = mesh.copy()
        mesh_centered.apply_translation(-bounds_center)

        # Verify centering
        new_center = (mesh_centered.bounds[0] + mesh_centered.bounds[1]) / 2.0
        print(f"[CenterMesh] New center: [{new_center[0]:.3f}, {new_center[1]:.3f}, {new_center[2]:.3f}]")

        # Preserve metadata
        mesh_centered.metadata = mesh.metadata.copy()
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
