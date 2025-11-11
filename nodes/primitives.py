"""
Primitive Nodes - Create basic geometric shapes
"""

from . import mesh_utils


class CreatePrimitive:
    """
    Create primitive geometry (cube, sphere, plane)
    Uses trimesh creation functions for high-quality primitives.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape": (["cube", "sphere", "plane"], {
                    "default": "cube"
                }),
                "size": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "subdivisions": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "create_primitive"
    CATEGORY = "geompack/primitives"

    def create_primitive(self, shape, size, subdivisions=2):
        """
        Create a primitive mesh.

        Args:
            shape: Type of primitive (cube, sphere, plane)
            size: Size of the primitive
            subdivisions: Number of subdivisions (for sphere and plane)

        Returns:
            tuple: (trimesh.Trimesh,)
        """
        if shape == "cube":
            mesh = mesh_utils.create_cube(size)
        elif shape == "sphere":
            mesh = mesh_utils.create_sphere(radius=size/2.0, subdivisions=subdivisions)
        elif shape == "plane":
            mesh = mesh_utils.create_plane(size=size, subdivisions=subdivisions)
        else:
            raise ValueError(f"Unknown shape: {shape}")

        print(f"[CreatePrimitive] Created {shape}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        return (mesh,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackCreatePrimitive": CreatePrimitive,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackCreatePrimitive": "Create Primitive",
}
