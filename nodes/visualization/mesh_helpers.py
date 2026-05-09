import trimesh


def is_point_cloud(mesh) -> bool:
    """Check if a trimesh object is a point cloud (has no faces)."""
    return not (hasattr(mesh, 'faces') and mesh.faces is not None and len(mesh.faces) > 0)


def get_face_count(mesh) -> int:
    """Safely get the number of faces from a mesh object."""
    return len(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None else 0


def get_geometry_type(mesh) -> str:
    """Get a human-readable string describing the geometry type."""
    return "Point Cloud" if is_point_cloud(mesh) else "Mesh"
