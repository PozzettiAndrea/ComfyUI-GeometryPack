"""
Analysis Nodes - Mesh information and topology analysis
"""

from . import mesh_utils


class MeshInfoNode:
    """
    Display detailed mesh information and statistics.
    Now using trimesh for enhanced mesh analysis.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_mesh_info"
    CATEGORY = "geompack/analysis"

    def get_mesh_info(self, trimesh):
        """
        Get information about the trimesh.

        Args:
            trimesh: trimesh.Trimesh object

        Returns:
            tuple: (info_string,)
        """
        info = mesh_utils.compute_mesh_info(trimesh)
        return (info,)


class MarkBoundaryEdgesNode:
    """
    Detect and mark boundary edges on trimesh.

    Analyzes the mesh topology to find boundary edges (edges belonging to
    only one face). Creates a vertex scalar field 'boundary_vertex' where:
    - 1.0 = vertex is on a boundary edge
    - 0.0 = vertex is interior

    This field can be visualized in VTK viewer to highlight mesh boundaries.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh_with_field", "info")
    FUNCTION = "mark_boundary"
    CATEGORY = "geompack/analysis"

    def mark_boundary(self, trimesh):
        """
        Detect boundary edges and add boundary vertex field to trimesh.

        Args:
            trimesh: Input trimesh.Trimesh object

        Returns:
            tuple: (mesh_with_boundary_field, info_string)
        """
        result_trimesh, error = mesh_utils.mark_boundary_vertices(trimesh)

        if error:
            raise RuntimeError(f"Failed to detect boundary edges: {error}")

        # Get statistics from metadata
        num_boundary_verts = result_trimesh.metadata.get('boundary_vertices_count', 0)
        num_boundary_edges = result_trimesh.metadata.get('boundary_edges_count', 0)

        # Determine if mesh is watertight (no boundary edges)
        is_watertight = num_boundary_edges == 0

        info = f"""Boundary Edge Analysis:

Boundary Edges: {num_boundary_edges:,}
Boundary Vertices: {num_boundary_verts:,} / {len(trimesh.vertices):,} ({100.0 * num_boundary_verts / len(trimesh.vertices):.1f}%)
Watertight: {'Yes' if is_watertight else 'No'}

Field Added: 'boundary_vertex'
  - Value 1.0 = on boundary
  - Value 0.0 = interior

Use VTK viewer to visualize the boundary field!
"""

        print(f"[MarkBoundaryEdges] Found {num_boundary_edges:,} boundary edges, {num_boundary_verts:,} boundary vertices")

        return (result_trimesh, info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackMeshInfo": MeshInfoNode,
    "GeomPackMarkBoundaryEdges": MarkBoundaryEdgesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackMeshInfo": "Mesh Info",
    "GeomPackMarkBoundaryEdges": "Mark Boundary Edges",
}
