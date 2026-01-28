# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Mesh Info Node - Display detailed mesh information
"""

from ..._utils import mesh_ops


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
    OUTPUT_NODE = True
    FUNCTION = "get_mesh_info"
    CATEGORY = "geompack/analysis"
    INPUT_IS_LIST = True

    def get_mesh_info(self, trimesh):
        """
        Get information about the trimesh(es).

        Args:
            trimesh: list of trimesh.Trimesh objects (when INPUT_IS_LIST=True)

        Returns:
            dict: {"result": (info_string,), "ui": {"text": [info]}}
        """
        # Handle batch processing - concatenate all info
        all_info = []
        for i, mesh in enumerate(trimesh):
            mesh_info = mesh_ops.compute_mesh_info(mesh)
            batch_header = f"{'='*60}\n=== Batch Item {i+1}/{len(trimesh)} ===\n{'='*60}\n\n"
            all_info.append(batch_header + mesh_info)

        # Join all info with separators
        combined_info = "\n\n".join(all_info)

        return {
            "result": (combined_info,),
            "ui": {"text": [combined_info]}
        }


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackMeshInfo": MeshInfoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackMeshInfo": "Mesh Info",
}
