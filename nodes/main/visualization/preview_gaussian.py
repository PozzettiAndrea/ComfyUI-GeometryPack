# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Preview Gaussian Splatting PLY files with gsplat.js viewer.

Displays 3D Gaussian Splats in an interactive WebGL viewer.
"""

import os

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_OUTPUT_FOLDER = None


class PreviewGaussianNode:
    """
    Preview Gaussian Splatting PLY files.

    Displays 3D Gaussian Splats in an interactive gsplat.js viewer
    with orbit controls and real-time rendering.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a Gaussian Splatting PLY file"
                }),
            },
            "optional": {
                "extrinsics": ("EXTRINSICS", {
                    "tooltip": "4x4 camera extrinsics matrix for initial view"
                }),
                "intrinsics": ("INTRINSICS", {
                    "tooltip": "3x3 camera intrinsics matrix for FOV"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_gaussian"
    CATEGORY = "geompack/visualization"

    def preview_gaussian(self, ply_path: str, extrinsics=None, intrinsics=None):
        """
        Prepare PLY file for gsplat.js preview.

        Args:
            ply_path: Path to the Gaussian Splatting PLY file
            extrinsics: Optional 4x4 camera extrinsics matrix
            intrinsics: Optional 3x3 camera intrinsics matrix

        Returns:
            dict: UI data for frontend widget
        """
        if not ply_path:
            print("[PreviewGaussian] No PLY path provided")
            return {"ui": {"error": ["No PLY path provided"]}}

        if not os.path.exists(ply_path):
            print(f"[PreviewGaussian] PLY file not found: {ply_path}")
            return {"ui": {"error": [f"File not found: {ply_path}"]}}

        # Get just the filename for the frontend
        filename = os.path.basename(ply_path)

        # Check if file is in ComfyUI output directory
        if COMFYUI_OUTPUT_FOLDER and ply_path.startswith(COMFYUI_OUTPUT_FOLDER):
            # File is already in output folder, just use the filename
            relative_path = os.path.relpath(ply_path, COMFYUI_OUTPUT_FOLDER)
        else:
            # File is elsewhere - for now just use basename
            # The viewer will construct the full URL
            relative_path = filename

        # Get file size
        file_size = os.path.getsize(ply_path)
        file_size_mb = file_size / (1024 * 1024)

        print(f"[PreviewGaussian] Loading PLY: {filename} ({file_size_mb:.2f} MB)")

        # Return metadata for frontend widget
        ui_data = {
            "ply_file": [relative_path],
            "filename": [filename],
            "file_size_mb": [round(file_size_mb, 2)],
        }

        # Add camera parameters if provided
        if extrinsics is not None:
            ui_data["extrinsics"] = [extrinsics]
        if intrinsics is not None:
            ui_data["intrinsics"] = [intrinsics]

        return {"ui": ui_data}


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewGaussian": PreviewGaussianNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewGaussian": "Preview Gaussian",
}
