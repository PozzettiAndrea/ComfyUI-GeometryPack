# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Preview Gaussian Splatting PLY files with gsplat.js viewer.

Displays 3D Gaussian Splats in an interactive WebGL viewer.
"""

import logging

import os

log = logging.getLogger("geometrypack")

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_OUTPUT_FOLDER = None
from comfy_api.latest import io


class PreviewGaussianNode(io.ComfyNode):
    """
    Preview Gaussian Splatting PLY files.

    Displays 3D Gaussian Splats in an interactive gsplat.js viewer
    with orbit controls and real-time rendering.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackPreviewGaussian",
            display_name="Preview Gaussian",
            category="geompack/visualization",
            is_output_node=True,
            inputs=[
                io.String.Input("ply_path", tooltip="Path to a Gaussian Splatting PLY file", force_input=True),
                io.Custom("EXTRINSICS").Input("extrinsics", tooltip="4x4 camera extrinsics matrix for initial view", optional=True),
                io.Custom("INTRINSICS").Input("intrinsics", tooltip="3x3 camera intrinsics matrix for FOV", optional=True),
            ],
        )

    @classmethod
    def execute(cls, ply_path: str, extrinsics=None, intrinsics=None):
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
            log.info("No PLY path provided")
            return io.NodeOutput(ui={"error": ["No PLY path provided"]})

        if not os.path.exists(ply_path):
            log.info("PLY file not found: %s", ply_path)
            return io.NodeOutput(ui={"error": [f"File not found: {ply_path}"]})

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

        log.info("Loading PLY: %s (%.2f MB)", filename, file_size_mb)

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

        return io.NodeOutput(ui=ui_data)


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewGaussian": PreviewGaussianNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewGaussian": "Preview Gaussian",
}
