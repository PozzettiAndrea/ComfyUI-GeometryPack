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
            outputs=[
                io.Custom("EXTRINSICS").Output(display_name="extrinsics"),
                io.Custom("INTRINSICS").Output(display_name="intrinsics"),
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
        from ..io.path_utils import resolve_read_path, searched_locations

        if not ply_path:
            log.info("No PLY path provided")
            return io.NodeOutput(extrinsics, intrinsics, ui={"error": ["No PLY path provided"]})

        resolved = resolve_read_path(ply_path)
        if resolved is None:
            searched = "\n  - ".join(searched_locations(ply_path))
            log.info("PLY file not found: %s (searched:\n  - %s)", ply_path, searched)
            return io.NodeOutput(extrinsics, intrinsics,
                                 ui={"error": [f"File not found: {ply_path}\nSearched in:\n  - {searched}"]})
        ply_path = resolved

        # Get just the filename for the frontend
        filename = os.path.basename(ply_path)

        # The viewer JS routes "(output|input|temp)/<subpath>" to the matching
        # /view?type=...&subfolder=... URL, so hand it the path relative to the
        # ComfyUI base whenever the file lives under it. Bare basenames are
        # mis-routed to type=output -- last-ditch fallback only.
        try:
            import folder_paths
            base = getattr(folder_paths, "base_path", None)
        except (ImportError, AttributeError):
            base = None
        try:
            under_base = base and os.path.commonpath(
                [os.path.abspath(ply_path), os.path.abspath(base)]) == os.path.abspath(base)
        except ValueError:  # different drives on Windows
            under_base = False
        if under_base:
            relative_path = os.path.relpath(ply_path, base)
        elif COMFYUI_OUTPUT_FOLDER and ply_path.startswith(COMFYUI_OUTPUT_FOLDER):
            relative_path = os.path.relpath(ply_path, COMFYUI_OUTPUT_FOLDER)
        else:
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

        # Add camera parameters if provided. The `ui` dict is JSON-
        # serialized for the WS broadcast (server.py publish_loop), so
        # Tensors must be converted to plain nested lists or the
        # publish loop crashes with `Object of type Tensor is not JSON
        # serializable` and tears down the whole asyncio main loop.
        def _to_jsonable(x):
            if x is None:
                return None
            # Strip leading singleton batch dims (e.g. [1, 4, 4] -> [4, 4]).
            try:
                import torch as _torch
                if isinstance(x, _torch.Tensor):
                    t = x.detach().float().cpu()
                    while t.dim() > 2 and t.shape[0] == 1:
                        t = t[0]
                    return t.tolist()
            except Exception:
                pass
            if hasattr(x, "tolist"):
                try:
                    return x.tolist()
                except Exception:
                    pass
            return x

        if extrinsics is not None:
            ui_data["extrinsics"] = [_to_jsonable(extrinsics)]
        if intrinsics is not None:
            ui_data["intrinsics"] = [_to_jsonable(intrinsics)]

        return io.NodeOutput(extrinsics, intrinsics, ui=ui_data)


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewGaussian": PreviewGaussianNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewGaussian": "Preview Gaussian",
}
