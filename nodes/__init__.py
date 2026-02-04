# SPDX-License-Identifier: GPL-3.0-or-later
"""ComfyUI GeometryPack Nodes."""

from .main import NODE_CLASS_MAPPINGS as main_mappings
from .main import NODE_DISPLAY_NAME_MAPPINGS as main_display
from .cgal import NODE_CLASS_MAPPINGS as cgal_mappings
from .cgal import NODE_DISPLAY_NAME_MAPPINGS as cgal_display
from .blender import NODE_CLASS_MAPPINGS as blender_mappings
from .blender import NODE_DISPLAY_NAME_MAPPINGS as blender_display
from .gpu import NODE_CLASS_MAPPINGS as gpu_mappings
from .gpu import NODE_DISPLAY_NAME_MAPPINGS as gpu_display

NODE_CLASS_MAPPINGS = {
    **main_mappings,
    **cgal_mappings,
    **blender_mappings,
    **gpu_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **main_display,
    **cgal_display,
    **blender_display,
    **gpu_display,
}
