# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Combine/split operations module."""

from .combine_meshes import NODE_CLASS_MAPPINGS as COMBINE_MAPPINGS
from .combine_meshes import NODE_DISPLAY_NAME_MAPPINGS as COMBINE_DISPLAY
from .combine_meshes_batch import NODE_CLASS_MAPPINGS as COMBINE_BATCH_MAPPINGS
from .combine_meshes_batch import NODE_DISPLAY_NAME_MAPPINGS as COMBINE_BATCH_DISPLAY
from .split_by_field import NODE_CLASS_MAPPINGS as SPLIT_FIELD_MAPPINGS
from .split_by_field import NODE_DISPLAY_NAME_MAPPINGS as SPLIT_FIELD_DISPLAY

# Combine all mappings
NODE_CLASS_MAPPINGS = {
    **COMBINE_MAPPINGS,
    **COMBINE_BATCH_MAPPINGS,
    **SPLIT_FIELD_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **COMBINE_DISPLAY,
    **COMBINE_BATCH_DISPLAY,
    **SPLIT_FIELD_DISPLAY,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
