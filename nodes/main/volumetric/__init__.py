# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Volumetric nodes - tetrahedralization for FEM / simulation workflows.
"""

from .tetrahedralize import (
    NODE_CLASS_MAPPINGS as TET_MAPS,
    NODE_DISPLAY_NAME_MAPPINGS as TET_DISP,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(TET_MAPS)
NODE_DISPLAY_NAME_MAPPINGS.update(TET_DISP)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
