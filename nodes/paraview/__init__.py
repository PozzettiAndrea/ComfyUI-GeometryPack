# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
ParaView/VTK filter nodes using PyVista.
"""

from .threshold_by_field import NODE_CLASS_MAPPINGS as THRESH_MAPS, NODE_DISPLAY_NAME_MAPPINGS as THRESH_DISP

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(THRESH_MAPS)
NODE_DISPLAY_NAME_MAPPINGS.update(THRESH_DISP)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
