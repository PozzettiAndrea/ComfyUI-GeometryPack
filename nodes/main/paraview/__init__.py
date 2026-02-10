# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
ParaView/VTK filter nodes using PyVista.
"""

from .pv_filter import ParaViewFilterNode, NODE_CLASS_MAPPINGS as PV_FILTER_MAPS, NODE_DISPLAY_NAME_MAPPINGS as PV_FILTER_DISP

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(PV_FILTER_MAPS)
NODE_DISPLAY_NAME_MAPPINGS.update(PV_FILTER_DISP)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
