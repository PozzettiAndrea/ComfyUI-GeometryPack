# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Blender remeshing nodes
"""

# The legacy self-contained dispatcher (GeomPackRemeshBlender, remesh.py) was
# retired 2026-08-06: it predated the unified GeomPackRemesh dispatcher and
# carried a THIRD private copy of the bpy voxel logic, which had already
# drifted from the backends once (the voxel_size 1.0 cap). Note its
# "blender_smooth" mode had no standalone backend and is gone with it; add a
# backends/smooth.py if it is ever wanted again.
from .backends import NODE_CLASS_MAPPINGS as BACKENDS_MAPS, NODE_DISPLAY_NAME_MAPPINGS as BACKENDS_DISP

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(BACKENDS_MAPS)
NODE_DISPLAY_NAME_MAPPINGS.update(BACKENDS_DISP)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
