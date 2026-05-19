# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
All geometry processing nodes — Blender, GPU, and main (CGAL/libigl/etc.).
"""

import importlib
import logging

log = logging.getLogger("geometrypack")
log.setLevel(logging.INFO)


def _safe_import(name):
    """Import a sibling submodule; log and return None if it fails."""
    try:
        return importlib.import_module("." + name, __name__)
    except Exception as e:
        log.warning("ComfyUI-GeometryPack: skipping %s (%s)", name, e)
        return None


# --- Blender nodes (import defensively: bpy may be broken on some installs) ---
blender_io = _safe_import("blender_io")
blender_boolean = _safe_import("blender_boolean")
blender_remeshing = _safe_import("blender_remeshing")
blender_texture_remeshing = _safe_import("blender_texture_remeshing")
blender_uv = _safe_import("blender_uv")

# --- GPU nodes ---
from . import remeshing_gpu
from . import fill_holes_gpu
from . import uv_gpu

# --- Main nodes ---
from . import io
from . import primitives
from . import analysis
from . import distance
from . import conversion
from . import transforms
from . import visualization
from . import skeleton
from . import combine
from . import uv
from . import repair
from . import remeshing
from . import reconstruction
from . import texture_remeshing
from . import smoothing
from . import decimation
from . import paraview
from . import reconstruction_cgal
from . import boolean
from . import remeshing_cgal
from . import repair_cgal
from . import decimation_cgal

# Collect all node class mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _merge(mod):
    if mod is None:
        return
    NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))


# Blender (may be None if the bpy-dependent package failed to import)
_merge(blender_io)
_merge(blender_boolean)
_merge(blender_remeshing)
_merge(blender_texture_remeshing)
_merge(blender_uv)

# GPU
NODE_CLASS_MAPPINGS.update(remeshing_gpu.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(fill_holes_gpu.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(uv_gpu.NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS.update(remeshing_gpu.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(fill_holes_gpu.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(uv_gpu.NODE_DISPLAY_NAME_MAPPINGS)

# Main
NODE_CLASS_MAPPINGS.update(io.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(primitives.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(analysis.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(distance.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(conversion.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(transforms.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(visualization.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(skeleton.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(combine.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(uv.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(repair.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(remeshing.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(reconstruction.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(texture_remeshing.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(paraview.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(reconstruction_cgal.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(boolean.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(remeshing_cgal.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(repair_cgal.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(smoothing.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(decimation.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(decimation_cgal.NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS.update(io.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(primitives.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(analysis.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(distance.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(conversion.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(transforms.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(visualization.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(skeleton.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(combine.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(uv.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(repair.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(remeshing.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(reconstruction.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(texture_remeshing.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(paraview.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(reconstruction_cgal.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(boolean.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(remeshing_cgal.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(repair_cgal.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(smoothing.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(decimation.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(decimation_cgal.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
