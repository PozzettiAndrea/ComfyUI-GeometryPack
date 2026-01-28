# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Main nodes - Basic dependencies (trimesh, numpy, pymeshlab, etc.)
These nodes will always load regardless of CGAL or Blender availability.
"""

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

# Collect all node class mappings
NODE_CLASS_MAPPINGS = {}
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

# Collect all display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}
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

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
