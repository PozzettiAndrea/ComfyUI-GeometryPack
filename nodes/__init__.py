"""
ComfyUI GeometryPack Nodes
Organized by functional category
"""

# Import all node modules
from . import io
from . import primitives
from . import analysis
from . import distance
from . import conversion
from . import remeshing
from . import repair
from . import uv
from . import transforms
from . import visualization
from . import examples

# Collect all node class mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(io.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(primitives.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(analysis.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(distance.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(conversion.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(remeshing.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(repair.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(uv.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(transforms.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(visualization.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(examples.NODE_CLASS_MAPPINGS)

# Collect all display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(io.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(primitives.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(analysis.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(distance.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(conversion.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(remeshing.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(repair.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(uv.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(transforms.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(visualization.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(examples.NODE_DISPLAY_NAME_MAPPINGS)

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
