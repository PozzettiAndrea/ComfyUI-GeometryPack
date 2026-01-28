# Import submodules
from . import blender_io
from . import boolean
from . import remeshing
from . import texture_remeshing

# Collect all node class mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(blender_io.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(boolean.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(remeshing.NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(texture_remeshing.NODE_CLASS_MAPPINGS)

# Collect all display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(blender_io.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(boolean.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(remeshing.NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(texture_remeshing.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
