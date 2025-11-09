"""
ComfyUI GeomPack - Geometry Processing Custom Nodes

This package provides mesh processing nodes for ComfyUI using trimesh, CGAL, and Blender.
Includes custom 3D preview widget powered by Three.js.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Set web directory for JavaScript extensions (3D mesh preview widget)
# This tells ComfyUI where to find our JavaScript files and HTML viewer
# Files will be served at /extensions/ComfyUI-GeomPack/*
WEB_DIRECTORY = "./web"

# Export the mappings so ComfyUI can discover the nodes
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
