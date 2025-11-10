"""
ComfyUI GeomPack - Geometry Processing Custom Nodes

This package provides mesh processing nodes for ComfyUI using trimesh, CGAL, and Blender.
Includes custom 3D preview widget powered by Three.js.
"""

import sys
import os
from pathlib import Path

# Only run initialization when loaded by ComfyUI, not during pytest
# This prevents import errors when pytest collects test modules
if 'pytest' not in sys.modules:
    # Check if CGAL is available
    try:
        from CGAL import CGAL_Polygon_mesh_processing
        print("[GeomPack] CGAL Python package found - CGAL Isotropic Remesh node available")
    except ImportError:
        print("[GeomPack] WARNING: CGAL Python package not found")
        print("[GeomPack] The CGAL Isotropic Remesh node will not be available")
        print("[GeomPack] Install with: pip install cgal")
        print("[GeomPack] You can use PyMeshLab Remesh as an alternative")

    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    # During testing, don't import nodes
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Set web directory for JavaScript extensions (3D mesh preview widget)
# This tells ComfyUI where to find our JavaScript files and HTML viewer
# Files will be served at /extensions/ComfyUI-GeomPack/*
WEB_DIRECTORY = "./web"

# Export the mappings so ComfyUI can discover the nodes
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
