"""
ComfyUI GeomPack - Geometry Processing Custom Nodes

This package provides mesh processing nodes for ComfyUI using trimesh, CGAL, and Blender.
Includes custom 3D preview widget powered by Three.js.
"""

import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

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

    # Setup custom server routes for save functionality
    try:
        from aiohttp import web
        from server import PromptServer
        import folder_paths

        routes = PromptServer.instance.routes

        @routes.post("/geometrypack/save_preview")
        async def save_preview_mesh(request):
            """
            Save a preview mesh file with a timestamped filename.

            Request JSON:
                {
                    "temp_filename": "preview_vtk_fields_abc123.vtp"
                }

            Response JSON:
                {
                    "success": true,
                    "saved_filename": "mesh_20250112_143022.vtp",
                    "message": "Mesh saved successfully"
                }
            """
            try:
                json_data = await request.json()
                temp_filename = json_data.get("temp_filename")

                if not temp_filename:
                    return web.json_response({
                        "success": False,
                        "error": "No temp_filename provided"
                    }, status=400)

                # Get the output directory
                output_dir = folder_paths.get_output_directory()
                temp_filepath = os.path.join(output_dir, temp_filename)

                # Check if temp file exists
                if not os.path.exists(temp_filepath):
                    return web.json_response({
                        "success": False,
                        "error": f"Temporary file not found: {temp_filename}"
                    }, status=404)

                # Generate timestamped filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_ext = os.path.splitext(temp_filename)[1]  # Preserve original extension
                saved_filename = f"mesh_{timestamp}{file_ext}"
                saved_filepath = os.path.join(output_dir, saved_filename)

                # Copy the file (keep temporary file)
                shutil.copy2(temp_filepath, saved_filepath)

                print(f"[GeometryPack] Saved preview mesh: {saved_filename}")

                return web.json_response({
                    "success": True,
                    "saved_filename": saved_filename,
                    "message": f"Mesh saved successfully as {saved_filename}"
                })

            except Exception as e:
                print(f"[GeometryPack] Error saving preview mesh: {str(e)}")
                return web.json_response({
                    "success": False,
                    "error": str(e)
                }, status=500)

        print("[GeomPack] Custom server routes registered")

    except Exception as e:
        print(f"[GeomPack] WARNING: Failed to register server routes: {str(e)}")
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
