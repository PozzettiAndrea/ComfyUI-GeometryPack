# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

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
# Use PYTEST_CURRENT_TEST env var which is only set when pytest is actually running tests
if 'PYTEST_CURRENT_TEST' not in os.environ:
    # Setup import stubs BEFORE importing nodes
    # This allows imports like 'import trimesh' to succeed even though
    # trimesh is only installed in the isolated pixi environment
    try:
        from comfy_env import setup_isolated_imports
        setup_isolated_imports(__file__)
    except ImportError:
        print("[GeomPack] comfy-env not installed, import stubbing disabled")
    except Exception as e:
        print(f"[GeomPack] Failed to setup import stubs: {e}")

    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    # Enable process isolation - all nodes run in pixi env (Python 3.11)
    try:
        from comfy_env import enable_isolation
        enable_isolation(NODE_CLASS_MAPPINGS)
    except ImportError:
        pass  # Already warned above
    except Exception as e:
        print(f"[GeomPack] Failed to enable isolation: {e}")

    # Generate widget visibility mappings now that nodes are loaded
    def _generate_widget_mappings():
        try:
            from comfy_dynamic_widgets import scan_specific_nodes, generate_mappings
            import json

            configs = scan_specific_nodes(NODE_CLASS_MAPPINGS)

            if not configs:
                print("[GeometryPack] No visible_when metadata found in any nodes")
                return

            mappings = generate_mappings(configs)

            custom_node_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(custom_node_dir, "web", "js", "mappings.json")

            with open(output_path, "w") as f:
                json.dump(mappings, f, indent=2)

            node_count = len(configs)
            selector_count = sum(len(c.get("selectors", {})) for c in configs.values())
            print(f"[GeometryPack] Generated widget mappings for {node_count} nodes ({selector_count} selectors)")

        except ImportError:
            print("[GeometryPack] Warning: comfy-dynamic-widgets not installed, skipping widget mappings")
        except Exception as e:
            print(f"[GeometryPack] Error generating widget mappings: {e}")

    _generate_widget_mappings()

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

        @routes.post("/geompack/analyze")
        async def analyze_mesh(request):
            """
            Compute analysis field on a cached mesh and re-export.

            Request JSON:
                {
                    "mesh_id": "abc123def456",
                    "analysis_type": "open_edges" | "components" | "self_intersect"
                }

            Response JSON:
                {
                    "success": true,
                    "filename": "analysis_abc123def456.vtp",
                    "field_name": "boundary_vertex",
                    "count": 42
                }
            """
            try:
                from .nodes.visualization.preview_mesh_analysis import (
                    get_cached_mesh, add_field_to_cached_mesh,
                    compute_boundary_vertices, compute_connected_components,
                    compute_self_intersections
                )
                from .nodes.visualization._vtp_export import export_mesh_with_scalars_vtp

                json_data = await request.json()
                mesh_id = json_data.get("mesh_id")
                analysis_type = json_data.get("analysis_type")

                if not mesh_id or not analysis_type:
                    return web.json_response({
                        "success": False,
                        "error": "Missing mesh_id or analysis_type"
                    }, status=400)

                # Get cached mesh
                cache_entry = get_cached_mesh(mesh_id)
                if not cache_entry:
                    return web.json_response({
                        "success": False,
                        "error": f"Mesh not found in cache: {mesh_id}"
                    }, status=404)

                mesh = cache_entry['mesh']
                filename = cache_entry['filename']

                # Compute the requested analysis
                field_name = None
                count = 0

                if analysis_type == "open_edges":
                    mesh, count = compute_boundary_vertices(mesh)
                    field_name = "boundary_vertex"
                elif analysis_type == "components":
                    mesh, count = compute_connected_components(mesh)
                    field_name = "face.part_id"
                elif analysis_type == "self_intersect":
                    mesh, count = compute_self_intersections(mesh)
                    field_name = "face.self_intersect"
                else:
                    return web.json_response({
                        "success": False,
                        "error": f"Unknown analysis type: {analysis_type}"
                    }, status=400)

                # Track the field
                add_field_to_cached_mesh(mesh_id, field_name)

                # Re-export the mesh with the new field
                output_dir = folder_paths.get_output_directory()
                filepath = os.path.join(output_dir, filename)

                export_mesh_with_scalars_vtp(mesh, filepath)

                print(f"[GeomPack] Analysis complete: {analysis_type} -> {field_name} ({count})")

                return web.json_response({
                    "success": True,
                    "filename": filename,
                    "field_name": field_name,
                    "count": count
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[GeomPack] Error in analyze_mesh: {str(e)}")
                return web.json_response({
                    "success": False,
                    "error": str(e)
                }, status=500)

        @routes.post("/geompack/find_location")
        async def find_location(request):
            """
            Look up 3D coordinates for a face ID, vertex ID, or parse XYZ coordinates.

            Request JSON:
                {
                    "mesh_id": "abc123def456",
                    "query": "f123" | "v456" | "(1.0, 2.0, 3.0)" | "1.0 2.0 3.0"
                }

            Response JSON:
                {
                    "success": true,
                    "point": [x, y, z],
                    "type": "face" | "vertex" | "coordinate",
                    "id": 123  // for face/vertex
                }
            """
            try:
                from .nodes.visualization.preview_mesh_analysis import get_cached_mesh
                import numpy as np
                import re

                json_data = await request.json()
                mesh_id = json_data.get("mesh_id")
                query = json_data.get("query", "").strip()

                if not mesh_id or not query:
                    return web.json_response({
                        "success": False,
                        "error": "Missing mesh_id or query"
                    }, status=400)

                # Get cached mesh
                cache_entry = get_cached_mesh(mesh_id)
                if not cache_entry:
                    return web.json_response({
                        "success": False,
                        "error": f"Mesh not found in cache: {mesh_id}"
                    }, status=404)

                mesh = cache_entry['mesh']

                # Try to parse the query
                point = None
                query_type = None
                element_id = None

                # Pattern 1: Face ID - "f123", "face123", "face 123", "F123"
                face_match = re.match(r'^[fF](?:ace)?\s*(\d+)$', query)
                if face_match:
                    face_id = int(face_match.group(1))
                    if 0 <= face_id < len(mesh.faces):
                        # Get face center (centroid of the 3 vertices)
                        face = mesh.faces[face_id]
                        vertices = mesh.vertices[face]
                        point = vertices.mean(axis=0).tolist()
                        query_type = "face"
                        element_id = face_id
                    else:
                        return web.json_response({
                            "success": False,
                            "error": f"Face ID {face_id} out of range (0-{len(mesh.faces)-1})"
                        }, status=400)

                # Pattern 2: Vertex ID - "v123", "vertex123", "vertex 123", "p123", "point123"
                if point is None:
                    vertex_match = re.match(r'^[vVpP](?:ertex|oint)?\s*(\d+)$', query)
                    if vertex_match:
                        vertex_id = int(vertex_match.group(1))
                        if 0 <= vertex_id < len(mesh.vertices):
                            point = mesh.vertices[vertex_id].tolist()
                            query_type = "vertex"
                            element_id = vertex_id
                        else:
                            return web.json_response({
                                "success": False,
                                "error": f"Vertex ID {vertex_id} out of range (0-{len(mesh.vertices)-1})"
                            }, status=400)

                # Pattern 3: XYZ coordinates - "(1.0, 2.0, 3.0)" or "1.0 2.0 3.0" or "1.0, 2.0, 3.0"
                if point is None:
                    # Remove parentheses if present
                    coord_str = query.strip('()[]')
                    # Split by comma or whitespace
                    parts = re.split(r'[,\s]+', coord_str)
                    if len(parts) == 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            point = [x, y, z]
                            query_type = "coordinate"
                        except ValueError:
                            pass

                # Pattern 4: Just a number - assume face ID
                if point is None:
                    try:
                        face_id = int(query)
                        if 0 <= face_id < len(mesh.faces):
                            face = mesh.faces[face_id]
                            vertices = mesh.vertices[face]
                            point = vertices.mean(axis=0).tolist()
                            query_type = "face"
                            element_id = face_id
                        else:
                            return web.json_response({
                                "success": False,
                                "error": f"Face ID {face_id} out of range (0-{len(mesh.faces)-1})"
                            }, status=400)
                    except ValueError:
                        pass

                if point is None:
                    return web.json_response({
                        "success": False,
                        "error": f"Could not parse query: '{query}'. Use: f123, v456, or (x, y, z)"
                    }, status=400)

                response = {
                    "success": True,
                    "point": point,
                    "type": query_type
                }
                if element_id is not None:
                    response["id"] = element_id

                print(f"[GeomPack] Find location: {query} -> {query_type} at {point}")

                return web.json_response(response)

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[GeomPack] Error in find_location: {str(e)}")
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
