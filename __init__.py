# SPDX-License-Identifier: GPL-3.0-or-later
"""ComfyUI GeometryPack - Geometry Processing Custom Nodes."""

import os
import shutil
from pathlib import Path
from datetime import datetime

from comfy_env import wrap_nodes
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

wrap_nodes()


def _generate_widget_mappings():
    """Generate widget visibility mappings from node metadata."""
    try:
        from comfy_dynamic_widgets import scan_specific_nodes, generate_mappings
        import json

        configs = scan_specific_nodes(NODE_CLASS_MAPPINGS)
        if not configs:
            return

        mappings = generate_mappings(configs)
        output_path = os.path.join(os.path.dirname(__file__), "web", "js", "mappings.json")
        with open(output_path, "w") as f:
            json.dump(mappings, f, indent=2)
    except ImportError:
        pass
    except Exception:
        pass


_generate_widget_mappings()

# Server routes
try:
    from aiohttp import web
    from server import PromptServer
    import folder_paths

    routes = PromptServer.instance.routes

    @routes.post("/geometrypack/save_preview")
    async def save_preview_mesh(request):
        try:
            json_data = await request.json()
            temp_filename = json_data.get("temp_filename")
            if not temp_filename:
                return web.json_response({"success": False, "error": "No temp_filename"}, status=400)

            output_dir = folder_paths.get_output_directory()
            temp_filepath = os.path.join(output_dir, temp_filename)
            if not os.path.exists(temp_filepath):
                return web.json_response({"success": False, "error": "File not found"}, status=404)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_ext = os.path.splitext(temp_filename)[1]
            saved_filename = f"mesh_{timestamp}{file_ext}"
            shutil.copy2(temp_filepath, os.path.join(output_dir, saved_filename))

            return web.json_response({"success": True, "saved_filename": saved_filename})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @routes.post("/geompack/analyze")
    async def analyze_mesh(request):
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
                return web.json_response({"success": False, "error": "Missing params"}, status=400)

            cache_entry = get_cached_mesh(mesh_id)
            if not cache_entry:
                return web.json_response({"success": False, "error": "Mesh not found"}, status=404)

            mesh = cache_entry['mesh']
            filename = cache_entry['filename']

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
                return web.json_response({"success": False, "error": f"Unknown: {analysis_type}"}, status=400)

            add_field_to_cached_mesh(mesh_id, field_name)
            filepath = os.path.join(folder_paths.get_output_directory(), filename)
            export_mesh_with_scalars_vtp(mesh, filepath)

            return web.json_response({"success": True, "filename": filename, "field_name": field_name, "count": count})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    @routes.post("/geompack/find_location")
    async def find_location(request):
        try:
            from .nodes.visualization.preview_mesh_analysis import get_cached_mesh
            import re

            json_data = await request.json()
            mesh_id = json_data.get("mesh_id")
            query = json_data.get("query", "").strip()

            if not mesh_id or not query:
                return web.json_response({"success": False, "error": "Missing params"}, status=400)

            cache_entry = get_cached_mesh(mesh_id)
            if not cache_entry:
                return web.json_response({"success": False, "error": "Mesh not found"}, status=404)

            mesh = cache_entry['mesh']
            point, query_type, element_id = None, None, None

            # Face ID: f123, face123
            face_match = re.match(r'^[fF](?:ace)?\s*(\d+)$', query)
            if face_match:
                face_id = int(face_match.group(1))
                if 0 <= face_id < len(mesh.faces):
                    point = mesh.vertices[mesh.faces[face_id]].mean(axis=0).tolist()
                    query_type, element_id = "face", face_id

            # Vertex ID: v123, vertex123
            if point is None:
                vertex_match = re.match(r'^[vVpP](?:ertex|oint)?\s*(\d+)$', query)
                if vertex_match:
                    vertex_id = int(vertex_match.group(1))
                    if 0 <= vertex_id < len(mesh.vertices):
                        point = mesh.vertices[vertex_id].tolist()
                        query_type, element_id = "vertex", vertex_id

            # XYZ coordinates
            if point is None:
                coord_str = query.strip('()[]')
                parts = re.split(r'[,\s]+', coord_str)
                if len(parts) == 3:
                    try:
                        point = [float(parts[0]), float(parts[1]), float(parts[2])]
                        query_type = "coordinate"
                    except ValueError:
                        pass

            # Plain number = face ID
            if point is None:
                try:
                    face_id = int(query)
                    if 0 <= face_id < len(mesh.faces):
                        point = mesh.vertices[mesh.faces[face_id]].mean(axis=0).tolist()
                        query_type, element_id = "face", face_id
                except ValueError:
                    pass

            if point is None:
                return web.json_response({"success": False, "error": f"Could not parse: {query}"}, status=400)

            response = {"success": True, "point": point, "type": query_type}
            if element_id is not None:
                response["id"] = element_id
            return web.json_response(response)
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

except Exception:
    pass

WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
