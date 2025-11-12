"""
Visualization Nodes - 3D mesh preview and visualization
"""

import trimesh as trimesh_module
import os
import tempfile
import uuid
import sys
from pathlib import Path

# Import create_bbox_visualization from Hunyuan3D-Part
# Add parent directory to path to allow cross-custom-node imports
custom_nodes_dir = Path(__file__).parent.parent.parent
hunyuan_utils_path = custom_nodes_dir / "ComfyUI-Hunyuan3D-Part" / "node_utils"
if str(hunyuan_utils_path) not in sys.path:
    sys.path.insert(0, str(hunyuan_utils_path))

try:
    from mesh_utils import create_bbox_visualization, export_scene_to_vtp
    BBOX_VIZ_AVAILABLE = True
except ImportError:
    print("[GeomPack] WARNING: Could not import bbox functions from Hunyuan3D-Part")
    print("[GeomPack] The Preview Bounding Boxes node will not be available")
    BBOX_VIZ_AVAILABLE = False
    export_scene_to_vtp = None

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except:
    COMFYUI_OUTPUT_FOLDER = None


def _export_mesh_with_scalars_vtp(trimesh: trimesh_module.Trimesh, filepath: str):
    """
    Export trimesh to VTK PolyData XML format (.vtp) with vertex scalar attributes.

    VTP format preserves vertex attributes (scalar fields) which can be visualized
    in VTK.js with color mapping.

    Args:
        trimesh: Trimesh object with optional vertex_attributes
        filepath: Output .vtp file path
    """
    import xml.etree.ElementTree as ET

    print(f"[_export_mesh_with_scalars_vtp] Exporting to VTP: {filepath}")

    # Create VTK PolyData XML structure
    vtk_file = ET.Element('VTKFile', type='PolyData', version='1.0', byte_order='LittleEndian')
    poly_data = ET.SubElement(vtk_file, 'PolyData')

    num_verts = len(trimesh.vertices)
    num_faces = len(trimesh.faces)

    piece = ET.SubElement(poly_data, 'Piece', NumberOfPoints=str(num_verts), NumberOfPolys=str(num_faces))

    # Points section
    points = ET.SubElement(piece, 'Points')
    points_data_array = ET.SubElement(points, 'DataArray',
                                       type='Float32',
                                       NumberOfComponents='3',
                                       format='ascii')
    # Flatten vertices to space-separated string
    verts_flat = trimesh.vertices.flatten()
    points_data_array.text = ' '.join(map(str, verts_flat))

    # PointData section (scalar fields)
    point_data = ET.SubElement(piece, 'PointData')

    # Add vertex attributes as scalar arrays
    if hasattr(trimesh, 'vertex_attributes') and trimesh.vertex_attributes:
        for attr_name, attr_values in trimesh.vertex_attributes.items():
            print(f"[_export_mesh_with_scalars_vtp]   Adding scalar field: {attr_name}")
            scalar_array = ET.SubElement(point_data, 'DataArray',
                                          type='Float32',
                                          Name=attr_name,
                                          format='ascii')
            scalar_array.text = ' '.join(map(str, attr_values.flatten()))

    # Polys section (faces/triangles)
    polys = ET.SubElement(piece, 'Polys')

    # Connectivity: vertex indices for each face
    connectivity = ET.SubElement(polys, 'DataArray',
                                   type='Int32',
                                   Name='connectivity',
                                   format='ascii')
    faces_flat = trimesh.faces.flatten()
    connectivity.text = ' '.join(map(str, faces_flat))

    # Offsets: cumulative count of indices (each triangle has 3 vertices)
    offsets = ET.SubElement(polys, 'DataArray',
                             type='Int32',
                             Name='offsets',
                             format='ascii')
    offset_values = [(i + 1) * 3 for i in range(num_faces)]
    offsets.text = ' '.join(map(str, offset_values))

    # Write to file with pretty formatting
    tree = ET.ElementTree(vtk_file)
    ET.indent(tree, space='  ')
    tree.write(filepath, encoding='utf-8', xml_declaration=True)

    print(f"[_export_mesh_with_scalars_vtp] Export complete: {num_verts} vertices, {num_faces} faces")


class PreviewMeshNode:
    """
    Preview mesh with interactive 3D viewer.

    Displays mesh in an interactive Three.js viewer with orbit controls.
    Allows rotating, panning, and zooming to inspect the mesh geometry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_mesh"
    CATEGORY = "geompack/visualization"

    def preview_mesh(self, trimesh):
        """
        Export mesh to GLB and prepare for 3D preview.

        Args:
            trimesh: Input trimesh_module.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMesh] Preparing preview: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Generate unique filename
        filename = f"preview_{uuid.uuid4().hex[:8]}.glb"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export to GLB (best format for Three.js)
        try:
            trimesh.export(filepath, file_type='glb')
            print(f"[PreviewMesh] Exported to: {filepath}")
        except Exception as e:
            print(f"[PreviewMesh] Export failed: {e}")
            # Fallback to OBJ
            filename = filename.replace('.glb', '.obj')
            filepath = filepath.replace('.glb', '.obj')
            trimesh.export(filepath, file_type='obj')
            print(f"[PreviewMesh] Exported to OBJ: {filepath}")

        # Calculate bounding box info for camera setup
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Return metadata for frontend widget
        return {
            "ui": {
                "mesh_file": [filename],
                "vertex_count": [len(trimesh.vertices)],
                "face_count": [len(trimesh.faces)],
                "bounds_min": [bounds[0].tolist()],
                "bounds_max": [bounds[1].tolist()],
                "extents": [extents.tolist()],
                "max_extent": [float(max_extent)],
            }
        }


class PreviewMeshVTKNode:
    """
    Preview mesh with VTK.js scientific visualization viewer.

    Displays mesh in an interactive VTK.js viewer with trackball controls.
    Better for scientific visualization, mesh analysis, and large datasets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_mesh_vtk"
    CATEGORY = "geompack/visualization"

    def preview_mesh_vtk(self, trimesh):
        """
        Export mesh to STL and prepare for VTK.js preview.

        Args:
            trimesh: Input trimesh_module.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshVTK] Preparing preview: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Generate unique filename
        filename = f"preview_vtk_{uuid.uuid4().hex[:8]}.stl"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export to STL (native format for VTK.js)
        try:
            trimesh.export(filepath, file_type='stl')
            print(f"[PreviewMeshVTK] Exported to: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshVTK] Export failed: {e}")
            # Fallback to OBJ
            filename = filename.replace('.stl', '.obj')
            filepath = filepath.replace('.stl', '.obj')
            trimesh.export(filepath, file_type='obj')
            print(f"[PreviewMeshVTK] Exported to OBJ: {filepath}")

        # Calculate bounding box info for camera setup
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Check if mesh is watertight
        is_watertight = trimesh.is_watertight

        # Calculate volume and area (only if watertight)
        volume = None
        area = None
        try:
            if is_watertight:
                volume = float(trimesh.volume)
            area = float(trimesh.area)
        except Exception as e:
            print(f"[PreviewMeshVTK] Could not calculate volume/area: {e}")

        # Get field names (vertex/face data arrays)
        field_names = []
        if hasattr(trimesh, 'vertex_attributes') and trimesh.vertex_attributes:
            field_names.extend([f"vertex.{k}" for k in trimesh.vertex_attributes.keys()])
        if hasattr(trimesh, 'face_attributes') and trimesh.face_attributes:
            field_names.extend([f"face.{k}" for k in trimesh.face_attributes.keys()])

        # Return metadata for frontend widget
        ui_data = {
            "mesh_file": [filename],
            "vertex_count": [len(trimesh.vertices)],
            "face_count": [len(trimesh.faces)],
            "bounds_min": [bounds[0].tolist()],
            "bounds_max": [bounds[1].tolist()],
            "extents": [extents.tolist()],
            "max_extent": [float(max_extent)],
            "is_watertight": [bool(is_watertight)],
        }

        # Add optional fields if available
        if volume is not None:
            ui_data["volume"] = [volume]
        if area is not None:
            ui_data["area"] = [area]
        if field_names:
            ui_data["field_names"] = [field_names]

        print(f"[PreviewMeshVTK] Mesh info: watertight={is_watertight}, volume={volume}, area={area}, fields={len(field_names)}")

        return {"ui": ui_data}


class PreviewMeshVTKFiltersNode:
    """
    Preview mesh with VTK.js scientific visualization viewer including filters.

    Displays mesh in an interactive VTK.js viewer with trackball controls
    and filter controls (smoothing, outline, cutting plane).
    Better for scientific visualization, mesh analysis, and large datasets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_mesh_vtk_filters"
    CATEGORY = "geompack/visualization"

    def preview_mesh_vtk_filters(self, trimesh):
        """
        Export mesh to STL and prepare for VTK.js preview with filters.

        Args:
            trimesh: Input trimesh_module.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshVTKFilters] Preparing preview: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Check if mesh has vertex attributes (scalar fields)
        has_vertex_attrs = hasattr(trimesh, 'vertex_attributes') and len(trimesh.vertex_attributes) > 0

        print(f"[PreviewMeshVTKFilters] hasattr(trimesh, 'vertex_attributes'): {hasattr(trimesh, 'vertex_attributes')}")
        if hasattr(trimesh, 'vertex_attributes'):
            print(f"[PreviewMeshVTKFilters] trimesh.vertex_attributes: {trimesh.vertex_attributes.keys()}")
            print(f"[PreviewMeshVTKFilters] Number of vertex attributes: {len(trimesh.vertex_attributes)}")

        # Generate unique filename
        # Use VTP format if we have vertex attributes, otherwise STL
        if has_vertex_attrs:
            filename = f"preview_vtk_filters_{uuid.uuid4().hex[:8]}.vtp"
            file_format = 'vtp'
            print(f"[PreviewMeshVTKFilters] Mesh has vertex attributes, using VTP format")
        else:
            filename = f"preview_vtk_filters_{uuid.uuid4().hex[:8]}.stl"
            file_format = 'stl'
            print(f"[PreviewMeshVTKFilters] No vertex attributes found, using STL format")

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export mesh with vertex attributes
        try:
            if file_format == 'vtp':
                # Export to VTK PolyData format (XML) which preserves vertex attributes
                _export_mesh_with_scalars_vtp(trimesh, filepath)
                print(f"[PreviewMeshVTKFilters] Exported VTP with {len(trimesh.vertex_attributes)} scalar fields to: {filepath}")
            else:
                # Export to STL (no scalar data)
                trimesh.export(filepath, file_type='stl')
                print(f"[PreviewMeshVTKFilters] Exported to STL: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshVTKFilters] Export failed: {e}")
            # Fallback to OBJ
            filename = filename.replace('.vtp', '.obj').replace('.stl', '.obj')
            filepath = filepath.replace('.vtp', '.obj').replace('.stl', '.obj')
            trimesh.export(filepath, file_type='obj')
            print(f"[PreviewMeshVTKFilters] Exported to OBJ: {filepath}")

        # Calculate bounding box info for camera setup
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Check if mesh is watertight
        is_watertight = trimesh.is_watertight

        # Calculate volume and area (only if watertight)
        volume = None
        area = None
        try:
            if is_watertight:
                volume = float(trimesh.volume)
            area = float(trimesh.area)
        except Exception as e:
            print(f"[PreviewMeshVTKFilters] Could not calculate volume/area: {e}")

        # Get field names (vertex/face data arrays)
        field_names = []
        if hasattr(trimesh, 'vertex_attributes') and trimesh.vertex_attributes:
            field_names.extend([f"vertex.{k}" for k in trimesh.vertex_attributes.keys()])
        if hasattr(trimesh, 'face_attributes') and trimesh.face_attributes:
            field_names.extend([f"face.{k}" for k in trimesh.face_attributes.keys()])

        # Return metadata for frontend widget
        ui_data = {
            "mesh_file": [filename],
            "vertex_count": [len(trimesh.vertices)],
            "face_count": [len(trimesh.faces)],
            "bounds_min": [bounds[0].tolist()],
            "bounds_max": [bounds[1].tolist()],
            "extents": [extents.tolist()],
            "max_extent": [float(max_extent)],
            "is_watertight": [bool(is_watertight)],
        }

        # Add optional fields if available
        if volume is not None:
            ui_data["volume"] = [volume]
        if area is not None:
            ui_data["area"] = [area]
        if field_names:
            ui_data["field_names"] = [field_names]

        print(f"[PreviewMeshVTKFilters] Mesh info: watertight={is_watertight}, volume={volume}, area={area}, fields={len(field_names)}")

        return {"ui": ui_data}


class PreviewMeshVTKFieldsNode:
    """
    Preview mesh with VTK.js viewer optimized for scalar field visualization.

    Specifically designed for visualizing vertex scalar fields (like boundary edges,
    curvature, distance fields, etc.) with color mapping. Always exports as VTP
    format to preserve scalar data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_mesh_vtk_fields"
    CATEGORY = "geompack/visualization"

    def preview_mesh_vtk_fields(self, trimesh):
        """
        Export mesh with scalar fields to VTP and prepare for VTK.js preview.

        Args:
            trimesh: Input trimesh_module.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshVTKFields] Preparing preview: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Check if mesh has vertex attributes (scalar fields)
        has_vertex_attrs = hasattr(trimesh, 'vertex_attributes') and len(trimesh.vertex_attributes) > 0

        if hasattr(trimesh, 'vertex_attributes'):
            print(f"[PreviewMeshVTKFields] trimesh.vertex_attributes: {trimesh.vertex_attributes.keys()}")
            print(f"[PreviewMeshVTKFields] Number of vertex attributes: {len(trimesh.vertex_attributes)}")

        if not has_vertex_attrs:
            print(f"[PreviewMeshVTKFields] WARNING: No vertex attributes found. This node is for scalar field visualization.")

        # Always use VTP format for this node (designed for scalar fields)
        filename = f"preview_vtk_fields_{uuid.uuid4().hex[:8]}.vtp"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export mesh with vertex attributes as VTP
        try:
            _export_mesh_with_scalars_vtp(trimesh, filepath)
            num_fields = len(trimesh.vertex_attributes) if has_vertex_attrs else 0
            print(f"[PreviewMeshVTKFields] Exported VTP with {num_fields} scalar fields to: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshVTKFields] Export failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to export mesh with scalar fields: {e}")

        # Calculate bounding box info for camera setup
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Check if mesh is watertight
        is_watertight = trimesh.is_watertight

        # Get field names (vertex/face data arrays)
        field_names = []
        if hasattr(trimesh, 'vertex_attributes') and trimesh.vertex_attributes:
            field_names.extend(list(trimesh.vertex_attributes.keys()))
        if hasattr(trimesh, 'face_attributes') and trimesh.face_attributes:
            field_names.extend([f"face.{k}" for k in trimesh.face_attributes.keys()])

        # Return metadata for frontend widget
        ui_data = {
            "mesh_file": [filename],
            "vertex_count": [len(trimesh.vertices)],
            "face_count": [len(trimesh.faces)],
            "bounds_min": [bounds[0].tolist()],
            "bounds_max": [bounds[1].tolist()],
            "extents": [extents.tolist()],
            "max_extent": [float(max_extent)],
            "is_watertight": [bool(is_watertight)],
            "field_names": [field_names],
        }

        print(f"[PreviewMeshVTKFields] Scalar fields: {field_names}")

        return {"ui": ui_data}


class PreviewBoundingBoxesVTKNode:
    """
    Preview mesh with bounding boxes using VTK.js viewer.

    Displays the original mesh with wireframe bounding boxes overlaid,
    showing the segmentation results from P3-SAM or other bbox-generating nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "bboxes": ("BBOXES_3D",),
                "line_width": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                    "display": "slider",
                    "tooltip": "Thickness of bounding box wireframe lines in pixels. Rendered by VTK.js."
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_bboxes_vtk"
    CATEGORY = "geompack/visualization"

    def preview_bboxes_vtk(self, trimesh, bboxes, line_width):
        """
        Export mesh with bounding boxes to VTP and prepare for VTK.js preview.

        Args:
            trimesh: Input trimesh_module.Trimesh object
            bboxes: BBOXES_3D dict with 'bboxes' (array [N, 2, 3]) and 'num_parts'
            line_width: Thickness of wireframe lines in pixels (for VTK.js rendering)

        Returns:
            dict: UI data for frontend widget
        """
        if not BBOX_VIZ_AVAILABLE:
            raise RuntimeError(
                "create_bbox_visualization not available. "
                "Make sure ComfyUI-Hunyuan3D-Part is installed."
            )

        # Extract bboxes array from BBOXES_3D dictionary
        if isinstance(bboxes, dict):
            bboxes_array = bboxes['bboxes']
            num_parts = bboxes.get('num_parts', len(bboxes_array))
        else:
            # Fallback for raw array
            bboxes_array = bboxes
            num_parts = len(bboxes_array)

        print(f"[PreviewBoundingBoxesVTK] Preparing preview with {num_parts} bounding boxes")
        print(f"[PreviewBoundingBoxesVTK] Mesh: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[PreviewBoundingBoxesVTK] Line width: {line_width}px")

        # Create scene with mesh and bounding boxes
        # Use Path3D wireframes (rendered as lines in VTK.js)
        scene = create_bbox_visualization(trimesh, bboxes_array, use_tubes=False)

        # Generate unique filename - use VTP which supports wireframes
        filename = f"preview_bboxes_vtk_{uuid.uuid4().hex[:8]}.vtp"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export scene to VTP (preserves wireframes)
        try:
            # Use custom VTP exporter that preserves both mesh and wireframe lines
            export_scene_to_vtp(scene, filepath)
            print(f"[PreviewBoundingBoxesVTK] Exported scene with {num_parts} bounding boxes to: {filepath}")
        except Exception as e:
            print(f"[PreviewBoundingBoxesVTK] VTP export failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to STL with tubes
            print(f"[PreviewBoundingBoxesVTK] Falling back to STL with tube meshes")
            scene = create_bbox_visualization(trimesh, bboxes_array, use_tubes=True)
            filename = filename.replace('.vtp', '.stl')
            filepath = filepath.replace('.vtp', '.stl')
            scene.export(filepath, file_type='stl')
            print(f"[PreviewBoundingBoxesVTK] Exported to: {filepath}")

        # Calculate bounding box info for camera setup
        # Use the original mesh bounds for camera positioning
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Return metadata for frontend widget
        ui_data = {
            "mesh_file": [filename],
            "vertex_count": [len(trimesh.vertices)],
            "face_count": [len(trimesh.faces)],
            "bounds_min": [bounds[0].tolist()],
            "bounds_max": [bounds[1].tolist()],
            "extents": [extents.tolist()],
            "max_extent": [float(max_extent)],
            "num_bboxes": [num_parts],
            "line_width": [float(line_width)],
        }

        print(f"[PreviewBoundingBoxesVTK] Created visualization with {num_parts} bounding boxes")

        return {"ui": ui_data}


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMesh": PreviewMeshNode,
    "GeomPackPreviewMeshVTK": PreviewMeshVTKNode,
    "GeomPackPreviewMeshVTKFilters": PreviewMeshVTKFiltersNode,
    "GeomPackPreviewMeshVTKFields": PreviewMeshVTKFieldsNode,
    "GeomPackPreviewBoundingBoxesVTK": PreviewBoundingBoxesVTKNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMesh": "Preview Mesh (3D)",
    "GeomPackPreviewMeshVTK": "Preview Mesh (VTK)",
    "GeomPackPreviewMeshVTKFilters": "Preview Mesh (VTK with Filters)",
    "GeomPackPreviewMeshVTKFields": "Preview Mesh (VTK with Fields)",
    "GeomPackPreviewBoundingBoxesVTK": "Preview Bounding Boxes (VTK)",
}
