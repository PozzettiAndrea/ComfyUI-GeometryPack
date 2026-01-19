# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Helper function for exporting mesh to VTK PolyData XML format (.vtp) with scalar attributes.

VTP format preserves vertex attributes (PointData) and face attributes (CellData)
which can be visualized in VTK.js with color mapping.

Supports both meshes (with faces) and point clouds (without faces).
"""

import numpy as np
import trimesh as trimesh_module
import xml.etree.ElementTree as ET

from .._utils.mesh_ops import is_point_cloud, get_face_count


def export_mesh_with_scalars_vtp(trimesh: trimesh_module.Trimesh, filepath: str):
    """
    Export trimesh to VTK PolyData XML format (.vtp) with scalar attributes.

    VTP format preserves vertex attributes (PointData) and face attributes (CellData)
    which can be visualized in VTK.js with color mapping.

    Supports both meshes (with faces) and point clouds (without faces).
    For point clouds, uses Verts section instead of Polys section.

    Args:
        trimesh: Trimesh or PointCloud object with optional vertex_attributes and face_attributes
        filepath: Output .vtp file path
    """
    is_pc = is_point_cloud(trimesh)
    geometry_type = "point cloud" if is_pc else "mesh"

    print(f"[_export_mesh_with_scalars_vtp] Exporting {geometry_type} to VTP: {filepath}")

    # Create VTK PolyData XML structure
    vtk_file = ET.Element('VTKFile', type='PolyData', version='1.0', byte_order='LittleEndian')
    poly_data = ET.SubElement(vtk_file, 'PolyData')

    num_verts = len(trimesh.vertices)
    num_faces = get_face_count(trimesh)

    # For point clouds, set NumberOfVerts instead of NumberOfPolys
    if is_pc:
        piece = ET.SubElement(poly_data, 'Piece',
                             NumberOfPoints=str(num_verts),
                             NumberOfVerts=str(num_verts),
                             NumberOfPolys='0')
    else:
        piece = ET.SubElement(poly_data, 'Piece',
                             NumberOfPoints=str(num_verts),
                             NumberOfPolys=str(num_faces))

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

    # CellData section (face attributes) - only for meshes with faces
    if not is_pc:
        cell_data = ET.SubElement(piece, 'CellData')

        # Add face attributes as scalar arrays
        if hasattr(trimesh, 'face_attributes') and trimesh.face_attributes:
            for attr_name, attr_values in trimesh.face_attributes.items():
                attr_arr = np.asarray(attr_values)
                # Skip high-dimensional arrays (e.g., 448-dim feature vectors)
                if attr_arr.ndim > 1 and attr_arr.shape[1] > 4:
                    print(f"[_export_mesh_with_scalars_vtp]   Skipping high-dim field: {attr_name} (shape {attr_arr.shape})")
                    continue
                print(f"[_export_mesh_with_scalars_vtp]   Adding face field: {attr_name}")
                num_components = attr_arr.shape[1] if attr_arr.ndim > 1 else 1
                scalar_array = ET.SubElement(cell_data, 'DataArray',
                                              type='Float32',
                                              Name=attr_name,
                                              NumberOfComponents=str(num_components),
                                              format='ascii')
                scalar_array.text = ' '.join(map(str, attr_arr.flatten()))

    # Geometry section: Verts for point clouds, Polys for meshes
    if is_pc:
        # For point clouds, create individual vertex cells
        verts = ET.SubElement(piece, 'Verts')

        # Connectivity: one index per point (0, 1, 2, 3, ...)
        connectivity = ET.SubElement(verts, 'DataArray',
                                       type='Int32',
                                       Name='connectivity',
                                       format='ascii')
        connectivity.text = ' '.join(map(str, range(num_verts)))

        # Offsets: cumulative count (1, 2, 3, 4, ...)
        offsets = ET.SubElement(verts, 'DataArray',
                                 type='Int32',
                                 Name='offsets',
                                 format='ascii')
        offsets.text = ' '.join(map(str, range(1, num_verts + 1)))
    else:
        # For meshes, create polygon cells (faces/triangles)
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

    if is_pc:
        print(f"[_export_mesh_with_scalars_vtp] Export complete: {num_verts} points")
    else:
        print(f"[_export_mesh_with_scalars_vtp] Export complete: {num_verts} vertices, {num_faces} faces")
