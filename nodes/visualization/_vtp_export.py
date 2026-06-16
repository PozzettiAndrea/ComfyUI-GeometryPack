# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Helper function for exporting mesh to VTK PolyData XML format (.vtp) with scalar attributes.

VTP format preserves vertex attributes (PointData) and face attributes (CellData)
which can be visualized in VTK.js with color mapping.

Supports both meshes (with faces) and point clouds (without faces).
"""

import logging

import numpy as np
import trimesh as trimesh_module
import xml.etree.ElementTree as ET

from .mesh_helpers import is_point_cloud, get_face_count

log = logging.getLogger("geometrypack")


def export_mesh_with_scalars_vtp(trimesh: trimesh_module.Trimesh, filepath: str):
    """Export trimesh to a .vtp with scalar attributes, BINARY + uncompressed.

    Binary (base64) is ~3-4x smaller than ASCII and is parsed by vtk.js as a typed
    array instead of parseFloat-ing millions of text numbers on the main thread --
    that text parse is what froze the browser on large meshes. It is LOSSLESS (full
    float32 / int connectivity). MUST be uncompressed: the bundled vtk.js has no
    zlib, so it can read ascii/binary VTP but not zlib-compressed VTP.

    Falls back to the ASCII writer if the `vtk` module is unavailable or anything
    goes wrong, so exports never hard-fail.
    """
    try:
        _export_binary_vtp(trimesh, filepath)
    except Exception as e:
        log.warning("Binary VTP export failed (%s); falling back to ASCII writer", e)
        _export_ascii_vtp(trimesh, filepath)


def _vtk_ready(name, arr):
    """Contiguous, dtype-trimmed copy for VTK: 0/1 masks -> uint8, else float32."""
    is_mask = (name.startswith("is_") or name.startswith("touches_")
               or name in ("boundary_vertex", "nonmanifold_vertex", "winding_inconsistent"))
    dt = np.uint8 if is_mask else np.float32
    return np.ascontiguousarray(np.asarray(arr).astype(dt))


def _export_binary_vtp(trimesh: trimesh_module.Trimesh, filepath: str):
    """Write a binary, uncompressed .vtp via the vtk XML writer (lossless)."""
    import vtk
    from vtk.util import numpy_support as nps

    is_pc = is_point_cloud(trimesh)
    verts = np.ascontiguousarray(np.asarray(trimesh.vertices, dtype=np.float32))
    n_verts = len(verts)

    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(nps.numpy_to_vtk(verts, deep=1))
    poly.SetPoints(pts)

    if is_pc:
        offsets = np.arange(0, n_verts + 1, dtype=np.int64)
        conn = np.arange(n_verts, dtype=np.int64)
        ca = vtk.vtkCellArray()
        ca.SetData(nps.numpy_to_vtkIdTypeArray(offsets, deep=1),
                   nps.numpy_to_vtkIdTypeArray(conn, deep=1))
        if n_verts < 2_000_000_000 and hasattr(ca, "ConvertTo32BitStorage"):
            ca.ConvertTo32BitStorage()  # int32 connectivity ~halves the cell bytes
        poly.SetVerts(ca)
    else:
        faces = np.asarray(trimesh.faces)
        n_faces = len(faces)
        conn = np.ascontiguousarray(faces.reshape(-1).astype(np.int64))
        offsets = np.arange(0, 3 * (n_faces + 1), 3, dtype=np.int64)
        ca = vtk.vtkCellArray()
        ca.SetData(nps.numpy_to_vtkIdTypeArray(offsets, deep=1),
                   nps.numpy_to_vtkIdTypeArray(conn, deep=1))
        if n_verts < 2_000_000_000 and hasattr(ca, "ConvertTo32BitStorage"):
            ca.ConvertTo32BitStorage()  # int32 connectivity ~halves the cell bytes
        poly.SetPolys(ca)

    # PointData: vertex normals (built-in) + vertex_attributes
    point_data = poly.GetPointData()
    if not is_pc and hasattr(trimesh, "vertex_normals"):
        try:
            nrm = np.asarray(trimesh.vertex_normals, dtype=np.float32)
            if nrm.shape == (n_verts, 3):
                a = nps.numpy_to_vtk(np.ascontiguousarray(nrm), deep=1)
                a.SetName("normals")
                point_data.AddArray(a)
        except Exception as e:
            log.warning("Could not export normals: %s", e)
    if getattr(trimesh, "vertex_attributes", None):
        for name, vals in trimesh.vertex_attributes.items():
            v = np.asarray(vals)
            if v.ndim > 1 and v.shape[1] > 4:
                continue
            a = nps.numpy_to_vtk(_vtk_ready(name, v), deep=1)
            a.SetName(name)
            point_data.AddArray(a)

    # CellData: face_attributes (meshes only)
    if not is_pc and getattr(trimesh, "face_attributes", None):
        cell_data = poly.GetCellData()
        for name, vals in trimesh.face_attributes.items():
            v = np.asarray(vals)
            if v.ndim > 1 and v.shape[1] > 4:
                continue
            a = nps.numpy_to_vtk(_vtk_ready(name, v), deep=1)
            a.SetName(name)
            cell_data.AddArray(a)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filepath)
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()      # base64 binary DataArrays (vtk.js reads these)
    writer.SetCompressorTypeToNone()  # NO zlib -- vtk.js cannot inflate
    writer.Write()
    log.info("Exported binary VTP: %s (%d verts, %d %s)", filepath, n_verts,
             n_verts if is_pc else len(trimesh.faces), "points" if is_pc else "faces")


def _export_ascii_vtp(trimesh: trimesh_module.Trimesh, filepath: str):
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

    log.info("Exporting %s to VTP: %s", geometry_type, filepath)

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

    # Add vertex normals if available (built-in trimesh property, not in vertex_attributes)
    if not is_pc and hasattr(trimesh, 'vertex_normals') and len(trimesh.vertex_normals) > 0:
        try:
            normals = np.asarray(trimesh.vertex_normals, dtype=np.float32)
            if normals.shape == (num_verts, 3):
                log.info("Adding normals field (%d vertices)", num_verts)
                normals_array = ET.SubElement(point_data, 'DataArray',
                                              type='Float32',
                                              Name='normals',
                                              NumberOfComponents='3',
                                              format='ascii')
                normals_array.text = ' '.join(map(str, normals.flatten()))
        except Exception as e:
            log.warning("Could not export normals: %s", e)

    # Add vertex attributes as scalar arrays
    if hasattr(trimesh, 'vertex_attributes') and trimesh.vertex_attributes:
        for attr_name, attr_values in trimesh.vertex_attributes.items():
            attr_arr = np.asarray(attr_values)
            num_components = attr_arr.shape[1] if attr_arr.ndim > 1 else 1
            log.info("Adding scalar field: %s (components: %d)", attr_name, num_components)
            scalar_array = ET.SubElement(point_data, 'DataArray',
                                          type='Float32',
                                          Name=attr_name,
                                          NumberOfComponents=str(num_components),
                                          format='ascii')
            scalar_array.text = ' '.join(map(str, attr_arr.flatten()))

    # CellData section (face attributes) - only for meshes with faces
    if not is_pc:
        cell_data = ET.SubElement(piece, 'CellData')

        # Add face attributes as scalar arrays
        if hasattr(trimesh, 'face_attributes') and trimesh.face_attributes:
            for attr_name, attr_values in trimesh.face_attributes.items():
                attr_arr = np.asarray(attr_values)
                # Skip high-dimensional arrays (e.g., 448-dim feature vectors)
                if attr_arr.ndim > 1 and attr_arr.shape[1] > 4:
                    log.info("Skipping high-dim field: %s (shape %s)", attr_name, attr_arr.shape)
                    continue
                log.info("Adding face field: %s", attr_name)
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
        log.info("Export complete: %d points", num_verts)
    else:
        log.info("Export complete: %d vertices, %d faces", num_verts, num_faces)
