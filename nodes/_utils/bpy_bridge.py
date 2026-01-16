# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Direct bpy (Blender as Python module) bridge using comfy-env isolation.

This module provides functions for Blender operations that run in an isolated
Python 3.11 environment with bpy installed. Functions are called via comfy-env's
VenvWorker.call_module() mechanism.

Data is passed via IPC (serialized as lists), so functions accept lists and
return lists/dicts with list values.

Usage from host environment:
    from comfy_env import VenvWorker
    worker = VenvWorker(python='_env_geometrypack/bin/python', sys_path=[...])
    result = worker.call_module('bpy_bridge', 'bpy_smart_uv_project', **kwargs)
"""

import numpy as np


def _to_list(arr):
    """Convert numpy array or list to nested list for IPC serialization."""
    if hasattr(arr, 'tolist'):
        return arr.tolist()
    return arr


def _to_numpy(data, dtype=np.float32):
    """Convert list to numpy array."""
    return np.array(data, dtype=dtype)


def bpy_smart_uv_project(vertices, faces, angle_limit, island_margin, scale_to_bounds):
    """
    Direct bpy Smart UV Project.

    Args:
        vertices: list of [x,y,z] vertex positions
        faces: list of [i,j,k] face indices
        angle_limit: Angle limit in radians for island creation
        island_margin: Margin between UV islands (0.0 to 1.0)
        scale_to_bounds: Whether to scale UVs to fill [0,1] bounds

    Returns:
        dict with 'vertices', 'faces', 'uvs' as lists
    """
    import bpy
    import bmesh

    # Convert inputs
    vertices = _to_numpy(vertices, np.float32)
    faces = _to_numpy(faces, np.int32)

    # Create new mesh and object
    mesh = bpy.data.meshes.new("UVMesh")
    obj = bpy.data.objects.new("UVObject", mesh)

    # Link to scene
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Create mesh from arrays
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()

    # Switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')

    # Select all faces
    bm = bmesh.from_edit_mesh(mesh)
    for face in bm.faces:
        face.select = True
    bmesh.update_edit_mesh(mesh)

    # Apply Smart UV Project
    bpy.ops.uv.smart_project(
        angle_limit=angle_limit,
        island_margin=island_margin,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=scale_to_bounds
    )

    # Switch back to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Extract results
    result_vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
    result_faces = np.array([p.vertices[:] for p in mesh.polygons], dtype=np.int32)

    # Extract UVs - handle vertex splitting at seams
    uv_layer = mesh.uv_layers.active
    if uv_layer:
        loop_uvs = np.array([uv_layer.data[i].uv[:] for i in range(len(uv_layer.data))], dtype=np.float32)

        new_vertices = []
        new_faces = []
        uvs_per_loop = []
        vertex_map = {}

        for poly in mesh.polygons:
            new_face = []
            for loop_idx in poly.loop_indices:
                orig_vert_idx = mesh.loops[loop_idx].vertex_index
                uv = tuple(loop_uvs[loop_idx])
                key = (orig_vert_idx, uv)

                if key not in vertex_map:
                    vertex_map[key] = len(new_vertices)
                    new_vertices.append(result_vertices[orig_vert_idx].tolist())
                    uvs_per_loop.append(list(uv))

                new_face.append(vertex_map[key])
            new_faces.append(new_face)

        result_vertices = new_vertices
        result_faces = new_faces
        result_uvs = uvs_per_loop
    else:
        result_vertices = _to_list(result_vertices)
        result_faces = _to_list(result_faces)
        result_uvs = [[0.0, 0.0]] * len(result_vertices)

    # Cleanup
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    return {
        'vertices': result_vertices,
        'faces': result_faces,
        'uvs': result_uvs
    }


def bpy_cube_uv_project(vertices, faces, cube_size, scale_to_bounds):
    """Direct bpy Cube UV Project."""
    import bpy
    import bmesh

    vertices = _to_numpy(vertices, np.float32)
    faces = _to_numpy(faces, np.int32)

    mesh = bpy.data.meshes.new("UVMesh")
    obj = bpy.data.objects.new("UVObject", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()

    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    for face in bm.faces:
        face.select = True
    bmesh.update_edit_mesh(mesh)

    bpy.ops.uv.cube_project(cube_size=cube_size, scale_to_bounds=scale_to_bounds)

    bpy.ops.object.mode_set(mode='OBJECT')

    result_vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
    result_faces = np.array([p.vertices[:] for p in mesh.polygons], dtype=np.int32)

    uv_layer = mesh.uv_layers.active
    if uv_layer:
        loop_uvs = np.array([uv_layer.data[i].uv[:] for i in range(len(uv_layer.data))], dtype=np.float32)
        new_vertices, new_faces, uvs_per_loop = [], [], []
        vertex_map = {}

        for poly in mesh.polygons:
            new_face = []
            for loop_idx in poly.loop_indices:
                orig_vert_idx = mesh.loops[loop_idx].vertex_index
                uv = tuple(loop_uvs[loop_idx])
                key = (orig_vert_idx, uv)
                if key not in vertex_map:
                    vertex_map[key] = len(new_vertices)
                    new_vertices.append(result_vertices[orig_vert_idx].tolist())
                    uvs_per_loop.append(list(uv))
                new_face.append(vertex_map[key])
            new_faces.append(new_face)

        result_vertices, result_faces, result_uvs = new_vertices, new_faces, uvs_per_loop
    else:
        result_vertices = _to_list(result_vertices)
        result_faces = _to_list(result_faces)
        result_uvs = [[0.0, 0.0]] * len(result_vertices)

    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    return {'vertices': result_vertices, 'faces': result_faces, 'uvs': result_uvs}


def bpy_cylinder_uv_project(vertices, faces, radius, scale_to_bounds):
    """Direct bpy Cylinder UV Project."""
    import bpy
    import bmesh

    vertices = _to_numpy(vertices, np.float32)
    faces = _to_numpy(faces, np.int32)

    mesh = bpy.data.meshes.new("UVMesh")
    obj = bpy.data.objects.new("UVObject", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()

    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    for face in bm.faces:
        face.select = True
    bmesh.update_edit_mesh(mesh)

    bpy.ops.uv.cylinder_project(radius=radius, scale_to_bounds=scale_to_bounds)

    bpy.ops.object.mode_set(mode='OBJECT')

    result_vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
    result_faces = np.array([p.vertices[:] for p in mesh.polygons], dtype=np.int32)

    uv_layer = mesh.uv_layers.active
    if uv_layer:
        loop_uvs = np.array([uv_layer.data[i].uv[:] for i in range(len(uv_layer.data))], dtype=np.float32)
        new_vertices, new_faces, uvs_per_loop = [], [], []
        vertex_map = {}

        for poly in mesh.polygons:
            new_face = []
            for loop_idx in poly.loop_indices:
                orig_vert_idx = mesh.loops[loop_idx].vertex_index
                uv = tuple(loop_uvs[loop_idx])
                key = (orig_vert_idx, uv)
                if key not in vertex_map:
                    vertex_map[key] = len(new_vertices)
                    new_vertices.append(result_vertices[orig_vert_idx].tolist())
                    uvs_per_loop.append(list(uv))
                new_face.append(vertex_map[key])
            new_faces.append(new_face)

        result_vertices, result_faces, result_uvs = new_vertices, new_faces, uvs_per_loop
    else:
        result_vertices = _to_list(result_vertices)
        result_faces = _to_list(result_faces)
        result_uvs = [[0.0, 0.0]] * len(result_vertices)

    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    return {'vertices': result_vertices, 'faces': result_faces, 'uvs': result_uvs}


def bpy_sphere_uv_project(vertices, faces, scale_to_bounds):
    """Direct bpy Sphere UV Project."""
    import bpy
    import bmesh

    vertices = _to_numpy(vertices, np.float32)
    faces = _to_numpy(faces, np.int32)

    mesh = bpy.data.meshes.new("UVMesh")
    obj = bpy.data.objects.new("UVObject", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()

    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    for face in bm.faces:
        face.select = True
    bmesh.update_edit_mesh(mesh)

    bpy.ops.uv.sphere_project(scale_to_bounds=scale_to_bounds)

    bpy.ops.object.mode_set(mode='OBJECT')

    result_vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
    result_faces = np.array([p.vertices[:] for p in mesh.polygons], dtype=np.int32)

    uv_layer = mesh.uv_layers.active
    if uv_layer:
        loop_uvs = np.array([uv_layer.data[i].uv[:] for i in range(len(uv_layer.data))], dtype=np.float32)
        new_vertices, new_faces, uvs_per_loop = [], [], []
        vertex_map = {}

        for poly in mesh.polygons:
            new_face = []
            for loop_idx in poly.loop_indices:
                orig_vert_idx = mesh.loops[loop_idx].vertex_index
                uv = tuple(loop_uvs[loop_idx])
                key = (orig_vert_idx, uv)
                if key not in vertex_map:
                    vertex_map[key] = len(new_vertices)
                    new_vertices.append(result_vertices[orig_vert_idx].tolist())
                    uvs_per_loop.append(list(uv))
                new_face.append(vertex_map[key])
            new_faces.append(new_face)

        result_vertices, result_faces, result_uvs = new_vertices, new_faces, uvs_per_loop
    else:
        result_vertices = _to_list(result_vertices)
        result_faces = _to_list(result_faces)
        result_uvs = [[0.0, 0.0]] * len(result_vertices)

    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    return {'vertices': result_vertices, 'faces': result_faces, 'uvs': result_uvs}


def bpy_voxel_remesh(vertices, faces, voxel_size):
    """Direct bpy voxel remesh."""
    import bpy

    vertices = _to_numpy(vertices, np.float32)
    faces = _to_numpy(faces, np.int32)

    mesh = bpy.data.meshes.new("RemeshMesh")
    obj = bpy.data.objects.new("RemeshObject", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()

    obj.data.remesh_voxel_size = voxel_size
    bpy.ops.object.voxel_remesh()

    mesh = obj.data
    result_vertices = [list(v.co) for v in mesh.vertices]
    result_faces = [list(p.vertices) for p in mesh.polygons]

    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    return {'vertices': result_vertices, 'faces': result_faces}


def bpy_quadriflow_remesh(vertices, faces, target_face_count):
    """Direct bpy Quadriflow remesh."""
    import bpy

    vertices = _to_numpy(vertices, np.float32)
    faces = _to_numpy(faces, np.int32)

    mesh = bpy.data.meshes.new("RemeshMesh")
    obj = bpy.data.objects.new("RemeshObject", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()

    bpy.ops.object.quadriflow_remesh(
        use_mesh_symmetry=False,
        use_preserve_sharp=False,
        use_preserve_boundary=False,
        smooth_normals=False,
        mode='FACES',
        target_faces=target_face_count,
        seed=0
    )

    mesh = obj.data
    result_vertices = [list(v.co) for v in mesh.vertices]
    result_faces = [list(p.vertices) for p in mesh.polygons]

    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    return {'vertices': result_vertices, 'faces': result_faces}


def bpy_boolean_operation(vertices_a, faces_a, vertices_b, faces_b, operation):
    """
    Direct bpy boolean operation with EXACT solver.

    Args:
        vertices_a, faces_a: Mesh A data
        vertices_b, faces_b: Mesh B data
        operation: One of 'UNION', 'DIFFERENCE', 'INTERSECT'

    Returns:
        dict with 'vertices', 'faces' as lists
    """
    import bpy

    vertices_a = _to_numpy(vertices_a, np.float32)
    faces_a = _to_numpy(faces_a, np.int32)
    vertices_b = _to_numpy(vertices_b, np.float32)
    faces_b = _to_numpy(faces_b, np.int32)

    # Create mesh A
    mesh_a = bpy.data.meshes.new("MeshA")
    obj_a = bpy.data.objects.new("ObjectA", mesh_a)
    bpy.context.collection.objects.link(obj_a)
    mesh_a.from_pydata(vertices_a.tolist(), [], faces_a.tolist())
    mesh_a.update()

    # Create mesh B
    mesh_b = bpy.data.meshes.new("MeshB")
    obj_b = bpy.data.objects.new("ObjectB", mesh_b)
    bpy.context.collection.objects.link(obj_b)
    mesh_b.from_pydata(vertices_b.tolist(), [], faces_b.tolist())
    mesh_b.update()

    # Select A as active
    bpy.ops.object.select_all(action='DESELECT')
    obj_a.select_set(True)
    bpy.context.view_layer.objects.active = obj_a

    # Add boolean modifier
    bool_mod = obj_a.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_mod.operation = operation
    bool_mod.object = obj_b
    bool_mod.solver = 'EXACT'

    # Apply modifier
    bpy.ops.object.modifier_apply(modifier="Boolean")

    mesh_a = obj_a.data
    result_vertices = [list(v.co) for v in mesh_a.vertices]
    result_faces = [list(p.vertices) for p in mesh_a.polygons]

    # Cleanup
    bpy.data.objects.remove(obj_b, do_unlink=True)
    bpy.data.meshes.remove(mesh_b)
    bpy.data.objects.remove(obj_a, do_unlink=True)
    bpy.data.meshes.remove(mesh_a)

    return {'vertices': result_vertices, 'faces': result_faces}


def bpy_import_blend(blend_path):
    """
    Import .blend file and extract mesh data.

    Args:
        blend_path: Path to .blend file

    Returns:
        dict with 'vertices', 'faces' as lists, 'name' as string
    """
    import bpy

    bpy.ops.wm.open_mainfile(filepath=blend_path)

    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

    if not mesh_objects:
        return {'vertices': [], 'faces': [], 'name': 'empty'}

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for obj in mesh_objects:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        mesh.transform(obj.matrix_world)

        verts = [list(v.co) for v in mesh.vertices]

        for poly in mesh.polygons:
            if len(poly.vertices) == 3:
                all_faces.append([
                    poly.vertices[0] + vertex_offset,
                    poly.vertices[1] + vertex_offset,
                    poly.vertices[2] + vertex_offset
                ])
            elif len(poly.vertices) > 3:
                # Fan triangulation
                v0 = poly.vertices[0]
                for i in range(1, len(poly.vertices) - 1):
                    all_faces.append([
                        v0 + vertex_offset,
                        poly.vertices[i] + vertex_offset,
                        poly.vertices[i+1] + vertex_offset
                    ])

        all_vertices.extend(verts)
        vertex_offset += len(verts)
        obj_eval.to_mesh_clear()

    return {
        'vertices': all_vertices,
        'faces': all_faces,
        'name': mesh_objects[0].name if mesh_objects else 'combined'
    }


def bpy_import_fbx(fbx_path):
    """
    Import .fbx file and extract mesh data.

    Args:
        fbx_path: Path to .fbx file

    Returns:
        dict with 'vertices', 'faces' as lists, 'name' as string
    """
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

    if not mesh_objects:
        return {'vertices': [], 'faces': [], 'name': 'empty'}

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for obj in mesh_objects:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        mesh.transform(obj.matrix_world)

        verts = [list(v.co) for v in mesh.vertices]

        for poly in mesh.polygons:
            if len(poly.vertices) == 3:
                all_faces.append([
                    poly.vertices[0] + vertex_offset,
                    poly.vertices[1] + vertex_offset,
                    poly.vertices[2] + vertex_offset
                ])
            elif len(poly.vertices) > 3:
                v0 = poly.vertices[0]
                for i in range(1, len(poly.vertices) - 1):
                    all_faces.append([
                        v0 + vertex_offset,
                        poly.vertices[i] + vertex_offset,
                        poly.vertices[i+1] + vertex_offset
                    ])

        all_vertices.extend(verts)
        vertex_offset += len(verts)
        obj_eval.to_mesh_clear()

    return {
        'vertices': all_vertices,
        'faces': all_faces,
        'name': mesh_objects[0].name if mesh_objects else 'combined'
    }
