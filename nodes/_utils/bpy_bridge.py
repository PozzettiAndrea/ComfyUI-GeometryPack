# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Direct bpy (Blender as Python module) bridge using comfy-env isolation.

This module provides isolated functions for Blender operations using the
comfy-env @isolated decorator. Operations run in a separate Python 3.11
environment with bpy installed, avoiding subprocess overhead and temp file I/O.

All functions accept numpy arrays and return numpy arrays directly via IPC.
"""

from comfy_env import isolated
import numpy as np


@isolated(env="geometrypack", import_paths=[".", ".."])
def bpy_smart_uv_project(vertices, faces, angle_limit, island_margin, scale_to_bounds):
    """
    Direct bpy Smart UV Project.

    Args:
        vertices: numpy array of shape (N, 3) with vertex positions
        faces: numpy array of shape (M, 3) with face indices
        angle_limit: Angle limit in radians for island creation
        island_margin: Margin between UV islands (0.0 to 1.0)
        scale_to_bounds: Whether to scale UVs to fill [0,1] bounds

    Returns:
        dict with 'vertices', 'faces', 'uvs' as numpy arrays
    """
    import bpy
    import bmesh
    import numpy as np

    # Create new mesh and object
    mesh = bpy.data.meshes.new("UVMesh")
    obj = bpy.data.objects.new("UVObject", mesh)

    # Link to scene
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Create mesh from numpy arrays
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

    # Extract UVs
    uv_layer = mesh.uv_layers.active
    if uv_layer:
        # UVs are per-loop, need to reorganize per-vertex
        # For simplicity, we'll duplicate vertices at UV seams (like the OBJ export does)
        loop_uvs = np.array([uv_layer.data[i].uv[:] for i in range(len(uv_layer.data))], dtype=np.float32)

        # Build per-face-vertex UVs
        uvs_per_loop = []
        new_vertices = []
        new_faces = []
        vertex_map = {}  # (original_vertex_idx, uv_tuple) -> new_vertex_idx

        for poly in mesh.polygons:
            new_face = []
            for loop_idx in poly.loop_indices:
                orig_vert_idx = mesh.loops[loop_idx].vertex_index
                uv = tuple(loop_uvs[loop_idx])
                key = (orig_vert_idx, uv)

                if key not in vertex_map:
                    vertex_map[key] = len(new_vertices)
                    new_vertices.append(result_vertices[orig_vert_idx])
                    uvs_per_loop.append(uv)

                new_face.append(vertex_map[key])
            new_faces.append(new_face)

        result_vertices = np.array(new_vertices, dtype=np.float32)
        result_faces = np.array(new_faces, dtype=np.int32)
        result_uvs = np.array(uvs_per_loop, dtype=np.float32)
    else:
        result_uvs = np.zeros((len(result_vertices), 2), dtype=np.float32)

    # Cleanup
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    return {
        'vertices': result_vertices,
        'faces': result_faces,
        'uvs': result_uvs
    }


@isolated(env="geometrypack", import_paths=[".", ".."])
def bpy_cube_uv_project(vertices, faces, cube_size, scale_to_bounds):
    """
    Direct bpy Cube UV Project.

    Args:
        vertices: numpy array of shape (N, 3) with vertex positions
        faces: numpy array of shape (M, 3) with face indices
        cube_size: Size of the projection cube
        scale_to_bounds: Whether to scale UVs to fill [0,1] bounds

    Returns:
        dict with 'vertices', 'faces', 'uvs' as numpy arrays
    """
    import bpy
    import bmesh
    import numpy as np

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
                    new_vertices.append(result_vertices[orig_vert_idx])
                    uvs_per_loop.append(uv)
                new_face.append(vertex_map[key])
            new_faces.append(new_face)

        result_vertices = np.array(new_vertices, dtype=np.float32)
        result_faces = np.array(new_faces, dtype=np.int32)
        result_uvs = np.array(uvs_per_loop, dtype=np.float32)
    else:
        result_uvs = np.zeros((len(result_vertices), 2), dtype=np.float32)

    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    return {'vertices': result_vertices, 'faces': result_faces, 'uvs': result_uvs}


@isolated(env="geometrypack", import_paths=[".", ".."])
def bpy_cylinder_uv_project(vertices, faces, radius, scale_to_bounds):
    """
    Direct bpy Cylinder UV Project.

    Args:
        vertices: numpy array of shape (N, 3) with vertex positions
        faces: numpy array of shape (M, 3) with face indices
        radius: Cylinder radius for projection
        scale_to_bounds: Whether to scale UVs to fill [0,1] bounds

    Returns:
        dict with 'vertices', 'faces', 'uvs' as numpy arrays
    """
    import bpy
    import bmesh
    import numpy as np

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
                    new_vertices.append(result_vertices[orig_vert_idx])
                    uvs_per_loop.append(uv)
                new_face.append(vertex_map[key])
            new_faces.append(new_face)

        result_vertices = np.array(new_vertices, dtype=np.float32)
        result_faces = np.array(new_faces, dtype=np.int32)
        result_uvs = np.array(uvs_per_loop, dtype=np.float32)
    else:
        result_uvs = np.zeros((len(result_vertices), 2), dtype=np.float32)

    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    return {'vertices': result_vertices, 'faces': result_faces, 'uvs': result_uvs}


@isolated(env="geometrypack", import_paths=[".", ".."])
def bpy_sphere_uv_project(vertices, faces, scale_to_bounds):
    """
    Direct bpy Sphere UV Project.

    Args:
        vertices: numpy array of shape (N, 3) with vertex positions
        faces: numpy array of shape (M, 3) with face indices
        scale_to_bounds: Whether to scale UVs to fill [0,1] bounds

    Returns:
        dict with 'vertices', 'faces', 'uvs' as numpy arrays
    """
    import bpy
    import bmesh
    import numpy as np

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
                    new_vertices.append(result_vertices[orig_vert_idx])
                    uvs_per_loop.append(uv)
                new_face.append(vertex_map[key])
            new_faces.append(new_face)

        result_vertices = np.array(new_vertices, dtype=np.float32)
        result_faces = np.array(new_faces, dtype=np.int32)
        result_uvs = np.array(uvs_per_loop, dtype=np.float32)
    else:
        result_uvs = np.zeros((len(result_vertices), 2), dtype=np.float32)

    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    return {'vertices': result_vertices, 'faces': result_faces, 'uvs': result_uvs}


@isolated(env="geometrypack", import_paths=[".", ".."])
def bpy_voxel_remesh(vertices, faces, voxel_size):
    """
    Direct bpy voxel remesh.

    Args:
        vertices: numpy array of shape (N, 3) with vertex positions
        faces: numpy array of shape (M, 3) with face indices
        voxel_size: Voxel size for remeshing

    Returns:
        dict with 'vertices', 'faces' as numpy arrays
    """
    import bpy
    import numpy as np

    mesh = bpy.data.meshes.new("RemeshMesh")
    obj = bpy.data.objects.new("RemeshObject", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()

    # Apply voxel remesh
    obj.data.remesh_voxel_size = voxel_size
    bpy.ops.object.voxel_remesh()

    # Get updated mesh reference after remesh
    mesh = obj.data

    result_vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
    result_faces = np.array([p.vertices[:] for p in mesh.polygons], dtype=np.int32)

    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    return {'vertices': result_vertices, 'faces': result_faces}


@isolated(env="geometrypack", import_paths=[".", ".."])
def bpy_quadriflow_remesh(vertices, faces, target_face_count):
    """
    Direct bpy Quadriflow remesh.

    Args:
        vertices: numpy array of shape (N, 3) with vertex positions
        faces: numpy array of shape (M, 3) with face indices
        target_face_count: Target number of faces

    Returns:
        dict with 'vertices', 'faces' as numpy arrays
    """
    import bpy
    import numpy as np

    mesh = bpy.data.meshes.new("RemeshMesh")
    obj = bpy.data.objects.new("RemeshObject", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()

    # Apply Quadriflow remesh
    bpy.ops.object.quadriflow_remesh(
        use_mesh_symmetry=False,
        use_preserve_sharp=False,
        use_preserve_boundary=False,
        smooth_normals=False,
        mode='FACES',
        target_faces=target_face_count,
        seed=0
    )

    # Get updated mesh reference after remesh
    mesh = obj.data

    result_vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
    result_faces = np.array([p.vertices[:] for p in mesh.polygons], dtype=np.int32)

    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    return {'vertices': result_vertices, 'faces': result_faces}


@isolated(env="geometrypack", import_paths=[".", ".."])
def bpy_boolean_operation(vertices_a, faces_a, vertices_b, faces_b, operation):
    """
    Direct bpy boolean operation with EXACT solver.

    Args:
        vertices_a: numpy array of shape (N, 3) - mesh A vertices
        faces_a: numpy array of shape (M, 3) - mesh A faces
        vertices_b: numpy array of shape (P, 3) - mesh B vertices
        faces_b: numpy array of shape (Q, 3) - mesh B faces
        operation: One of 'UNION', 'DIFFERENCE', 'INTERSECT'

    Returns:
        dict with 'vertices', 'faces' as numpy arrays
    """
    import bpy
    import numpy as np

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

    # Get result from mesh A (which now has the boolean result)
    mesh_a = obj_a.data
    result_vertices = np.array([v.co[:] for v in mesh_a.vertices], dtype=np.float32)
    result_faces = np.array([p.vertices[:] for p in mesh_a.polygons], dtype=np.int32)

    # Cleanup
    bpy.data.objects.remove(obj_b, do_unlink=True)
    bpy.data.meshes.remove(mesh_b)
    bpy.data.objects.remove(obj_a, do_unlink=True)
    bpy.data.meshes.remove(mesh_a)

    return {'vertices': result_vertices, 'faces': result_faces}


@isolated(env="geometrypack", import_paths=[".", ".."])
def bpy_import_blend(blend_path):
    """
    Import .blend file and extract mesh data.

    Args:
        blend_path: Path to .blend file

    Returns:
        dict with 'vertices', 'faces' as numpy arrays, 'name' as string
    """
    import bpy
    import numpy as np

    # Load the blend file
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    # Find mesh objects
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

    if not mesh_objects:
        return {'vertices': np.array([], dtype=np.float32).reshape(0, 3),
                'faces': np.array([], dtype=np.int32).reshape(0, 3),
                'name': 'empty'}

    # Combine all meshes
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for obj in mesh_objects:
        # Get evaluated mesh (with modifiers applied)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()

        # Apply object transform
        mesh.transform(obj.matrix_world)

        verts = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
        faces = np.array([p.vertices[:] for p in mesh.polygons if len(p.vertices) == 3], dtype=np.int32)

        # Triangulate non-triangle faces
        for poly in mesh.polygons:
            if len(poly.vertices) > 3:
                # Simple fan triangulation
                v0 = poly.vertices[0]
                for i in range(1, len(poly.vertices) - 1):
                    all_faces.append([v0 + vertex_offset,
                                      poly.vertices[i] + vertex_offset,
                                      poly.vertices[i+1] + vertex_offset])

        all_vertices.append(verts)
        if len(faces) > 0:
            all_faces.extend((faces + vertex_offset).tolist())

        vertex_offset += len(verts)
        obj_eval.to_mesh_clear()

    result_vertices = np.vstack(all_vertices) if all_vertices else np.array([], dtype=np.float32).reshape(0, 3)
    result_faces = np.array(all_faces, dtype=np.int32) if all_faces else np.array([], dtype=np.int32).reshape(0, 3)

    return {
        'vertices': result_vertices,
        'faces': result_faces,
        'name': mesh_objects[0].name if mesh_objects else 'combined'
    }


@isolated(env="geometrypack", import_paths=[".", ".."])
def bpy_import_fbx(fbx_path):
    """
    Import .fbx file and extract mesh data.

    Args:
        fbx_path: Path to .fbx file

    Returns:
        dict with 'vertices', 'faces' as numpy arrays, 'name' as string
    """
    import bpy
    import numpy as np

    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import FBX
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # Find mesh objects
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

    if not mesh_objects:
        return {'vertices': np.array([], dtype=np.float32).reshape(0, 3),
                'faces': np.array([], dtype=np.int32).reshape(0, 3),
                'name': 'empty'}

    # Combine all meshes
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for obj in mesh_objects:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        mesh.transform(obj.matrix_world)

        verts = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
        faces = np.array([p.vertices[:] for p in mesh.polygons if len(p.vertices) == 3], dtype=np.int32)

        for poly in mesh.polygons:
            if len(poly.vertices) > 3:
                v0 = poly.vertices[0]
                for i in range(1, len(poly.vertices) - 1):
                    all_faces.append([v0 + vertex_offset,
                                      poly.vertices[i] + vertex_offset,
                                      poly.vertices[i+1] + vertex_offset])

        all_vertices.append(verts)
        if len(faces) > 0:
            all_faces.extend((faces + vertex_offset).tolist())

        vertex_offset += len(verts)
        obj_eval.to_mesh_clear()

    result_vertices = np.vstack(all_vertices) if all_vertices else np.array([], dtype=np.float32).reshape(0, 3)
    result_faces = np.array(all_faces, dtype=np.int32) if all_faces else np.array([], dtype=np.int32).reshape(0, 3)

    return {
        'vertices': result_vertices,
        'faces': result_faces,
        'name': mesh_objects[0].name if mesh_objects else 'combined'
    }
