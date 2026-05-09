# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Blender voxel remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _bpy_setup_object(vertices, faces):
    """Create a Blender mesh object from vertices and faces."""
    import bpy
    mesh = bpy.data.meshes.new("RemeshMesh")
    obj = bpy.data.objects.new("RemeshObject", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    return obj, mesh


def _bpy_extract_and_cleanup(obj):
    """Extract vertices/faces from object, then delete it. Triangulates quads/n-gons."""
    import bpy
    mesh = obj.data
    result_vertices = [list(v.co) for v in mesh.vertices]
    result_faces = []
    for p in mesh.polygons:
        verts = list(p.vertices)
        if len(verts) == 3:
            result_faces.append(verts)
        elif len(verts) == 4:
            result_faces.append([verts[0], verts[1], verts[2]])
            result_faces.append([verts[0], verts[2], verts[3]])
        else:
            for i in range(1, len(verts) - 1):
                result_faces.append([verts[0], verts[i], verts[i + 1]])
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)
    return {'vertices': result_vertices, 'faces': result_faces}


class RemeshBlenderVoxelNode(io.ComfyNode):
    """Blender voxel remeshing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_BlenderVoxel",
            display_name="Remesh Blender Voxel (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("voxel_size", default=1, min=0.001, max=1.0, step=0.01, display_mode="number", tooltip="Voxel size. Smaller = more detail, more faces. Output is always watertight."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, voxel_size=1.0):
        import bpy

        log.info("Backend: blender_voxel")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: voxel_size=%s", voxel_size)

        obj, mesh = _bpy_setup_object(
            np.asarray(trimesh.vertices, dtype=np.float32),
            np.asarray(trimesh.faces, dtype=np.int32)
        )
        obj.data.remesh_voxel_size = voxel_size
        bpy.ops.object.voxel_remesh()
        result = _bpy_extract_and_cleanup(obj)

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=np.array(result['vertices'], dtype=np.float32),
            faces=np.array(result['faces'], dtype=np.int32),
            process=False
        )
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {'algorithm': 'blender_voxel', 'voxel_size': voxel_size}

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (Blender Voxel): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"voxel_size={voxel_size}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_BlenderVoxel": RemeshBlenderVoxelNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_BlenderVoxel": "Remesh Blender Voxel (backend)"}
