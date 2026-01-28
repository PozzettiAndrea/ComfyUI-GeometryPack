# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remesh Blender Node - Blender voxel and quadriflow remeshing
Requires bpy (Blender Python module).
"""

import numpy as np
import trimesh as trimesh_module


def _bpy_voxel_remesh(vertices, faces, voxel_size):
    """Blender voxel remesh using bpy."""
    from ..._utils import setup_bpy_dll_path
    setup_bpy_dll_path()
    import bpy

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


def _bpy_quadriflow_remesh(vertices, faces, target_face_count):
    """Blender Quadriflow remesh using bpy."""
    from ..._utils import setup_bpy_dll_path
    setup_bpy_dll_path()
    import bpy

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


class RemeshBlenderNode:
    """
    Remesh Blender - Blender-based remeshing using bpy.

    Available backends:
    - blender_voxel: Voxel-based remeshing (watertight output)
    - blender_quadriflow: Quadriflow quad remeshing

    Requires bpy (Blender Python module) to be installed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "backend": ([
                    "blender_voxel",
                    "blender_quadriflow",
                ], {
                    "default": "blender_voxel",
                    "tooltip": "Remeshing algorithm. blender_voxel=watertight output, blender_quadriflow=quad remesh"
                }),
            },
            "optional": {
                # Blender voxel
                "voxel_size": ("FLOAT", {
                    "default": 1,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Voxel size for Blender voxel remesh. Smaller = more detail, more faces. Output is always watertight.",
                    "visible_when": {"backend": ["blender_voxel"]},
                }),
                # Quadriflow
                "target_face_count": ("INT", {
                    "default": 500000,
                    "min": 1000,
                    "max": 5000000,
                    "step": 1000,
                    "tooltip": "Target number of output faces for quadriflow backend.",
                    "visible_when": {"backend": ["blender_quadriflow"]},
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("remeshed_mesh", "info")
    FUNCTION = "remesh"
    CATEGORY = "geompack/remeshing"
    OUTPUT_NODE = True

    def remesh(self, trimesh, backend, voxel_size=1.0, target_face_count=500000):
        """Apply Blender-based remeshing."""
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        print(f"\n{'='*60}")
        print(f"[Remesh Blender] Backend: {backend}")
        print(f"[Remesh Blender] Input: {initial_vertices:,} vertices, {initial_faces:,} faces")

        if backend == "blender_voxel":
            print(f"[Remesh Blender] Parameters: voxel_size={voxel_size}")
            remeshed_mesh, info = self._blender_voxel(trimesh, voxel_size)
        elif backend == "blender_quadriflow":
            print(f"[Remesh Blender] Parameters: target_face_count={target_face_count:,}")
            remeshed_mesh, info = self._blender_quadriflow(trimesh, target_face_count)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        print(f"{'='*60}\n")

        vertex_change = len(remeshed_mesh.vertices) - initial_vertices
        face_change = len(remeshed_mesh.faces) - initial_faces

        print(f"[Remesh Blender] Output: {len(remeshed_mesh.vertices)} vertices ({vertex_change:+d}), "
              f"{len(remeshed_mesh.faces)} faces ({face_change:+d})")

        return {"ui": {"text": [info]}, "result": (remeshed_mesh, info)}

    def _blender_voxel(self, trimesh, voxel_size):
        """Blender voxel remeshing using bpy."""
        print(f"[Remesh Blender] Running Blender voxel remesh (voxel_size={voxel_size})...")
        result = _bpy_voxel_remesh(
            vertices=np.asarray(trimesh.vertices, dtype=np.float32),
            faces=np.asarray(trimesh.faces, dtype=np.int32),
            voxel_size=voxel_size
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=np.array(result['vertices'], dtype=np.float32),
            faces=np.array(result['faces'], dtype=np.int32),
            process=False
        )

        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'blender_voxel',
            'voxel_size': voxel_size,
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces)
        }

        info = f"""Remesh Results (Blender Voxel):

Voxel Size: {voxel_size}
Method: bpy

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}
"""
        return remeshed_mesh, info

    def _blender_quadriflow(self, trimesh, target_face_count):
        """Blender Quadriflow remeshing using bpy."""
        print(f"[Remesh Blender] Running Blender Quadriflow (target_faces={target_face_count})...")
        result = _bpy_quadriflow_remesh(
            vertices=np.asarray(trimesh.vertices, dtype=np.float32),
            faces=np.asarray(trimesh.faces, dtype=np.int32),
            target_face_count=target_face_count
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=np.array(result['vertices'], dtype=np.float32),
            faces=np.array(result['faces'], dtype=np.int32),
            process=False
        )

        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'blender_quadriflow',
            'target_face_count': target_face_count,
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces)
        }

        info = f"""Remesh Results (Blender Quadriflow):

Target Face Count: {target_face_count:,}
Method: bpy

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}

Quadriflow creates quad-dominant meshes with good topology.
"""
        return remeshed_mesh, info


NODE_CLASS_MAPPINGS = {
    "GeomPackRemeshBlender": RemeshBlenderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemeshBlender": "Remesh Blender",
}
