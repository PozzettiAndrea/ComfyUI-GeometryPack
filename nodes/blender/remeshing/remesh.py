# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remesh Blender Node - Blender remeshing backends using bpy.
Supports: voxel, smooth, sharp, blocks.
Requires bpy (Blender Python module).
"""

import numpy as np
import trimesh as trimesh_module


def _bpy_setup_object(vertices, faces):
    """Create a Blender mesh object from vertices and faces. Returns (obj, mesh)."""
    import bpy

    mesh = bpy.data.meshes.new("RemeshMesh")
    obj = bpy.data.objects.new("RemeshObject", mesh)
    bpy.context.collection.objects.link(obj)

    # Deselect everything first (default scene has a Cube selected)
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

    # Triangulate polygons with >3 verts (safety for any backend producing quads)
    result_faces = []
    for p in mesh.polygons:
        verts = list(p.vertices)
        if len(verts) == 3:
            result_faces.append(verts)
        elif len(verts) == 4:
            # Split quad into 2 triangles
            result_faces.append([verts[0], verts[1], verts[2]])
            result_faces.append([verts[0], verts[2], verts[3]])
        else:
            # Fan triangulation for n-gons
            for i in range(1, len(verts) - 1):
                result_faces.append([verts[0], verts[i], verts[i + 1]])

    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    return {'vertices': result_vertices, 'faces': result_faces}


def _bpy_voxel_remesh(vertices, faces, voxel_size):
    """Blender voxel remesh using bpy."""
    import bpy

    obj, mesh = _bpy_setup_object(vertices, faces)

    obj.data.remesh_voxel_size = voxel_size
    bpy.ops.object.voxel_remesh()

    return _bpy_extract_and_cleanup(obj)


def _bpy_remesh_modifier(vertices, faces, mode, octree_depth=6, scale=0.9, sharpness=1.0):
    """Blender Remesh Modifier (Smooth/Sharp/Blocks) using bpy."""
    import bpy

    obj, mesh = _bpy_setup_object(vertices, faces)

    mod = obj.modifiers.new(name="Remesh", type='REMESH')
    mod.mode = mode  # 'SMOOTH', 'SHARP', or 'BLOCKS'
    mod.octree_depth = octree_depth
    mod.scale = scale
    if mode == 'SHARP':
        mod.sharpness = sharpness

    bpy.ops.object.modifier_apply(modifier="Remesh")

    return _bpy_extract_and_cleanup(obj)


class RemeshBlenderNode:
    """
    Remesh Blender - Blender-based remeshing using bpy.

    Available backends:
    - blender_voxel: Voxel-based remeshing (watertight output)
    - blender_smooth: Smooth remesh modifier
    - blender_sharp: Sharp remesh modifier (preserves edges)
    - blender_blocks: Blocky remesh modifier

    Requires bpy (Blender Python module) to be installed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "backend": ([
                    "blender_voxel",
                    "blender_smooth",
                    "blender_sharp",
                    "blender_blocks",
                ], {
                    "default": "blender_voxel",
                    "tooltip": "Remeshing algorithm. voxel=watertight, smooth/sharp/blocks=modifier-based"
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
                # Modifier-based (Smooth/Sharp/Blocks)
                "octree_depth": ("INT", {
                    "default": 6,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Resolution of the remesh. Higher = more detail, more faces.",
                    "visible_when": {"backend": ["blender_smooth", "blender_sharp", "blender_blocks"]},
                }),
                "scale": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "number",
                    "tooltip": "Ratio of output size to input bounding box.",
                    "visible_when": {"backend": ["blender_smooth", "blender_sharp", "blender_blocks"]},
                }),
                "sharpness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Edge sharpness for Sharp mode.",
                    "visible_when": {"backend": ["blender_sharp"]},
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("remeshed_mesh", "info")
    FUNCTION = "remesh"
    CATEGORY = "geompack/remeshing"
    OUTPUT_NODE = True

    def remesh(self, trimesh, backend, voxel_size=1.0,
               octree_depth=6, scale=0.9, sharpness=1.0):
        """Apply Blender-based remeshing."""
        # Sanitize hidden widget values (ComfyUI sends '' for hidden visible_when widgets)
        voxel_size = float(voxel_size) if voxel_size not in (None, '') else 1.0
        octree_depth = int(octree_depth) if octree_depth not in (None, '') else 6
        scale = float(scale) if scale not in (None, '') else 0.9
        sharpness = float(sharpness) if sharpness not in (None, '') else 1.0

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        print(f"\n{'='*60}")
        print(f"[Remesh Blender] Backend: {backend}")
        print(f"[Remesh Blender] Input: {initial_vertices:,} vertices, {initial_faces:,} faces")

        if backend == "blender_voxel":
            print(f"[Remesh Blender] Parameters: voxel_size={voxel_size}")
            remeshed_mesh, info = self._blender_voxel(trimesh, voxel_size)
        elif backend in ("blender_smooth", "blender_sharp", "blender_blocks"):
            mode = backend.replace("blender_", "").upper()
            print(f"[Remesh Blender] Parameters: mode={mode}, octree_depth={octree_depth}, scale={scale}"
                  + (f", sharpness={sharpness}" if backend == "blender_sharp" else ""))
            remeshed_mesh, info = self._blender_modifier(trimesh, mode, octree_depth, scale, sharpness)
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

    def _blender_modifier(self, trimesh, mode, octree_depth, scale, sharpness):
        """Blender Remesh Modifier (Smooth/Sharp/Blocks)."""
        print(f"[Remesh Blender] Running Blender Remesh Modifier (mode={mode}, depth={octree_depth})...")
        result = _bpy_remesh_modifier(
            vertices=np.asarray(trimesh.vertices, dtype=np.float32),
            faces=np.asarray(trimesh.faces, dtype=np.int32),
            mode=mode,
            octree_depth=octree_depth,
            scale=scale,
            sharpness=sharpness
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=np.array(result['vertices'], dtype=np.float32),
            faces=np.array(result['faces'], dtype=np.int32),
            process=False
        )

        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': f'blender_{mode.lower()}',
            'octree_depth': octree_depth,
            'scale': scale,
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces)
        }

        mode_label = mode.capitalize()
        params = f"Octree Depth: {octree_depth}\nScale: {scale}"
        if mode == 'SHARP':
            params += f"\nSharpness: {sharpness}"

        info = f"""Remesh Results (Blender {mode_label}):

{params}
Method: bpy (Remesh Modifier)

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}
"""
        return remeshed_mesh, info


NODE_CLASS_MAPPINGS = {
    "GeomPackRemeshBlender": RemeshBlenderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemeshBlender": "Remesh Blender",
}
