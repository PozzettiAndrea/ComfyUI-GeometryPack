# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remesh GPU Node - GPU-accelerated remeshing using CuMesh
Requires CUDA, torch, and cumesh.
"""

import numpy as np
import trimesh as trimesh_module

from ..._utils import mesh_ops


class RemeshGPUNode:
    """
    Remesh GPU - GPU-accelerated dual-contouring remeshing using CuMesh.

    Uses the same algorithm as TRELLIS2 for high-quality mesh generation.
    Requires CUDA-capable GPU, torch, and cumesh package.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "target_face_count": ("INT", {
                    "default": 500000,
                    "min": 1000,
                    "max": 5000000,
                    "step": 1000,
                    "tooltip": "Target number of output faces after simplification.",
                }),
                "remesh_band": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Band width for dual-contouring. Affects surface detail capture. Higher = smoother but may lose fine details.",
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("remeshed_mesh", "info")
    FUNCTION = "remesh"
    CATEGORY = "geompack/remeshing"
    OUTPUT_NODE = True

    def remesh(self, trimesh, target_face_count=500000, remesh_band=1.0):
        """Apply GPU-accelerated CuMesh remeshing."""
        import torch
        import cumesh as CuMesh

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        print(f"\n{'='*60}")
        print(f"[Remesh GPU] Backend: cumesh (CUDA)")
        print(f"[Remesh GPU] Input: {initial_vertices:,} vertices, {initial_faces:,} faces")
        print(f"[Remesh GPU] Parameters: target_face_count={target_face_count:,}, remesh_band={remesh_band}")
        print(f"{'='*60}\n")

        # Hardcoded resolution = 512 (same as TRELLIS2)
        grid_resolution = 512

        remeshed_mesh, error = mesh_ops.cumesh_dc_remesh(
            trimesh, grid_resolution, fill_holes_first=False, band=remesh_band
        )
        if remeshed_mesh is None:
            raise ValueError(f"CuMesh remeshing failed: {error}")

        pre_simplify_faces = len(remeshed_mesh.faces)
        vertices = torch.tensor(remeshed_mesh.vertices, dtype=torch.float32).cuda()
        faces = torch.tensor(remeshed_mesh.faces, dtype=torch.int32).cuda()

        cumesh_obj = CuMesh.CuMesh()
        cumesh_obj.init(vertices, faces)

        # Skip pre-simplify unify on large meshes - CuMesh crashes on >2M faces
        if len(faces) < 2_000_000:
            cumesh_obj.unify_face_orientations()
            print(f"[Remesh GPU] Unified face orientations (pre-simplify)")
        else:
            print(f"[Remesh GPU] Skipping pre-simplify unify (mesh too large: {len(faces):,} faces)")

        # Simplify to target
        cumesh_obj.simplify(target_face_count, verbose=True)
        print(f"[Remesh GPU] After simplify: {cumesh_obj.num_faces:,} faces")

        # Unify after simplify (on smaller mesh, should work)
        cumesh_obj.unify_face_orientations()
        print(f"[Remesh GPU] Unified face orientations (post-simplify)")

        final_verts, final_faces = cumesh_obj.read()
        remeshed_mesh = trimesh_module.Trimesh(
            vertices=final_verts.cpu().numpy(),
            faces=final_faces.cpu().numpy(),
            process=False
        )

        # Preserve metadata
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'cumesh',
            'remesh_band': remesh_band,
            'target_face_count': target_face_count,
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces)
        }

        vertex_change = len(remeshed_mesh.vertices) - initial_vertices
        face_change = len(remeshed_mesh.faces) - initial_faces

        print(f"[Remesh GPU] Output: {len(remeshed_mesh.vertices)} vertices ({vertex_change:+d}), "
              f"{len(remeshed_mesh.faces)} faces ({face_change:+d})")

        info = f"""Remesh Results (CuMesh GPU):

Band Width: {remesh_band}
Target Face Count: {target_face_count:,}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After Remesh: {pre_simplify_faces:,} faces
After Simplify: {len(remeshed_mesh.faces):,} faces

GPU-accelerated dual contouring (same algorithm as TRELLIS2).
"""
        return {"ui": {"text": [info]}, "result": (remeshed_mesh, info)}


NODE_CLASS_MAPPINGS = {
    "GeomPackRemeshGPU": RemeshGPUNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemeshGPU": "Remesh GPU",
}
