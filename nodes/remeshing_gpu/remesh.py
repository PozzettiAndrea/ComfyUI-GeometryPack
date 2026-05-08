# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remesh GPU Node - GPU-accelerated remeshing using CuMesh
Requires CUDA, torch, and cumesh.
"""

import logging
from typing import Tuple, Optional

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def cumesh_dc_remesh(
    mesh: trimesh_module.Trimesh,
    grid_resolution: int = 128,
    fill_holes_first: bool = True,
    band: float = 1.0,
) -> Tuple[Optional[trimesh_module.Trimesh], str]:
    """
    GPU-accelerated dual-contouring remeshing using CuMesh.

    Uses the same algorithm as TRELLIS2: CuMesh.remeshing.remesh_narrow_band_dc()
    """
    # Lazy imports - only available in isolated env
    import torch
    import cumesh as CuMesh

    try:
        log.info("Input: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
        log.info("Grid resolution: %d, band: %s", grid_resolution, band)

        # Convert to GPU tensors
        import comfy.model_management
        device = comfy.model_management.get_torch_device()
        assert device.type == "cuda", f"CuMesh requires CUDA but got device '{device}' — cumesh is GPU-only, no CPU fallback"
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)
        faces = torch.tensor(mesh.faces, dtype=torch.int32).to(device)

        # Calculate bounding box and scale
        bbox_min = vertices.min(dim=0).values
        bbox_max = vertices.max(dim=0).values
        bbox_size = bbox_max - bbox_min
        scale = bbox_size.max().item()

        # Center the mesh
        center = (bbox_min + bbox_max) / 2
        vertices_centered = vertices - center

        # Initialize CuMesh for pre-processing
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices_centered, faces)

        # Pre-unify face orientations
        cumesh.unify_face_orientations()

        # Optionally fill holes
        if fill_holes_first:
            cumesh.fill_holes()
            log.info("Filled holes")

        # Read current state after preprocessing
        curr_verts, curr_faces = cumesh.read()

        # Build BVH for the remeshing operation
        bvh = CuMesh.cuBVH(curr_verts, curr_faces)

        # Run dual-contouring remesh
        log.info("Running dual-contouring remesh...")
        new_verts, new_faces = CuMesh.remeshing.remesh_narrow_band_dc(
            curr_verts, curr_faces,
            center=torch.zeros(3, device='cuda'),
            scale=(grid_resolution + 3 * band) / grid_resolution * scale,
            resolution=grid_resolution,
            band=band,
            project_back=0.0,
            verbose=True,
            bvh=bvh,
        )

        # Clean up BVH
        del bvh, curr_verts, curr_faces

        log.info("After remesh: %d vertices, %d faces", len(new_verts), len(new_faces))

        # Restore center offset
        final_verts = new_verts + center

        # Create result mesh
        remeshed_mesh = trimesh_module.Trimesh(
            vertices=final_verts.cpu().numpy().astype(np.float32),
            faces=new_faces.cpu().numpy(),
            process=False
        )

        # Cleanup GPU memory
        del cumesh, vertices, faces, vertices_centered
        del new_verts, new_faces, final_verts
        import comfy.model_management
        comfy.model_management.soft_empty_cache()

        return remeshed_mesh, ""

    except Exception as e:
        import traceback
        log.error("CuMesh remesh failed", exc_info=True)
        return None, f"Error during CuMesh remesh: {str(e)}"


class RemeshGPUNode(io.ComfyNode):
    """
    Remesh GPU - GPU-accelerated dual-contouring remeshing using CuMesh.

    Uses the same algorithm as TRELLIS2 for high-quality mesh generation.
    Requires CUDA-capable GPU, torch, and cumesh package.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_GPU",
            display_name="Remesh GPU (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("target_face_count", default=500000, min=1000, max=5000000, step=1000, tooltip="Target number of output faces after simplification.", optional=True),
                io.Float.Input("remesh_band", default=1.0, min=0.1, max=5.0, step=0.1, tooltip="Band width for dual-contouring. Affects surface detail capture. Higher = smoother but may lose fine details.", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, target_face_count=500000, remesh_band=1.0):
        """Apply GPU-accelerated CuMesh remeshing."""
        import torch
        import cumesh as CuMesh

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        log.info("Backend: cumesh (CUDA)")
        log.info("Input: %s vertices, %s faces", f"{initial_vertices:,}", f"{initial_faces:,}")
        log.info("Parameters: target_face_count=%s, remesh_band=%s", f"{target_face_count:,}", remesh_band)

        # Hardcoded resolution = 512 (same as TRELLIS2)
        grid_resolution = 512

        remeshed_mesh, error = cumesh_dc_remesh(
            trimesh, grid_resolution, fill_holes_first=False, band=remesh_band
        )
        if remeshed_mesh is None:
            raise ValueError(f"CuMesh remeshing failed: {error}")

        pre_simplify_faces = len(remeshed_mesh.faces)
        import comfy.model_management
        device = comfy.model_management.get_torch_device()
        assert device.type == "cuda", f"CuMesh requires CUDA but got device '{device}' — cumesh is GPU-only, no CPU fallback"
        vertices = torch.tensor(remeshed_mesh.vertices, dtype=torch.float32).to(device)
        faces = torch.tensor(remeshed_mesh.faces, dtype=torch.int32).to(device)

        cumesh_obj = CuMesh.CuMesh()
        cumesh_obj.init(vertices, faces)

        # Skip pre-simplify unify on large meshes - CuMesh crashes on >2M faces
        if len(faces) < 2_000_000:
            cumesh_obj.unify_face_orientations()
            log.info("Unified face orientations (pre-simplify)")
        else:
            log.info("Skipping pre-simplify unify (mesh too large: %s faces)", f"{len(faces):,}")

        # Simplify to target
        cumesh_obj.simplify(target_face_count, verbose=True)
        log.info("After simplify: %s faces", f"{cumesh_obj.num_faces:,}")

        # Unify after simplify (on smaller mesh, should work)
        cumesh_obj.unify_face_orientations()
        log.info("Unified face orientations (post-simplify)")

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

        log.info("Output: %d vertices (%+d), %d faces (%+d)",
                 len(remeshed_mesh.vertices), vertex_change,
                 len(remeshed_mesh.faces), face_change)

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
        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {
    "GeomPackRemesh_GPU": RemeshGPUNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemesh_GPU": "Remesh GPU (backend)",
}
