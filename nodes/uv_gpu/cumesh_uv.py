# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""CuMesh GPU-accelerated UV unwrapping backend node."""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class UVCuMeshNode(io.ComfyNode):
    """CuMesh GPU-accelerated UV unwrapping backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackUV_CuMesh",
            display_name="UV CuMesh (backend)",
            category="geompack/uv",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("chart_cone_angle", default=90.0, min=0.0, max=359.9, step=1.0),
                io.Int.Input("chart_refine_iterations", default=0, min=0, max=10, step=1),
                io.Int.Input("chart_global_iterations", default=1, min=0, max=10, step=1),
                io.Int.Input("chart_smooth_strength", default=1, min=0, max=10, step=1),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="unwrapped_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, chart_cone_angle=90.0, chart_refine_iterations=0,
                chart_global_iterations=1, chart_smooth_strength=1):
        """CuMesh GPU-accelerated UV unwrapping with fast clustering + xatlas."""
        try:
            import torch
            import cumesh as CuMesh
        except ImportError as e:
            raise ImportError(
                f"cumesh not installed or CUDA not available: {e}\n"
                "CuMesh requires CUDA and PyTorch. Install with:\n"
                "  pip install cumesh (or run install.py)\n"
                "Alternatively, use 'xatlas' method which works on CPU."
            )

        import comfy.model_management
        device = comfy.model_management.get_torch_device()
        assert device.type == "cuda", f"CuMesh requires CUDA but got device '{device}' — cumesh is GPU-only, no CPU fallback"

        log.info("Backend: cumesh")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("CuMesh: chart_cone_angle=%s, refine_iter=%s, global_iter=%s, smooth=%s",
                 chart_cone_angle, chart_refine_iterations, chart_global_iterations, chart_smooth_strength)

        # Convert to torch tensors on GPU
        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32).to(device)
        faces = torch.tensor(trimesh.faces, dtype=torch.int32).to(device)

        # Convert cone angle to radians
        chart_cone_angle_rad = np.radians(chart_cone_angle)

        # Initialize CuMesh
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)

        # UV Unwrap with two-stage process (fast clustering + xatlas)
        log.info("Running CuMesh UV unwrap...")
        out_vertices, out_faces, out_uvs, out_vmaps = cumesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": chart_cone_angle_rad,
                "refine_iterations": chart_refine_iterations,
                "global_iterations": chart_global_iterations,
                "smooth_strength": chart_smooth_strength,
            },
            return_vmaps=True,
            verbose=True,
        )

        # Convert back to numpy
        out_vertices_np = out_vertices.cpu().numpy()
        out_faces_np = out_faces.cpu().numpy()
        out_uvs_np = out_uvs.cpu().numpy()

        # Flip V coordinate (cumesh uses different UV convention)
        out_uvs_np[:, 1] = 1 - out_uvs_np[:, 1]

        # Build result trimesh
        unwrapped = trimesh_module.Trimesh(
            vertices=out_vertices_np,
            faces=out_faces_np,
            process=False
        )

        from trimesh.visual import TextureVisuals
        unwrapped.visual = TextureVisuals(uv=out_uvs_np)

        # Preserve metadata
        unwrapped.metadata = trimesh.metadata.copy()
        unwrapped.metadata['uv_unwrap'] = {
            'algorithm': 'cumesh',
            'chart_cone_angle': chart_cone_angle,
            'chart_refine_iterations': chart_refine_iterations,
            'chart_global_iterations': chart_global_iterations,
            'chart_smooth_strength': chart_smooth_strength,
            'original_vertices': len(trimesh.vertices),
            'unwrapped_vertices': len(out_vertices_np),
            'vertex_duplication_ratio': len(out_vertices_np) / len(trimesh.vertices)
        }

        # Clean up GPU memory
        comfy.model_management.soft_empty_cache()

        log.info("Output: %d vertices, %d faces", len(unwrapped.vertices), len(unwrapped.faces))

        info = f"""UV Unwrap Results (CuMesh):

Algorithm: CuMesh GPU-accelerated (fast clustering + xatlas)
Optimized for: Large meshes, GPU acceleration

Parameters:
  Chart Cone Angle: {chart_cone_angle}°
  Refine Iterations: {chart_refine_iterations}
  Global Iterations: {chart_global_iterations}
  Smooth Strength: {chart_smooth_strength}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(out_vertices_np):,}
  Faces: {len(unwrapped.faces):,}
  Vertex Duplication: {len(out_vertices_np)/len(trimesh.vertices):.2f}x

Two-stage GPU-accelerated UV unwrapping with vertex splitting at seams.
"""
        return io.NodeOutput(unwrapped, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackUV_CuMesh": UVCuMeshNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackUV_CuMesh": "UV CuMesh (backend)"}
