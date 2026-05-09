# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Normalize Mesh to Bounding Box - Center and scale mesh isotropically.

Normalizes mesh/pointcloud to fit within a target bounding box by:
1. Centering to origin
2. Scaling isotropically (uniform scale) to fit target size

Useful for:
- Preparing meshes for TripoSF (target_size=1.0 -> [-0.5, 0.5] box)
- Standardizing mesh sizes for processing
- Ensuring normals are estimated in the correct coordinate space
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class NormalizeMeshToBBox(io.ComfyNode):
    """
    Normalize mesh/pointcloud to bounding box.

    Centers mesh at origin and scales isotropically to fit target size.
    Stores normalization parameters in metadata for potential denormalization.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackNormalizeMeshToBBox",
            display_name="Normalize to BBox",
            category="geompack/transforms",
            description='Center and scale mesh isotropically to fit target bounding box. Use before estimating normals for TripoSF.',
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh", tooltip="Input mesh or pointcloud to normalize"),
                io.Float.Input("target_size", default=1.0, min=0.001, max=100.0, step=0.1, tooltip="Target bounding box size. 1.0 = [-0.5, 0.5] box (default for TripoSF)", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="normalized_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, target_size=1.0):
        """
        Normalize mesh to target bounding box.

        Args:
            trimesh: Input mesh or pointcloud
            target_size: Target bounding box size (1.0 = [-0.5, 0.5])

        Returns:
            Tuple of (normalized_mesh, info_string)
        """
        log.info("Input: %d vertices", len(trimesh.vertices))
        log.info("Target size: %s (bbox: [%.2f, %.2f])", target_size, -target_size / 2, target_size / 2)

        # Get input bounds
        input_bounds = trimesh.bounds
        input_extents = trimesh.extents

        # Handle case where bounds/extents are None
        if input_bounds is None or input_extents is None:
            vertices_arr = np.asarray(trimesh.vertices)
            if len(vertices_arr) > 0:
                input_bounds = np.array([vertices_arr.min(axis=0), vertices_arr.max(axis=0)])
                input_extents = input_bounds[1] - input_bounds[0]
            else:
                raise ValueError("Cannot normalize empty mesh")

        center = (input_bounds[0] + input_bounds[1]) / 2
        max_extent = max(input_extents)

        log.info("Original center: [%.3f, %.3f, %.3f]", center[0], center[1], center[2])
        log.info("Original extents: [%.3f, %.3f, %.3f]", input_extents[0], input_extents[1], input_extents[2])
        log.info("Max extent: %.3f", max_extent)

        # Copy to avoid modifying original
        result = trimesh.copy()

        # Center to origin
        result.apply_translation(-center)

        # Scale to target size
        scale_factor = target_size / max_extent
        result.apply_scale(scale_factor)

        log.info("Scale factor: %.6f", scale_factor)
        log.info("[OK] Normalized to [%.2f, %.2f] bbox", -target_size / 2, target_size / 2)

        # Preserve existing metadata
        if hasattr(trimesh, 'metadata'):
            result.metadata = trimesh.metadata.copy()
        else:
            result.metadata = {}

        # Store normalization parameters for potential denormalization
        result.metadata['normalized'] = True
        result.metadata['normalization'] = {
            'original_center': center.tolist(),
            'scale_factor': float(scale_factor),
            'original_extents': input_extents.tolist(),
            'target_size': float(target_size),
        }

        # Create info string
        info = f"""Normalization Results:

Original Bounds:
  Min: [{input_bounds[0][0]:.3f}, {input_bounds[0][1]:.3f}, {input_bounds[0][2]:.3f}]
  Max: [{input_bounds[1][0]:.3f}, {input_bounds[1][1]:.3f}, {input_bounds[1][2]:.3f}]
  Extents: [{input_extents[0]:.3f}, {input_extents[1]:.3f}, {input_extents[2]:.3f}]

Normalization:
  Center translation: [{-center[0]:.3f}, {-center[1]:.3f}, {-center[2]:.3f}]
  Scale factor: {scale_factor:.6f}

New Bounds:
  Target size: {target_size} -> [{-target_size/2:.2f}, {target_size/2:.2f}] bbox
  Actual extents: [{result.extents[0]:.3f}, {result.extents[1]:.3f}, {result.extents[2]:.3f}]

Note: Use this BEFORE AddNormalsToPointCloud for TripoSF workflows.
"""

        return io.NodeOutput(result, info, ui={"text": [info]})


# Node registration
NODE_CLASS_MAPPINGS = {
    "GeomPackNormalizeMeshToBBox": NormalizeMeshToBBox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackNormalizeMeshToBBox": "Normalize to BBox",
}
