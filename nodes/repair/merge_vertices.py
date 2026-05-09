# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Merge duplicate vertices in mesh with configurable tolerance.
Useful for fixing disconnected mesh components from CAD meshing or other sources.
"""

import logging

import trimesh
import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class MergeVerticesNode(io.ComfyNode):
    """
    Merge duplicate vertices in a mesh with configurable tolerance.

    Vertices within the specified distance tolerance are merged into a single vertex.
    This is useful for:
    - Fixing disconnected mesh components from CAD meshing (OCC)
    - Repairing meshes with near-duplicate vertices from floating-point precision issues
    - Reducing vertex count for meshes with redundant vertices

    The tolerance is specified in world units. For example:
    - 1e-8: Very tight (nearly exact matches only)
    - 1e-5: Standard for CAD meshes (recommended)
    - 1e-3: Loose (for coarse meshes)
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackMergeVertices",
            display_name="Merge Vertices",
            category="geompack/repair",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.Float.Input("tolerance", default=1e-5, min=1e-8, max=1e-2, step=1e-6, tooltip="Distance tolerance for merging vertices (1e-5 recommended for CAD meshes)"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="merged_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, tolerance=1e-5):
        """
        Merge duplicate vertices within tolerance.

        Args:
            mesh: Input trimesh.Trimesh object
            tolerance: Distance tolerance for merging vertices

        Returns:
            tuple: (merged_trimesh, info_string)
        """
        log.info("Input: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
        log.info("Tolerance: %.2e", tolerance)

        # Check initial state
        verts_before = len(mesh.vertices)
        faces_before = len(mesh.faces)

        # Count connected components before
        try:
            components_before = len(mesh.split(only_watertight=False))
        except Exception:
            components_before = -1  # Unknown

        # Create a copy to avoid modifying original
        merged_mesh = mesh.copy()

        # Convert tolerance to digits for trimesh: 1e-5 -> 5 digits
        digits = max(0, -int(np.floor(np.log10(tolerance))))
        log.debug("Using %d decimal places for vertex matching", digits)

        # Merge vertices with specified precision
        merged_mesh.merge_vertices(digits_vertex=digits)

        # Check result
        verts_after = len(merged_mesh.vertices)
        faces_after = len(merged_mesh.faces)

        # Count connected components after
        try:
            components_after = len(merged_mesh.split(only_watertight=False))
        except Exception:
            components_after = -1  # Unknown

        # Calculate changes
        verts_removed = verts_before - verts_after
        faces_removed = faces_before - faces_after
        components_change = components_before - components_after if components_before >= 0 and components_after >= 0 else None

        # Build info string
        components_before_str = str(components_before) if components_before >= 0 else "?"
        components_after_str = str(components_after) if components_after >= 0 else "?"
        components_change_str = f" ({-components_change:+d})" if components_change is not None else ""

        info = f"""Vertex Merge Results:

Tolerance: {tolerance:.2e} ({digits} decimal places)

Before:
  Vertices: {verts_before:,}
  Faces: {faces_before:,}
  Components: {components_before_str}

After:
  Vertices: {verts_after:,} ({-verts_removed:+,})
  Faces: {faces_after:,} ({-faces_removed:+,})
  Components: {components_after_str}{components_change_str}

{'[OK] Vertices merged successfully!' if verts_removed > 0 else '[INFO] No duplicate vertices found within tolerance.'}
{'[OK] Mesh is now fully connected!' if components_after == 1 else ''}
{'[WARN] Mesh still has multiple disconnected components.' if components_after is not None and components_after > 1 else ''}
"""

        log.info("Removed %d duplicate vertices", verts_removed)
        if components_change is not None:
            log.info("Components: %d -> %d", components_before, components_after)

        return io.NodeOutput(merged_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {
    "GeomPackMergeVertices": MergeVerticesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackMergeVertices": "Merge Vertices",
}
