# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Fast quadric mesh simplification backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _fast_simplification_decimate(mesh, target_reduction, agg):
    """Fast quadric mesh simplification."""
    import fast_simplification

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)

    v_out, f_out = fast_simplification.simplify(
        v, f,
        target_reduction=target_reduction,
        agg=agg,
    )

    return trimesh_module.Trimesh(
        vertices=v_out,
        faces=f_out,
        process=False,
    )


class DecimateFastSimplificationNode(io.ComfyNode):
    """Fast quadric mesh simplification backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackDecimate_FastSimplification",
            display_name="Decimate Fast Simplification (backend)",
            category="geompack/decimation",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("target_reduction", default=0.5, min=0.01, max=0.99, step=0.01, tooltip="Fraction of faces to REMOVE. 0.5 = reduce to ~50%% of original faces, 0.9 = reduce to ~10%% of original."),
                io.Int.Input("aggressiveness", default=7, min=1, max=15, step=1, tooltip="How aggressively to simplify. Higher = faster but lower quality. Default 7."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="decimated_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, target_reduction=0.5, aggressiveness=7):
        log.info("Backend: fast_simplification")
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)
        log.info("Input: %s vertices, %s faces", f"{initial_vertices:,}", f"{initial_faces:,}")

        decimated = _fast_simplification_decimate(trimesh, target_reduction, aggressiveness)

        if hasattr(trimesh, "metadata") and trimesh.metadata:
            decimated.metadata = trimesh.metadata.copy()
        decimated.metadata["decimation"] = {
            "algorithm": "fast_simplification",
            "original_vertices": initial_vertices,
            "original_faces": initial_faces,
            "result_vertices": len(decimated.vertices),
            "result_faces": len(decimated.faces),
        }

        face_change = len(decimated.faces) - initial_faces
        face_pct = (face_change / initial_faces) * 100 if initial_faces > 0 else 0

        info = f"""Decimate Results (fast_simplification):

Target Reduction: {target_reduction:.0%}
Aggressiveness: {aggressiveness}

Before:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After:
  Vertices: {len(decimated.vertices):,}
  Faces: {len(decimated.faces):,}
  Reduction: {abs(face_pct):.1f}%
"""
        return io.NodeOutput(decimated, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackDecimate_FastSimplification": DecimateFastSimplificationNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackDecimate_FastSimplification": "Decimate Fast Simplification (backend)"}
