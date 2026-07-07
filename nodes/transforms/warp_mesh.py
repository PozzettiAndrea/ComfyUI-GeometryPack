# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Warp Mesh - displace vertices by a 3-component per-vertex field.

Picks a vertex field (e.g. ``displacement``, ``displacement_pred``,
``displacement_error``, ``normals``) stored in ``mesh.vertex_attributes`` whose
shape is ``(n_vertices, 3)`` and adds ``scale * field`` to every vertex.

Run the node standalone (it is an output node) to list the candidate 3-D fields:
the companion JS extension shows them in a clickable box under the node, and
clicking one fills the ``field_name`` widget. Re-run to apply the warp.
"""

import logging

import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _candidate_vector_fields(mesh):
    """Names of per-vertex fields shaped (n_vertices, 3)."""
    n = len(mesh.vertices)
    out = []
    attrs = getattr(mesh, "vertex_attributes", None) or {}
    for name, value in attrs.items():
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] == 3:
            out.append(str(name))
    return sorted(out)


class WarpMeshNode(io.ComfyNode):
    """Displace mesh vertices by a 3-D per-vertex field (vertices += scale * field)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackWarpMesh",
            display_name="Warp Mesh",
            category="geompack/transforms",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.String.Input(
                    "field_name", default="", multiline=False, optional=True,
                    tooltip="Name of a (n_vertices, 3) vertex field to displace by. "
                            "Leave empty and run to just list candidate fields.",
                ),
                io.Float.Input(
                    "scale", default=1.0, min=-1000.0, max=1000.0, step=0.01, optional=True,
                    tooltip="Multiplier applied to the field before adding to the vertices.",
                ),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="warped_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, field_name="", scale=1.0):
        mesh = trimesh
        candidates = _candidate_vector_fields(mesh)
        field_name = (field_name or "").strip()

        result = mesh.copy()
        warped = False

        if not field_name:
            info = ("No field selected. Candidate 3-D vertex fields:\n  "
                    + ("\n  ".join(candidates) if candidates else "(none)")
                    + "\n\nClick one in the box below (or type it into 'field_name') and re-run.")
        elif field_name not in candidates:
            info = (f"Field '{field_name}' is not a (n_vertices, 3) vertex field.\n"
                    f"Candidates:\n  " + ("\n  ".join(candidates) if candidates else "(none)"))
        else:
            disp = np.asarray(mesh.vertex_attributes[field_name], dtype=np.float64)
            result.vertices = np.asarray(mesh.vertices, dtype=np.float64) + float(scale) * disp
            warped = True
            mag = np.linalg.norm(disp, axis=1)
            info = (f"Warped by '{field_name}' (scale {scale:g}).\n"
                    f"  vertices: {len(result.vertices)}\n"
                    f"  |field| mean/max: {float(mag.mean()):.4g} / {float(mag.max()):.4g}\n"
                    f"  new bounds min: [{result.bounds[0][0]:.3f}, {result.bounds[0][1]:.3f}, {result.bounds[0][2]:.3f}]\n"
                    f"  new bounds max: [{result.bounds[1][0]:.3f}, {result.bounds[1][1]:.3f}, {result.bounds[1][2]:.3f}]")

        log.info("Warp Mesh: field=%r scale=%s warped=%s candidates=%s",
                 field_name, scale, warped, candidates)

        return io.NodeOutput(
            result, info,
            ui={"fields": [candidates], "selected": [field_name], "warped": [warped]},
        )


NODE_CLASS_MAPPINGS = {
    "GeomPackWarpMesh": WarpMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackWarpMesh": "Warp Mesh",
}
