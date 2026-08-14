# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Get Mesh From Batch - extract mesh(es) from a TRIMESH batch by index or slice.

The list-handling counterpart of Load Mesh Batch: takes the batch (a list of
trimesh objects) and a selection, returns the selected mesh(es).

selection accepts Python indexing syntax:
  "0"     first mesh          "-1"    last mesh
  "1:5"   meshes 1..4         ":3"    first three
  "::2"   every second        "-3:"   last three

A single index is clamped to the batch (out-of-range picks the last); slices
follow Python semantics (self-clamping, never error).
"""

import logging

from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _parse_selection(text, n):
    """'2' -> [2] (clamped); '1:5' / '::2' / '-3:' -> slice indices. Raises
    ValueError with the offending text on anything unparseable."""
    s = str(text).strip()
    if ":" not in s:
        try:
            idx = int(s)
        except ValueError:
            raise ValueError(f"Get Mesh From Batch: invalid selection {s!r} "
                             f"(expected an index like \"2\" or a slice like \"1:5\")")
        if idx < 0:
            idx += n
        return [max(0, min(idx, n - 1))]
    parts = s.split(":")
    if len(parts) > 3:
        raise ValueError(f"Get Mesh From Batch: invalid slice {s!r} (at most start:stop:step)")
    try:
        start, stop, step = (int(p) if p.strip() else None for p in (parts + [""] * 3)[:3])
    except ValueError:
        raise ValueError(f"Get Mesh From Batch: invalid slice {s!r} (parts must be integers)")
    if step == 0:
        raise ValueError("Get Mesh From Batch: slice step cannot be 0")
    return list(range(n))[slice(start, stop, step)]


class GetMeshFromBatch(io.ComfyNode):
    """Extract mesh(es) from a TRIMESH batch by index or Python slice."""

    INPUT_IS_LIST = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackGetMeshFromBatch",
            display_name="Get Mesh From Batch",
            category="geompack/io",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.String.Input(
                    "selection", default="0",
                    tooltip='Index ("2", "-1") or Python slice ("1:5", ":3", "::2", "-3:")'),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="meshes", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, trimesh, selection="0"):
        # Inputs arrive as lists (INPUT_IS_LIST=True); the batch itself is `trimesh`.
        sel_val = selection[0] if isinstance(selection, list) else selection
        if not trimesh:
            raise ValueError("Empty mesh batch provided")

        indices = _parse_selection(sel_val, len(trimesh))
        if not indices:
            raise ValueError(f"Get Mesh From Batch: selection {sel_val!r} matches "
                             f"no meshes (batch has {len(trimesh)})")

        picked = [trimesh[i] for i in indices]
        names = [(getattr(m, "metadata", None) or {}).get("file_name", "") for m in picked]
        log.info("Get Mesh From Batch: %r -> %d of %d mesh(es)%s", sel_val,
                 len(picked), len(trimesh),
                 f" ({', '.join(n for n in names if n)})" if any(names) else "")
        return io.NodeOutput(picked)


NODE_CLASS_MAPPINGS = {
    "GeomPackGetMeshFromBatch": GetMeshFromBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackGetMeshFromBatch": "Get Mesh From Batch",
}
