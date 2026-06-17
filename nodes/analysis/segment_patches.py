# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Segment a mesh into CADable patches and OUTPUT them (mesh + per-face region_id).

Pipeline:
  1. feature edges = dihedral angle >= crease_angle_deg (optionally broadened over
     `wide_rings` so chamfers / fine fillets read as one turn, not many tiny ones).
  2. flood-fill faces into regions, cutting at those feature edges.
  3. CLEAN UP: iteratively merge each region smaller than min_patch_faces (or
     min_patch_area_pct of the surface) into the neighbour it shares the most
     boundary with. This is what turns a speckled segmentation into a usable set
     of patches.

Outputs the mesh with face_attributes['region_id'] (1..R) so you can colour it in
Preview Mesh (fields), split it with Split By Field, or fit primitives per patch.
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

from ..visualization.preview_mesh_boundaries import _edge_values, _face_regions

log = logging.getLogger("geometrypack")


def _merge_small_regions(region_id, adj_pairs, wall_mask, face_areas, min_faces, min_area):
    """Merge regions below the size thresholds into their strongest neighbour
    (most shared boundary edges, ties broken by neighbour size). Iterates to a
    fixed point. Returns cleaned, contiguously-relabelled region_id (1..R)."""
    from collections import defaultdict

    labels = np.asarray(region_id, dtype=np.int64).copy()
    adj_pairs = np.asarray(adj_pairs).reshape(-1, 2)
    wall = np.asarray(wall_mask, dtype=bool)
    walls = adj_pairs[wall] if len(adj_pairs) else np.zeros((0, 2), np.int64)
    areas = np.asarray(face_areas, dtype=np.float64)

    def relabel(lbl):
        uniq, inv = np.unique(lbl, return_inverse=True)
        return (inv + 1).astype(np.int64)

    for _ in range(10000):  # converges fast (region count strictly drops per pass)
        labels = relabel(labels)
        n = int(labels.max()) if len(labels) else 0
        fcount = np.bincount(labels, minlength=n + 1)
        acount = np.bincount(labels, weights=areas, minlength=n + 1)

        small = [r for r in range(1, n + 1)
                 if fcount[r] > 0 and (fcount[r] < min_faces or acount[r] < min_area)]
        if not small:
            break

        # region-adjacency strength from the wall edges (boundary between regions)
        rl = labels[walls[:, 0]] if len(walls) else np.zeros(0, np.int64)
        rr = labels[walls[:, 1]] if len(walls) else np.zeros(0, np.int64)
        m = rl != rr
        nbr = defaultdict(lambda: defaultdict(int))
        for a, b in zip(rl[m].tolist(), rr[m].tolist()):
            nbr[a][b] += 1
            nbr[b][a] += 1

        # union-find merge of every small region into its strongest neighbour
        parent = list(range(n + 1))

        def find(x):
            r = x
            while parent[r] != r:
                r = parent[r]
            while parent[x] != r:
                parent[x], x = r, parent[x]
            return r

        merged = False
        for r in sorted(small, key=lambda r: fcount[r]):
            if not nbr[r]:
                continue  # isolated region with no neighbour: leave it
            target = max(nbr[r].items(), key=lambda kv: (kv[1], fcount[kv[0]]))[0]
            ra, rt = find(r), find(target)
            if ra != rt:
                parent[ra] = rt
                merged = True
        if not merged:
            break
        labels = np.array([find(int(l)) for l in labels], dtype=np.int64)

    return relabel(labels)


class SegmentPatchesNode(io.ComfyNode):
    """Flood-fill a mesh into CADable patches, merge speckle, output region_id."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSegmentPatches",
            display_name="Segment Mesh into Patches",
            category="geompack/analysis",
            description=(
                "Segment a mesh into CADable surface patches and output them as a per-face "
                "region_id field (1..R). Feature edges = dihedral >= crease_angle_deg cut the "
                "mesh; faces flood-fill into patches; then small/speckle patches are merged into "
                "their strongest neighbour so you get a clean set instead of fragments.\n"
                "\n"
                "wide_rings broadens the normals first (chamfers / fine fillets become one patch "
                "border instead of many tiny ones). min_patch_faces / min_patch_area_pct control "
                "how aggressively speckle is absorbed.\n"
                "\n"
                "Color the output in Preview Mesh (fields -> region_id), split it per-patch with "
                "Split By Field, or feed it to primitive fitting."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("crease_angle_deg", default=30.0, min=0.0, max=180.0, step=1.0,
                    tooltip="Dihedral angle (deg) above which an edge cuts patches apart."),
                io.Int.Input("wide_rings", default=0, min=0, max=6, step=1,
                    tooltip="Broaden each face's normal over this many rings before measuring the "
                            "dihedral, so chamfers / fine fillets read as a single border. 0 = off, "
                            "2-4 catches wider blends (also rounds genuinely-sharp single edges)."),
                io.Int.Input("min_patch_faces", default=20, min=0, max=1000000, step=1,
                    tooltip="Merge any patch with fewer than this many faces into a neighbour. "
                            "Raise to absorb more speckle."),
                io.Float.Input("min_patch_area_pct", default=0.05, min=0.0, max=50.0, step=0.01,
                    tooltip="Also merge any patch whose area is below this percent of the total "
                            "surface area. Catches thin/sliver patches that have many faces."),
                io.Combo.Input("preclean", options=["true", "false"], default="true",
                    tooltip="Merge duplicate vertices + drop degenerate faces first (recommended; "
                            "needed for correct face adjacency)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh_with_patches"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, crease_angle_deg=30.0, wide_rings=0,
                min_patch_faces=20, min_patch_area_pct=0.05, preclean="true"):
        mesh = trimesh.copy()
        if preclean == "true":
            try:
                mesh.merge_vertices()
            except Exception as e:
                log.debug("merge_vertices skipped: %s", e)
            try:
                mesh.update_faces(mesh.nondegenerate_faces())
                mesh.remove_unreferenced_vertices()
            except Exception as e:
                log.debug("degenerate cleanup skipped: %s", e)

        n_faces = int(len(mesh.faces))
        adj_pairs = np.asarray(getattr(mesh, "face_adjacency", np.zeros((0, 2), int)))
        _, vals, used, _ = _edge_values(mesh, "face_normals", "angle", wide_dihedral=int(wide_rings))
        passing = vals >= float(crease_angle_deg)

        region_id, n_raw = _face_regions(n_faces, adj_pairs, passing)

        areas = np.asarray(mesh.area_faces, dtype=np.float64)
        total_area = float(areas.sum()) or 1.0
        min_area = (float(min_patch_area_pct) / 100.0) * total_area
        region_id = _merge_small_regions(region_id, adj_pairs, passing, areas,
                                         int(min_patch_faces), min_area)
        n_final = int(region_id.max()) if len(region_id) else 0

        mesh.face_attributes["region_id"] = region_id.astype(np.float32)
        mesh.metadata = (mesh.metadata.copy() if mesh.metadata else {})
        mesh.metadata["patches"] = {
            "crease_angle_deg": float(crease_angle_deg), "wide_rings": int(wide_rings),
            "n_patches": n_final, "n_raw": int(n_raw),
        }

        # patch-size summary (top few)
        counts = np.bincount(region_id.astype(np.int64))
        sizes = sorted(counts[1:].tolist(), reverse=True)
        biggest = ", ".join(str(s) for s in sizes[:8])
        info = (
            f"Segment Mesh into Patches\n"
            f"faces={n_faces:,}  feature edges={int(passing.sum()):,} ({used})\n"
            f"patches: {n_raw:,} raw -> {n_final:,} after merging speckle\n"
            f"  (min {int(min_patch_faces)} faces / {min_patch_area_pct:g}% area)\n"
            f"largest patches (faces): {biggest}"
        )
        log.info("[SegmentPatches] %s", info.replace("\n", " | "))
        return io.NodeOutput(mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSegmentPatches": SegmentPatchesNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSegmentPatches": "Segment Mesh into Patches"}
