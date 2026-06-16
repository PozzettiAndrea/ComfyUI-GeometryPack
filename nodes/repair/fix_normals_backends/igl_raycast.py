# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""IGL raycast fix normals backend node."""

import logging
import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FixNormalsIglRaycastNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFixNormals_IglRaycast",
            display_name="Fix Normals IGL Raycast (backend)",
            category="geompack/repair",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("rays_minimum", default=10, min=1, max=1000, step=1, tooltip="Minimum rays cast per face for the inside/outside vote. More rays = more robust on noisy / self-intersecting meshes, slower."),
                io.Combo.Input("use_parity", options=["true", "false"], default="false", tooltip="Decide orientation by ray-hit parity (odd/even) instead of front/back hit counts. Parity suits watertight meshes; front/back voting is more robust on open meshes."),
                io.Combo.Input("facet_wise", options=["false", "true"], default="false", tooltip="Orient each facet independently instead of per connected component. Usually leave off (per-component is more coherent and avoids salt-and-pepper flips)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="fixed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, rays_minimum=10, use_parity="false", facet_wise="false"):
        import igl
        import igl.embree

        fixed_mesh = trimesh.copy()

        was_consistent = fixed_mesh.is_winding_consistent

        V = np.ascontiguousarray(fixed_mesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(fixed_mesh.faces, dtype=np.int64)

        # Batched, per-component raycast reorientation (Takayama et al. 2014) via
        # Embree. Replaces the old per-face Python ray loop (one igl.ray_mesh_intersect
        # call per face + an unscaled 1e-6 epsilon): a single call casts all rays over
        # the BVH and orients each connected component coherently.
        # Returns I = per-face flip flags, C = connected-component ids.
        I, C = igl.embree.reorient_facets_raycast(
            V, F,
            -1,                      # rays_total: -1 => auto (scaled by face area)
            int(rays_minimum),       # rays_minimum per face
            facet_wise == "true",    # per-facet vs per-component
            use_parity == "true",    # parity vs front/back voting
            False,                   # is_verbose
        )

        flip_mask = np.asarray(I, dtype=bool)

        # Flip faces by reversing vertex order
        F_out = F.copy()
        F_out[flip_mask] = F_out[flip_mask][:, [0, 2, 1]]

        num_flipped = int(np.sum(flip_mask))
        num_components = (int(np.asarray(C).max()) + 1) if len(np.asarray(C)) else 0
        fixed_mesh.faces = F_out

        log.info("igl_raycast: flipped %d/%d faces", num_flipped, len(F))

        is_consistent = fixed_mesh.is_winding_consistent

        info = (
            f"Normal Orientation Fix:\n"
            f"\n"
            f"Method: igl_raycast (embree, {'parity' if use_parity == 'true' else 'front/back'}, rays_min={rays_minimum})\n"
            f"Before: {'Consistent' if was_consistent else 'Inconsistent'}\n"
            f"After:  {'Consistent' if is_consistent else 'Inconsistent'}\n"
            f"Faces Flipped: {num_flipped}\n"
            f"Components: {num_components}\n"
            f"\n"
            f"Vertices: {len(fixed_mesh.vertices):,}\n"
            f"Faces: {len(fixed_mesh.faces):,}\n"
            f"\n"
            f"Note: Batched Embree raycast (Takayama 2014), oriented per connected component\n"
            f"{'[OK] Normals are now consistently oriented!' if is_consistent else '[WARN] Some inconsistencies may remain (check mesh topology)'}"
        )

        log.info("Normal orientation: %s -> %s", was_consistent, is_consistent)

        return io.NodeOutput(fixed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackFixNormals_IglRaycast": FixNormalsIglRaycastNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFixNormals_IglRaycast": "Fix Normals IGL Raycast (backend)"}
