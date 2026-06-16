# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""FaithC (Faithful Contouring) remeshing backend node.

FaithC (Luo et al., CVPR 2026) is a near-lossless voxel-based remesher: it builds
a hierarchical octree (resolution = 2^max_level) around the surface and extracts a
feature-preserving contour mesh from per-voxel anchors + edge flux -- a more
faithful alternative to SDF + Marching Cubes / dual contouring. GPU/CUDA only.

Uses the class API (atom3d MeshBVH + OctreeIndexer -> FCTEncoder -> FCTDecoder);
imports are deferred to execute() so the comfy-env metadata scan isn't pulled into
the heavy CUDA import.
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshFaithCNode(io.ComfyNode):
    """FaithC faithful-contouring remeshing backend (CUDA)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_FaithC",
            display_name="Remesh FaithC (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("faithc_max_level", default=7, min=4, max=10, step=1, tooltip=(
                    "Octree depth = THE grid resolution knob. Grid resolution = 2^level: "
                    "6=64, 7=128, 8=256, 9=512, 10=1024 cells per axis. FaithC is power-of-2 "
                    "only, so this is the one resolution control (there is no separate grid "
                    "size). Higher = finer features + many more faces + more VRAM/time "
                    "(res 128 ~ 140k faces, 256 ~ 560k on a typical part).")),
                io.Int.Input("faithc_min_level", default=4, min=1, max=7, step=1, tooltip=(
                    "Coarsest octree level where traversal starts (the top of the hierarchy "
                    "that gets fully populated before descending). Must be <= max_level. "
                    "Default 4; lower can speed up sparse meshes, rarely needs changing.")),
                io.Float.Input("faithc_lambda_n", default=1.0, min=0.0, max=10.0, step=0.1, tooltip=(
                    "QEF solver NORMAL-alignment weight. Higher = each voxel anchor snaps harder "
                    "onto the surface's tangent planes -> sharper, more faithful features/edges. "
                    "Lower = softer/rounder. Default 1.0.")),
                io.Float.Input("faithc_lambda_d", default=0.1, min=0.0, max=10.0, step=0.05, tooltip=(
                    "QEF solver DISTANCE/regularization weight. Higher = anchors pulled toward "
                    "the voxel center -> more regular, less spiky, but blunts sharp detail. "
                    "Lower = follows the surface more freely. Default 0.1 (normals dominate).")),
                io.Float.Input("faithc_weight_power", default=1.0, min=0.1, max=4.0, step=0.1, tooltip=(
                    "Exponent applied to the per-constraint QEF weights. >1 emphasizes the "
                    "strongest (most confident) surface samples; 1.0 = linear. Advanced -- "
                    "leave at 1.0 unless tuning.")),
                io.Combo.Input("faithc_clamp_anchors", options=["false", "true"], default="false", tooltip=(
                    "Clamp each anchor to its voxel bounds and re-project it onto the input "
                    "surface. Prevents stray/overshooting anchors (spikes) on noisy or thin "
                    "geometry at the cost of a little sharpness. Turn on if you see spikes.")),
                io.Combo.Input("faithc_triangulation", options=[
                    "auto", "length", "angle", "normal_abs", "normal", "simple_02", "simple_13"
                ], default="auto", tooltip=(
                    "How each extracted quad is split into 2 triangles. auto = normal_abs if "
                    "normals exist (recommended). length = shorter diagonal. angle = most-"
                    "equilateral split (best triangle quality). normal/normal_abs = align split "
                    "to the surface normal (sharpest features). simple_02/13 = fixed diagonal.")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, faithc_max_level=7, faithc_min_level=4,
                faithc_lambda_n=1.0, faithc_lambda_d=0.1, faithc_weight_power=1.0,
                faithc_clamp_anchors="false", faithc_triangulation="auto"):
        import torch
        from atom3d import MeshBVH
        from atom3d.grid import OctreeIndexer
        from faithcontour.encoder import FCTEncoder
        from faithcontour.decoder import FCTDecoder

        if not torch.cuda.is_available():
            raise RuntimeError("FaithC requires CUDA (GPU) -- no CPU fallback.")

        V0 = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F0 = np.ascontiguousarray(trimesh.faces, dtype=np.int64)

        # FaithC works in a normalized grid; normalize to ~[-1,1]^3 and keep the
        # transform so we can map the result back to world coordinates.
        bmin, bmax = V0.min(0), V0.max(0)
        center = (bmin + bmax) / 2.0
        extent = float(np.max(bmax - bmin))
        if extent <= 0:
            raise ValueError("Degenerate mesh (zero bounding-box extent).")
        scale = 2.0 / extent * 0.99

        L = int(faithc_max_level)
        log.info("Backend: faithc | max_level=%d (res %d) min_level=%d", L, 2 ** L, faithc_min_level)
        log.info("Input: %d vertices, %d faces", len(V0), len(F0))

        V = torch.tensor((V0 - center) * scale, dtype=torch.float32, device="cuda")
        F = torch.tensor(F0, dtype=torch.int64, device="cuda")

        bvh = MeshBVH(V, F)
        octree = OctreeIndexer(max_level=L, bounds=bvh.get_bounds())
        result = FCTEncoder(bvh, octree).encode(
            min_level=int(faithc_min_level),
            solver_weights={
                "lambda_n": float(faithc_lambda_n), "lambda_d": float(faithc_lambda_d),
                "weight_power": float(faithc_weight_power), "eps": 1e-9,
            },
            clamp_anchors=(faithc_clamp_anchors == "true"),
        )
        # decode() directly (not decode_from_result) so triangulation_mode is honored
        decoded = FCTDecoder(resolution=2 ** L, bounds=bvh.get_bounds()).decode(
            active_voxel_indices=result.active_voxel_indices,
            anchors=result.anchor,
            edge_flux_sign=result.edge_flux_sign,
            normals=result.normal,
            triangulation_mode=faithc_triangulation,
        )

        Vo = decoded.vertices.detach().cpu().numpy().astype(np.float64)
        Fo = decoded.faces.detach().cpu().numpy().astype(np.int64)
        del bvh, octree, result, decoded, V, F
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        if len(Fo) == 0:
            raise ValueError(
                "FaithC produced 0 faces. The input is likely perfectly axis-aligned / flat at "
                "this resolution (a degenerate case for edge-flux contouring). Try a higher "
                "faithc_max_level.")

        Vw = Vo / scale + center   # de-normalize back to world coordinates
        remeshed = trimesh_module.Trimesh(vertices=Vw, faces=Fo, process=False)
        remeshed.metadata = trimesh.metadata.copy()
        remeshed.metadata["remeshing"] = {
            "algorithm": "faithc", "max_level": L, "resolution": 2 ** L,
            "min_level": int(faithc_min_level),
            "lambda_n": float(faithc_lambda_n), "lambda_d": float(faithc_lambda_d),
            "weight_power": float(faithc_weight_power),
            "clamp_anchors": faithc_clamp_anchors == "true",
            "triangulation": faithc_triangulation,
        }

        log.info("Output: %d vertices, %d faces", len(Vw), len(Fo))
        info = (f"Remesh (FaithC, res {2 ** L}): "
                f"{len(V0):,}v/{len(F0):,}f -> {len(Vw):,}v/{len(Fo):,}f | "
                f"max_level={L}, min_level={faithc_min_level}")
        return io.NodeOutput(remeshed, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_FaithC": RemeshFaithCNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_FaithC": "Remesh FaithC (backend)"}
