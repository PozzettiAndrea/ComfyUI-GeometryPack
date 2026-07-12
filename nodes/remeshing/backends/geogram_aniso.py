# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Geogram curvature-adapted CVT remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshGeogramAnisoNode(io.ComfyNode):
    """Geogram curvature-adapted CVT remeshing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_GeogramAniso",
            display_name="Remesh Geogram Anisotropic (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("nb_points_aniso", default=5000, min=0, max=1000000, step=100, tooltip=(
                    "Target number of OUTPUT vertices. Same full resampling/retopology semantics "
                    "as the Geogram Smooth backend's nb_points -- the input's own vertex "
                    "positions are discarded and an entirely new point set is placed via "
                    "anisotropic CVT (see the anisotropy parameter below for what that means). "
                    "0 = keep the same vertex count as the input.")),
                io.Float.Input("anisotropy", default=0.04, min=0.005, max=0.5, step=0.005, display_mode="number", tooltip=(
                    "How strongly the remesher favors ELONGATING triangles along low-curvature "
                    "directions instead of keeping them uniform/isotropic. Practically: on a flat "
                    "or gently-curved region, you need far fewer long, thin triangles aligned with "
                    "the surface's flow to represent it accurately than you'd need uniform "
                    "triangles -- so anisotropic remeshing can hit the same visual fidelity with "
                    "meaningfully fewer total triangles, concentrated instead on genuinely curved/"
                    "detailed regions.\n\n"
                    "COUNTERINTUITIVE DIRECTION, worth calling out explicitly: LOWER values mean "
                    "MORE anisotropic (longer, thinner, more elongated triangles) -- this is the "
                    "opposite of what 'lower number = less effect' intuition suggests. Values near "
                    "the top of the exposed range (toward 0.5) approach plain ISOTROPIC/uniform "
                    "remeshing, similar to what the Geogram Smooth backend produces. Going below "
                    "roughly 0.01 produces very elongated, stringy-looking triangles, especially "
                    "noticeable on regions with genuinely high curvature in multiple directions "
                    "(where there's no single dominant 'flow' direction for triangles to align "
                    "with) -- if the output looks stringy or needle-like, raise this back up. The "
                    "existing typical working range is 0.02-0.1: start around 0.04 (the default) "
                    "and move toward 0.02 for more aggressive triangle-count savings on mostly-flat "
                    "meshes, or toward 0.1+ if 0.04 already looks too elongated on your mesh.")),
                io.Combo.Input("adjust", options=["true", "false"], default="true", tooltip=(
                    "Whether to snap the optimized point positions back onto the ORIGINAL input "
                    "surface after the anisotropic CVT optimization settles. Same purpose as the "
                    "Geogram Smooth backend's adjust: without this, output can drift slightly off "
                    "the original geometry (rounded sharp features, lost surface detail). "
                    "DECISION: leave ON (true) essentially always, for the same reason as the "
                    "Smooth backend -- this is what makes the result an accurate remesh of your "
                    "input rather than merely a well-distributed approximation of it.")),
                io.Float.Input("adjust_max_edge_distance", default=0.5, min=0.01, max=10.0, step=0.01, display_mode="number", tooltip=(
                    "How far the adjust step is allowed to search when snapping a point back onto "
                    "the original surface, as a fraction of local edge length. Same tradeoff as "
                    "the Geogram Smooth backend: too small leaves optimization drift uncorrected; "
                    "too large risks snapping to the wrong, distant part of the surface on thin or "
                    "closely-spaced features. This matters MORE here than in the isotropic Smooth "
                    "backend, since anisotropic remeshing deliberately creates long thin triangles "
                    "whose 'local edge length' varies a lot by direction -- if you see snap-to-"
                    "wrong-surface artifacts specifically on thin, elongated features, lower this "
                    "before lowering the anisotropy setting above.")),
                io.Float.Input("adjust_border_importance", default=2.0, min=0.1, max=20.0, step=0.1, display_mode="number", tooltip=(
                    "Extra weighting on boundary/border edges during the adjust step, so open mesh "
                    "boundaries are preferentially preserved rather than drifting/shrinking inward "
                    "during optimization. Same semantics as the Geogram Smooth backend; has no "
                    "effect on a fully closed/watertight mesh.")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, nb_points_aniso=5000, anisotropy=0.04,
                adjust="true", adjust_max_edge_distance=0.5, adjust_border_importance=2.0):
        import pygeogram

        log.info("Backend: geogram_anisotropic")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: nb_points=%s, anisotropy=%s, adjust=%s, "
                 "adjust_max_edge_distance=%s, adjust_border_importance=%s",
                 f"{nb_points_aniso:,}", anisotropy, adjust,
                 adjust_max_edge_distance, adjust_border_importance)

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        V_out, F_out = pygeogram.remesh_anisotropic(
            V, F, nb_points_aniso, anisotropy=anisotropy,
            adjust=(adjust == "true"),
            adjust_max_edge_distance=float(adjust_max_edge_distance),
            adjust_border_importance=float(adjust_border_importance),
        )

        remeshed_mesh = trimesh_module.Trimesh(vertices=V_out, faces=F_out, process=False)
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'geogram_anisotropic',
            'nb_points': nb_points_aniso,
            'anisotropy': anisotropy,
            'adjust': adjust,
            'adjust_max_edge_distance': adjust_max_edge_distance,
            'adjust_border_importance': adjust_border_importance,
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (Geogram CVT Anisotropic): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"nb_points={nb_points_aniso:,}, anisotropy={anisotropy}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_GeogramAniso": RemeshGeogramAnisoNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_GeogramAniso": "Remesh Geogram Anisotropic (backend)"}
