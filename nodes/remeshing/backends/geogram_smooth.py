# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Geogram CVT isotropic remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshGeogramSmoothNode(io.ComfyNode):
    """Geogram CVT isotropic remeshing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_GeogramSmooth",
            display_name="Remesh Geogram Smooth (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("nb_points", default=5000, min=0, max=1000000, step=100, tooltip=(
                    "Target number of OUTPUT vertices. This is a full resampling/retopology "
                    "operation, not a refinement of the existing mesh -- the input's own vertex "
                    "positions are discarded entirely and an entirely new, evenly-distributed set "
                    "of nb_points points is placed on the surface via the CVT (Centroidal Voronoi "
                    "Tessellation) algorithm below, then triangulated. 0 = keep the same vertex "
                    "count as the input (still a full resample, just targeting the same density).")),
                io.Int.Input("nb_lloyd_iter", default=5, min=1, max=50, step=1, tooltip=(
                    "Lloyd relaxation iterations -- the COARSE first phase of CVT remeshing. The "
                    "nb_points points start out essentially randomly placed on the surface; each "
                    "Lloyd iteration moves every point to the centroid of its own Voronoi cell "
                    "(the region of surface closest to it), which spreads clumped points apart and "
                    "pulls sparse regions together. This is the classic, simple, but SLOWLY "
                    "converging way to get an even distribution -- most of the improvement happens "
                    "in the first few iterations, with rapidly diminishing returns after that "
                    "(Lloyd's algorithm has a slow LINEAR convergence rate, which is exactly why "
                    "the faster Newton phase below exists as a second stage). "
                    "HOW TO PICK A VALUE: the default, 5, is enough to get past the worst of the "
                    "initial random clumping for most meshes. Going much beyond ~10-15 rarely "
                    "improves the result further -- if the output still looks unevenly distributed "
                    "after a run, raising nb_newton_iter (below) is almost always the more effective "
                    "fix, not raising this.")),
                io.Int.Input("nb_newton_iter", default=30, min=1, max=100, step=1, tooltip=(
                    "Newton-CVT refinement iterations -- the FAST second phase, run after Lloyd "
                    "relaxation above. Where Lloyd just averages toward each point's local Voronoi "
                    "centroid, Newton's method uses the actual gradient AND (approximate) curvature "
                    "of the CVT energy to take much more effective steps toward a truly optimal "
                    "even distribution -- this is where the actual remeshing QUALITY mostly comes "
                    "from, not the Lloyd phase. "
                    "HOW TO PICK A VALUE: a handful of iterations (5-10) gets rough uniformity; the "
                    "default, 30, is where this typically fully converges for most meshes; going "
                    "into the hundreds rarely buys anything further since the optimization is "
                    "already essentially converged well before then -- treat 30-50 as the practical "
                    "ceiling worth trying before concluding the mesh needs a different approach "
                    "(e.g. more nb_points, or checking the input isn't itself pathological). Cost "
                    "scales roughly proportionally with this value.")),
                io.Int.Input("newton_m", default=7, min=1, max=20, step=1, tooltip=(
                    "L-BFGS memory size for the Newton phase above -- how many of the OPTIMIZER's "
                    "own previous gradient/step pairs it keeps around to approximate the energy's "
                    "curvature (Hessian) for each Newton step. This is an internal optimizer tuning "
                    "knob, NOT a 'more = better convergence' parameter like nb_newton_iter -- it "
                    "controls how accurately curvature gets approximated per step, not how many "
                    "steps are taken. "
                    "HOW TO PICK A VALUE: the default, 7, is a standard, well-tested L-BFGS memory "
                    "size (this exact value is the common default across most L-BFGS "
                    "implementations in scientific computing, not specific to this tool) and rarely "
                    "needs changing. Small values (1-3) approximate curvature more crudely and may "
                    "need more nb_newton_iter to compensate for slower per-step progress; raising "
                    "this past ~10-15 mostly just costs more memory/time per iteration for very "
                    "little additional accuracy. If you're tempted to tune this, tune "
                    "nb_newton_iter instead first.")),
                io.Combo.Input("adjust", options=["true", "false"], default="true", tooltip=(
                    "Whether to snap the optimized CVT point positions back onto the ORIGINAL input "
                    "surface after the Lloyd + Newton phases above settle. The CVT optimization "
                    "itself operates somewhat loosely relative to the exact input geometry (it's "
                    "solving for an even DISTRIBUTION of points, not for staying glued to the "
                    "surface) -- without this snap-back step, the output can drift slightly off the "
                    "original shape: sharp features get rounded off and fine surface detail can be "
                    "lost. "
                    "DECISION: leave ON (true) essentially always -- this is what makes the output "
                    "an accurate remesh of your input rather than a merely evenly-distributed but "
                    "drifted approximation of it. The only reason to turn this off is if you "
                    "specifically want the raw relaxed/smoothed point positions rather than a "
                    "surface-accurate result (e.g. deliberately using CVT relaxation itself as a "
                    "smoothing operation).")),
                io.Float.Input("adjust_max_edge_distance", default=0.5, min=0.01, max=10.0, step=0.01, display_mode="number", tooltip=(
                    "How far the adjust step (above) is allowed to search when snapping a point "
                    "back onto the original surface, expressed as a fraction of local edge length "
                    "(not an absolute distance -- it scales with the mesh's own local triangle "
                    "size). "
                    "HOW TO PICK A VALUE: too SMALL and points that drifted far during CVT "
                    "optimization won't get fully corrected, so some detail loss from the "
                    "optimization phase persists into the final output. Too LARGE and, on thin or "
                    "closely-spaced features (e.g. a thin fin, or two nearby but separate surface "
                    "sheets), a point can get snapped to the WRONG, more distant part of the "
                    "surface instead of the nearby correct one -- creating a genuinely wrong result "
                    "rather than just an imprecise one. The default, 0.5, is a reasonable middle "
                    "ground for typical meshes; only lower it if you're seeing snap-to-wrong-surface "
                    "artifacts on thin features, and only raise it if you still see uncorrected "
                    "drift/detail loss after confirming adjust is enabled.")),
                io.Float.Input("adjust_border_importance", default=2.0, min=0.1, max=20.0, step=0.1, display_mode="number", tooltip=(
                    "Extra weighting applied specifically to boundary/border edges during the "
                    "adjust step above, so open mesh boundaries get preferentially held in place "
                    "rather than allowed to drift or shrink inward the way interior geometry might "
                    "during CVT relaxation. "
                    "HOW TO PICK A VALUE: the default, 2.0, already biases toward preserving "
                    "boundaries over interior detail, which is usually what you want for an open "
                    "(non-watertight) mesh -- shrinking/rounding boundary edges is a much more "
                    "visually obvious defect than minor interior smoothing. Raise this further (5-10) "
                    "if you're specifically seeing boundary edges creep inward or lose their shape "
                    "after remeshing an open mesh; this parameter has no effect at all on a fully "
                    "closed/watertight mesh, since there are no border edges for it to act on.")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, nb_points=5000, nb_lloyd_iter=5, nb_newton_iter=30, newton_m=7,
                adjust="true", adjust_max_edge_distance=0.5, adjust_border_importance=2.0):
        import pygeogram

        log.info("Backend: geogram_smooth")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: nb_points=%s, nb_lloyd_iter=%s, nb_newton_iter=%s, newton_m=%s, "
                 "adjust=%s, adjust_max_edge_distance=%s, adjust_border_importance=%s",
                 f"{nb_points:,}", nb_lloyd_iter, nb_newton_iter, newton_m,
                 adjust, adjust_max_edge_distance, adjust_border_importance)

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        V_out, F_out = pygeogram.remesh_smooth(
            V, F, nb_points,
            nb_lloyd_iter=nb_lloyd_iter,
            nb_newton_iter=nb_newton_iter,
            newton_m=newton_m,
            adjust=(adjust == "true"),
            adjust_max_edge_distance=float(adjust_max_edge_distance),
            adjust_border_importance=float(adjust_border_importance),
        )

        remeshed_mesh = trimesh_module.Trimesh(vertices=V_out, faces=F_out, process=False)
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'geogram_smooth',
            'nb_points': nb_points,
            'nb_lloyd_iter': nb_lloyd_iter,
            'nb_newton_iter': nb_newton_iter,
            'newton_m': newton_m,
            'adjust': adjust,
            'adjust_max_edge_distance': adjust_max_edge_distance,
            'adjust_border_importance': adjust_border_importance,
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (Geogram CVT Smooth): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"nb_points={nb_points:,}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_GeogramSmooth": RemeshGeogramSmoothNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_GeogramSmooth": "Remesh Geogram Smooth (backend)"}
