# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Sharpen Mesh Node - Single frontend with backend selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes.

Available backends:
- two_step: Two-phase bilateral normal filtering via pymeshlab. Smooths face
  normals (respecting dihedral angle thresholds), then repositions vertices to
  match. Sharpens creases while keeping faces flat. Best for CAD-like geometry
  from marching cubes, scanning, or neural SDF extraction.
- unsharp_mask: Geometric unsharp masking via pymeshlab. Subtracts a smoothed
  version from the original to amplify ridges and valleys.
- libigl_unsharp: Cotangent-weighted geometric unsharp mask via libigl.
  Geometrically superior to uniform-weight unsharp masking because cotangent
  Laplacian respects mesh geometry (triangle shape/area).
- l0_minimize: L0 normal minimization (He & Schaefer 2013). Minimizes the
  number of distinct face normal orientations, forcing the mesh into
  piecewise-flat regions with sharp edges at boundaries. Best for aggressive
  CAD-like sharpening.
- guided_normal: Guided mesh normal filtering (Zhang et al. 2015). Uses a
  min-range-metric guidance signal to drive bilateral normal filtering while
  preserving sharp edges. Interleaves vertex updates within normal iterations.
- fast_effective: Fast and Effective Feature-Preserving Mesh Denoising
  (Sun et al. TVCG 2007). Uses thresholded cosine-similarity weights for
  normal filtering: w = max(0, dot(ni,nj) - T)^2. Simple and fast.
- non_iterative: Non-Iterative Feature-Preserving Mesh Smoothing
  (Jones et al. SIGGRAPH 2003). Mollifies normals on a smoothed mesh copy,
  then does a single-pass bilateral vertex update using spatial and influence
  Gaussian weights with BFS face neighbor search.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class SharpenMeshNode(io.ComfyNode):
    """
    Sharpen Mesh - Unified sharpening with backend selection.

    Dispatches to hidden backend nodes via node expansion.
    """

    BACKEND_MAP = {
        "two_step":       "GeomPackSharpen_TwoStep",
        "unsharp_mask":   "GeomPackSharpen_UnsharpMask",
        "libigl_unsharp": "GeomPackSharpen_LibiglUnsharp",
        "l0_minimize":    "GeomPackSharpen_L0Minimize",
        "l0_faithful":    "GeomPackSharpen_L0Faithful",
        "guided_normal":  "GeomPackSharpen_GuidedNormal",
        "fast_effective":  "GeomPackSharpen_FastEffective",
        "non_iterative":  "GeomPackSharpen_NonIterative",
        "curvature_guided": "GeomPackSharpen_CurvatureGuided",
        "decrease_gaussian": "GeomPackSharpen_DecreaseGaussian",
        "taubin_sharpen": "GeomPackSharpen_Taubin",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpenMesh",
            display_name="Sharpen Mesh",
            category="geompack/smoothing",
            description=(
                "Sharpen / de-noise a mesh by edge-preserving normal filtering, then move "
                "vertices to match the filtered normals: flat regions get smoothed while "
                "sharp feature edges are kept (or enhanced). Choose a method in 'backend'; "
                "each exposes its own knobs (hover any widget for details).\n"
                "\n"
                "GUIDED NORMAL (Zhang et al. 2015) is a 2-stage bilateral filter: "
                "(1) for every face it picks a 'guidance' normal from the flattest "
                "sub-neighborhood, so the filter knows where the true surface is; "
                "(2) it averages each face normal with its neighbors, weighting by face "
                "area, spatial distance (sigma_s) and how much the guidance normals differ "
                "(sigma_r). Faces across a sharp edge have very different guidance normals "
                "-> near-zero weight -> the edge survives; faces on the same flat region get "
                "averaged -> noise removed. sigma_r is in DEGREES (angle at which normals "
                "stop blending: small = sharper, large = smoother); sigma_s is the spatial "
                "reach in multiples of average edge length. use_gpu runs a faithful "
                "vectorized torch port (CUDA), ~17x faster on large meshes."
            ),
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.DynamicCombo.Input("backend", tooltip=(
                        "Sharpening algorithm. "
                        "two_step=bilateral normal filtering (recommended for CAD-like edges), "
                        "unsharp_mask=geometric unsharp masking (pymeshlab), "
                        "libigl_unsharp=cotangent-weighted unsharp (geometry-aware), "
                        "l0_minimize=piecewise-flat L0 optimization (aggressive CAD prep), "
                        "guided_normal=guided normal filtering with min-range-metric (controllable), "
                        "fast_effective=thresholded cosine weight normal filtering (fast), "
                        "non_iterative=mollified normal single-pass bilateral (non-iterative)"
                    ), options=[
                    io.DynamicCombo.Option("two_step", [
                        io.Int.Input("smooth_steps", default=3, min=1, max=50, step=1, tooltip=(
                            "Number of two-step smoothing passes. "
                            "More steps = stronger sharpening effect."
                        )),
                        io.Float.Input("normal_threshold", default=60.0, min=0.0, max=180.0, step=0.5, tooltip=(
                            "Dihedral angle threshold in degrees. "
                            "Edges sharper than this angle are preserved as features. "
                            "Lower = more aggressive (more edges treated as creases). "
                            "60 is a good default for most CAD models."
                        )),
                    ]),
                    io.DynamicCombo.Option("unsharp_mask", [
                        io.Float.Input("weight", default=0.3, min=0.0, max=3.0, step=0.01, tooltip=(
                            "Unsharp mask weight controlling sharpening strength. "
                            "Higher = more pronounced sharpening."
                        )),
                        io.Int.Input("iterations", default=5, min=1, max=50, step=1, tooltip=(
                            "Smoothing iterations for the reference smooth mesh. "
                            "More iterations = larger-scale sharpening."
                        )),
                    ]),
                    io.DynamicCombo.Option("libigl_unsharp", [
                        io.Float.Input("weight", default=0.5, min=0.01, max=5.0, step=0.01, tooltip=(
                            "How much detail to add back. 0.5 = subtle sharpening, "
                            "1.0 = double the detail, 2.0+ = aggressive."
                        )),
                        io.Int.Input("iterations", default=3, min=1, max=50, step=1, tooltip=(
                            "Smoothing iterations for the reference mesh. "
                            "More iterations = smoother reference = sharpens broader features. "
                            "Fewer iterations = sharpens fine detail."
                        )),
                    ]),
                    io.DynamicCombo.Option("l0_minimize", [
                        io.Float.Input("alpha", default=0.001, min=0.0001, max=4.0, step=0.0001, tooltip=(
                            "Angle threshold -- SCALE-INVARIANT (depends on the angle between "
                            "faces, NOT on mesh size or vertex/face count). Adjacent faces whose "
                            "dihedral angle is below it get snapped flat.\n"
                            "alpha = 2*(1 - cos theta), range 0..4. Approx:\n"
                            "  0.001 ~ 1.8 deg\n  0.01 ~ 5.7 deg\n  0.1 ~ 18 deg\n"
                            "  0.5 ~ 41 deg\n  2.0 = 90 deg\n  4.0 = 180 deg\n"
                            "Grows by 'beta' each iteration."
                        )),
                        io.Float.Input("beta", default=2.0, min=1.1, max=10.0, step=0.1, tooltip=(
                            "Multiplies alpha after each iteration, so the angle threshold "
                            "escalates and progressively flattens sharper transitions. "
                            "NOTE: has NO effect with iterations=1 (alpha is only multiplied "
                            "between iterations). Use several iterations to see beta work."
                        )),
                        io.Int.Input("iterations", default=10, min=1, max=50, step=1, tooltip=(
                            "Number of L0 iterations. Each does ONE gentle vertex pass at the "
                            "current alpha, then alpha *= beta. With iterations=1 you get a "
                            "single small pass at the initial alpha (often barely visible) -- "
                            "use more iterations (and/or higher alpha) for stronger, propagated "
                            "flattening. Denser meshes need more iterations to spread flatness."
                        )),
                        io.Combo.Input("use_gpu", options=["true", "false"], default="true", tooltip=(
                            "Run the vectorized torch implementation instead of the pure-Python "
                            "loop. Uses CUDA when available (else vectorized CPU torch) -- orders "
                            "of magnitude faster on large meshes (the CPU path loops over every "
                            "edge in Python and is impractical beyond ~50k faces). Results can "
                            "differ slightly from the CPU path. Default true."
                        )),
                    ]),
                    # l0_faithful = the REAL global He & Schaefer 2013 L0 denoiser (vs 'l0_minimize'
                    # which is a local normal-snap). WHAT L0 MEANS: at every edge measure how much
                    # the surface BENDS; a real CAD shape is piecewise-flat (zero bend almost
                    # everywhere, a few sharp creases) = a SPARSE set of bends. L0 = "count the
                    # nonzero bends" -> minimizing it forces flat regions truly flat while the few
                    # real creases stay sharp (where L2 smoothing like Taubin rounds them off).
                    io.DynamicCombo.Option("l0_faithful", [
                        io.Float.Input("gamma_factor", default=0.2, min=0.0001, max=30.0, step=0.01,
                                       display_mode="number", tooltip=(
                            "THE main knob -- sets lambda, the PRICE the optimizer pays to KEEP each "
                            "crease. It minimizes  |stay near the input|^2  +  lambda * (number of "
                            "bent edges).\n"
                            "  HIGH gamma -> bends are expensive -> keep very few -> AGGRESSIVE "
                            "flattening; only the strongest creases survive (clean CAD faces, but "
                            "shallow features get erased).\n"
                            "  LOW gamma -> bends are cheap -> keep many -> GENTLE (preserves subtle "
                            "detail AND residual noise).\n"
                            "It is the crease-vs-noise threshold: an edge's bend is kept only when "
                            "|bend|^2 >= lambda/beta. lambda = gamma_factor * avg_edge^2 * "
                            "avg_dihedral, so it is SCALE-INVARIANT (avg_edge^2 -> identical result "
                            "on a 1mm or 1m part) and noise-adaptive (avg_dihedral). 0.2 = reference "
                            "default (validated: faces -> <1deg off-axis, a 90deg crease stays "
                            ">88deg). 0.02 = the paper's milder value. TUNE THIS: up for cleaner "
                            "faces, down to keep subtle features.\n"
                            "REASONABLE RANGE: 0.02 - 0.5. 0.02 = gentle (keeps fine detail/some "
                            "noise), 0.1 - 0.3 = typical CAD cleanup, 0.2 = default, >0.5 = "
                            "aggressive (erases shallow features), >1 = very flat (rarely useful).")),
                        io.Float.Input("beta_mu", default=1.414, min=1.05, max=3.0, step=0.01, tooltip=(
                            "Controls the NUMBER of iterations (quality vs speed, NOT 'how sharp').\n"
                            "WHY beta exists: 'count the bends' (L0) can't be optimized directly (a "
                            "bend is on/off, no gradient), so the method SPLITS it -- a helper "
                            "variable decides each bend (keep it, or zero it) and a penalty of "
                            "strength beta forces the surface to agree. beta is ramped up like "
                            "cooling metal: starts at 1e-3 (loose, smooths) and is multiplied by "
                            "beta_mu EACH iteration up to beta_max=1e3 (tight, decisions lock in). "
                            "beta_mu = how fast you tighten, so SMALLER = more, finer steps = better "
                            "feature preservation but slower.\n"
                            "REASONABLE RANGE: 1.1 - 2.0. Sweet spot 1.2 - 1.5. Below ~1.1 = "
                            "diminishing returns (lots of iters, marginal gain); above 2.0 = too "
                            "coarse (decisions jump). iters ~= ln(1e6)/ln(beta_mu):\n"
                            "  1.05 = ~283 (overkill)   1.10 = ~145 (very fine)\n"
                            "  1.20 = ~76 (fine)        1.414 = ~40 (DEFAULT, balanced)\n"
                            "  1.70 = ~26               2.0 = ~20 (fast/coarse)\n"
                            "GPU path makes the extra iterations cheap.")),
                        io.Float.Input("alpha_factor", default=0.1, min=0.0, max=2.0, step=0.01, tooltip=(
                            "Anti-fold safety strap; rarely changed -- leave at 0.1 (0 disables).\n"
                            "alpha weights a fairing term R = (p1 - p2 + p3 - p4) per edge that stops "
                            "triangles flipping over / overshooting when a crease snaps into place. "
                            "alpha0 = alpha_factor * avg_dihedral and HALVES every iteration: strong "
                            "early (while vertices move a lot, keeps the solve stable) then fades to "
                            "~0 so it doesn't blur the final result. Measured effect is small (it "
                            "decays fast).\n"
                            "REASONABLE RANGE: 0.0 - 0.3. 0.1 = default; raise toward 0.2-0.3 only "
                            "if you see triangles folding/spiking at sharp creases; 0 = off.")),
                        io.Float.Input("beta_max", default=1000.0, min=10.0, max=100000.0, step=10.0, tooltip=(
                            "Solver stop condition -- iterate until beta reaches this (paper 1e3). "
                            "beta ramps from 1e-3; by ~1e3 the keep/flatten decisions are fully "
                            "committed, so raising it mostly just runs extra iterations that lock in "
                            "residual detail (weak, LOGARITHMIC lever: 10x here only adds ~6-7 iters "
                            "at default beta_mu -- use beta_mu to change iteration count instead).\n"
                            "REASONABLE RANGE: 100 - 10000. 1000 = default; rarely worth changing.")),
                        io.Combo.Input("use_gpu", options=["true", "false"], default="true", tooltip=(
                            "Which solver runs the per-iteration system  (I + alpha*R^T R + "
                            "beta*D^T D) p = p* + beta*D^T delta.\n"
                            "  true = torch (CUDA if available, else CPU torch) with a warm-started "
                            "block CONJUGATE-GRADIENT solve. CG (not a direct factorization) because "
                            "that matrix changes EVERY iteration -- the bend operator D is rebuilt at "
                            "the current vertices -- so there is nothing to reuse on either device; "
                            "warm-starting from the previous iterate + the +I term make CG converge "
                            "in a few steps. Scales to large meshes; float32 so results differ "
                            "slightly. Falls back to CPU-direct if the GPU path errors.\n"
                            "  false = exact scipy sparse-DIRECT solve (SuperLU, float64). Best for "
                            "small/medium meshes; gets slow past ~100k vertices because the matrix "
                            "is refactorised every iteration.")),
                    ]),
                    io.DynamicCombo.Option("guided_normal", [
                        io.Int.Input("normal_iterations", default=5, min=1, max=1000, step=1, tooltip=(
                            "Iterations for guided bilateral normal filtering. "
                            "More iterations produce smoother/flatter regions while "
                            "preserving sharp edges. Main strength knob; high values "
                            "(50-200) progressively flatten."
                        )),
                        io.Int.Input("vertex_iterations", default=10, min=1, max=100, step=1, tooltip=(
                            "Iterations for updating vertex positions to match filtered "
                            "normals. More iterations give better convergence."
                        )),
                        io.Int.Input("neighborhood_rings", default=1, min=1, max=4, step=1, tooltip=(
                            "Face neighborhood size (k-ring) for guidance + filtering. "
                            "1 = faces sharing a vertex (standard). Higher = wider footprint, "
                            "so each pass hits HARDER / reaches farther (stronger than just "
                            "more iterations) -- but cost & memory grow ~rings^2. Try 2 for a "
                            "noticeably stronger effect."
                        )),
                        io.Float.Input("sigma_s", default=1.0, min=0.1, max=10.0, step=0.1, tooltip=(
                            "SPATIAL scale = neighborhood size as a multiple of the average "
                            "edge length. How far (in surface distance) a neighbor face still "
                            "influences the filter. Larger = wider smoothing (can blur small "
                            "features); smaller = more local. Default 1.0 (~one ring)."
                        )),
                        io.Float.Input("sigma_r_degrees", default=20.0, min=1.0, max=120.0, step=1.0, tooltip=(
                            "RANGE scale, in DEGREES: the angle between two faces' normals at "
                            "which they stop being averaged together (the edge-preservation "
                            "knob). SMALLER (e.g. 10 deg) = only near-parallel faces blend = "
                            "sharper. LARGER (e.g. 45 deg) = more faces blend = smoother. "
                            "Default 20 deg."
                        )),
                        io.Float.Input("vertex_anchor", default=0.5, min=0.01, max=10.0, step=0.01, display_mode="number", tooltip=(
                            "Drag-back strength of the foldless vertex update: each vertex is "
                            "pulled toward its ORIGINAL position while moving to match the "
                            "filtered normals. This regularization keeps strong smoothing "
                            "STABLE -- it stops the update from overshooting at creases, "
                            "collapsing triangles, and folding (which the old sweep did at high "
                            "iteration counts). LOWER (~0.05) = stronger smoothing; HIGHER "
                            "(~2.0) = gentler. Default 0.5."
                        )),
                        io.Combo.Input("use_gpu", options=["false", "true"], default="false", tooltip=(
                            "Run the faithful vectorized torch port instead of the per-face "
                            "Python loops. Uses CUDA when available (else vectorized CPU torch) "
                            "-- much faster on large meshes. Same guidance + bilateral filter; "
                            "results can differ slightly (float32 vs float64)."
                        )),
                    ]),
                    io.DynamicCombo.Option("fast_effective", [
                        io.Float.Input("threshold_T", default=0.5, min=1e-10, max=1.0, step=0.01, tooltip=(
                            "Cosine similarity threshold (Sun et al. TVCG 2007). "
                            "Normals with dot(ni,nj) > T contribute with weight "
                            "(dot-T)^2; below T they contribute nothing. "
                            "Lower = more normals averaged (smoother), "
                            "higher = only very similar normals averaged (sharper). "
                            "0.5 is a good default."
                        )),
                        io.Int.Input("normal_iterations", default=20, min=1, max=500, step=1, tooltip=(
                            "Iterations for normal filtering. More iterations "
                            "produce stronger flattening of near-flat regions."
                        )),
                        io.Int.Input("vertex_iterations", default=50, min=1, max=500, step=1, tooltip=(
                            "Iterations for vertex position update from filtered "
                            "normals. Boundary vertices are kept fixed."
                        )),
                    ]),
                    io.DynamicCombo.Option("non_iterative", [
                        io.Float.Input("sigma_f", default=1.0, min=0.001, max=10.0, step=0.1, tooltip=(
                            "Spatial sigma as multiple of average edge length "
                            "(Jones et al. SIGGRAPH 2003). Controls spatial extent "
                            "of the bilateral filter. Face neighbors are searched "
                            "within radius 2*sigma_f. Larger = smoother."
                        )),
                        io.Float.Input("sigma_g", default=1.0, min=0.001, max=10.0, step=0.1, tooltip=(
                            "Influence sigma as multiple of average edge length. "
                            "Controls sensitivity to projection distance (how far "
                            "the vertex moves toward each face plane). Smaller = "
                            "more feature-preserving."
                        )),
                    ]),
                    io.DynamicCombo.Option("decrease_gaussian", [
                        io.Int.Input("iterations", default=20, min=1, max=2000, step=1, tooltip=(
                            "Gradient-descent steps on the Gaussian-curvature energy sum K_i^2 "
                            "(K_i = vertex angle defect). More steps = lower |Gaussian curvature| "
                            "= more DEVELOPABLE (planes/cylinders/cones, K->0). Gauss-Bonnet keeps "
                            "the topological total fixed, so the flow flattens the bulk and the "
                            "anchor keeps the overall shape.")),
                        io.Float.Input("strength", default=0.05, min=0.001, max=1.0, step=0.001, display_mode="number", tooltip=(
                            "Per-step size as a fraction of average edge length (largest vertex "
                            "move ~ strength*edge). The foldless barrier prevents triangle "
                            "inversion regardless. Default 0.05.")),
                        io.Float.Input("anchor_weight", default=0.5, min=0.0, max=50.0, step=0.01, display_mode="number", tooltip=(
                            "Tikhonov lambda: how strongly vertices stay at their ORIGINAL "
                            "positions while K is reduced. LOWER = flatten harder (bigger shape "
                            "change toward developable); HIGHER = stay close to input. Default 0.5.")),
                        io.Combo.Input("regularizer", options=["developable", "reduce"], default="developable", tooltip=(
                            "developable = L1/sparsity on Gaussian curvature: push small K to 0 "
                            "and CONCENTRATE it onto sparse seams -> piecewise zero-Gaussian "
                            "(planes+cylinders+cones kept smooth, only seams curve). The CAD mode "
                            "(doesn't facet fillets). reduce = L2: lower |K| everywhere (gentler). "
                            "On real Tripo meshes developable removes ~35-50% of spurious K.")),
                        io.Float.Input("irls_eps", default=0.005, min=0.0005, max=0.5, step=0.0005, display_mode="number", tooltip=(
                            "(developable) IRLS L1 sparsity epsilon w=1/(|K|+eps). SMALLER = more "
                            "L0-like (crisper developable patches / sharper seams). Default 0.005.")),
                        io.Combo.Input("use_gpu", options=["true", "false"], default="true", tooltip=(
                            "Run on CUDA (recommended). false = CPU torch.")),
                    ]),
                    io.DynamicCombo.Option("taubin_sharpen", [
                        io.Float.Input("lambda_", default=0.5, min=0.01, max=1.0, step=0.01, tooltip=(
                            "Taubin low-pass shrink step. Higher = stronger smoothing per pass "
                            "(coarser feature scale -> coarser detail amplified).")),
                        io.Float.Input("mu", default=-0.53, min=-1.0, max=-0.01, step=0.01, tooltip=(
                            "Taubin un-shrink step (negative). |mu| > lambda keeps the low-pass "
                            "shrink-free. Typical -0.53 for lambda=0.5.")),
                        io.Int.Input("iterations", default=10, min=1, max=200, step=1, tooltip=(
                            "Taubin low-pass passes. MORE = smoother reference = sharpens "
                            "BROADER features; FEWER = sharpens fine detail.")),
                        io.Float.Input("enhancement", default=0.6, min=0.0, max=5.0, step=0.05, display_mode="number", tooltip=(
                            "Unsharp strength alpha: V_out = V0 + alpha*(V0 - low-pass). 0 = no "
                            "change, 1 = double the detail, >1 = aggressive. Pure Taubin "
                            "sharpening (built only from the Taubin smoother, stable). The "
                            "foldless barrier prevents triangle inversion.")),
                        io.Combo.Input("anti_flip", options=["true", "false"], default="true", tooltip=(
                            "Pass the displacement through the signed-area barrier so no "
                            "triangle folds (only matters at high enhancement).")),
                        io.Combo.Input("use_gpu", options=["true", "false"], default="true", tooltip=(
                            "Run on CUDA (recommended). false = CPU torch.")),
                    ]),
                    io.DynamicCombo.Option("curvature_guided", [
                        io.Combo.Input("regularizer", options=["tv", "bilateral"], default="tv", tooltip=(
                            "tv = TOTAL-VARIATION on the curvature field (Chambolle-Pock) -> "
                            "PIECEWISE-CONSTANT curvature: genuine regions of constant curvature "
                            "(planes/cylinders/spheres) with crisp jumps. bilateral = edge-aware "
                            "diffusion (denoises but makes smooth ramps, not plateaus)."
                        )),
                        io.Float.Input("tv_weight", default=0.5, min=0.02, max=8.0, step=0.02, display_mode="number", tooltip=(
                            "TV strength (tv mode), relative to median curvature. Higher = flatter, "
                            "fewer/larger constant-curvature regions; lower = more detail."
                        )),
                        io.Int.Input("iterations", default=5, min=0, max=100, step=1, tooltip=(
                            "tv: scales Chambolle-Pock passes (~iters x 30). bilateral: diffusion "
                            "passes on the curvature field. More = stronger / wider reach."
                        )),
                        io.Float.Input("sigma_s", default=2.0, min=0.1, max=10.0, step=0.1, tooltip=(
                            "(bilateral mode) Spatial scale (x average edge length)."
                        )),
                        io.Float.Input("curvature_sigma", default=0.5, min=0.02, max=5.0, step=0.02, display_mode="number", tooltip=(
                            "(bilateral mode) CURVATURE range scale (relative to curvature spread). "
                            "Smaller = crisper region boundaries. (tv mode uses tv_weight.)"
                        )),
                        io.Float.Input("anchor_weight", default=0.1, min=0.001, max=10.0, step=0.001, display_mode="number", tooltip=(
                            "How strongly the reconstruction sticks to the input positions "
                            "(Tikhonov lambda). LOWER = stronger curvature-domain reshaping; "
                            "HIGHER = stay close to input. Default 0.1."
                        )),
                        io.Int.Input("cg_iters", default=200, min=10, max=2000, step=10, tooltip=(
                            "Max preconditioned-CG iterations for the reconstruction solve."
                        )),
                        io.Combo.Input("use_gpu", options=["true", "false"], default="true", tooltip=(
                            "Run on CUDA (recommended -- torch sparse mat-vecs). false = CPU torch."
                        )),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, backend):
        from comfy_execution.graph_utils import GraphBuilder

        if cls.SCHEMA is None:
            cls.GET_SCHEMA()

        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]

        log.info("Sharpen dispatch: %s -> %s", selected, node_id)

        kwargs = {"trimesh": trimesh}
        for k, v in backend.items():
            if k == "backend":
                continue
            kwargs[k] = v

        graph = GraphBuilder()
        backend_node = graph.node(node_id, **kwargs)

        return {
            "result": (backend_node.out(0), backend_node.out(1)),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "GeomPackSharpenMesh": SharpenMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackSharpenMesh": "Sharpen Mesh",
}
