# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Remesh Node - Single frontend with backend selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes, each running in its own isolation env.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshNode(io.ComfyNode):
    """
    Remesh - Unified remeshing with backend selection.

    Dispatches to hidden backend nodes via node expansion.
    Backends span multiple isolation envs (main, blender, gpu).
    """

    # Map DynamicCombo option key -> hidden backend node_id
    BACKEND_MAP = {
        "pymeshlab_isotropic": "GeomPackRemesh_PyMeshLab",
        "instant_meshes":      "GeomPackRemesh_InstantMeshes",
        "quadriflow":          "GeomPackRemesh_QuadriFlow",
        "mmg_adaptive":        "GeomPackRemesh_MMG",
        "geogram_smooth":      "GeomPackRemesh_GeogramSmooth",
        "geogram_anisotropic": "GeomPackRemesh_GeogramAniso",
        "pmp_uniform":         "GeomPackRemesh_PMPUniform",
        "pmp_adaptive":        "GeomPackRemesh_PMPAdaptive",
        "quadwild":            "GeomPackRemesh_QuadWild",
        "cgal_isotropic":      "GeomPackRemesh_CGAL",
        "blender_voxel":       "GeomPackRemesh_BlenderVoxel",
        "blender_sharp":       "GeomPackRemesh_BlenderSharp",
        "blender_blocks":      "GeomPackRemesh_BlenderBlocks",
        "gpu_cumesh":          "GeomPackRemesh_GPU",
        "faithc":              "GeomPackRemesh_FaithC",
    }

    # Some frontend param names differ from backend param names
    # (to avoid conflicts between DynamicCombo options).
    # Map: frontend_param_name -> backend_param_name
    PARAM_REMAP = {
        "cgal_edge_length": "target_edge_length",
        "cgal_iterations": "iterations",
        "gpu_target_face_count": "target_face_count",
        "gpu_grid_resolution": "grid_resolution",
        "gpu_project_back": "project_back",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh",
            display_name="Remesh",
            category="geompack/remeshing",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.DynamicCombo.Input("backend", tooltip="Remeshing algorithm and backend", options=[
                    # ---- Main env backends ----
                    io.DynamicCombo.Option("pymeshlab_isotropic", [
                        io.Float.Input("target_edge_length", default=1.00, min=0.0001, max=10.0, step=0.0001, display_mode="number", tooltip="Target edge length for output triangles, in world units (relative to mesh scale). Used only when target_vertices and target_faces are both 0."),
                        io.Int.Input("target_vertices", default=0, min=0, max=20000000, step=100, tooltip="Target output vertex count (0 = off). Back-solves the edge length from the mesh area; overrides target_edge_length. Approximate."),
                        io.Int.Input("target_faces", default=0, min=0, max=40000000, step=100, tooltip="Target output face count (0 = off). Back-solves the edge length from the mesh area; overrides target_vertices and target_edge_length. Approximate."),
                        io.Int.Input("iterations", default=3, min=1, max=20, step=1, tooltip="Number of remeshing passes. More iterations = smoother result, slower processing."),
                        io.Float.Input("feature_angle", default=30.0, min=0.0, max=180.0, step=0.1, tooltip="Angle threshold (degrees) for feature/crease edge detection -- edges sharper than this are preserved. Lower = preserve more edges; 180 = none."),
                        io.Combo.Input("adaptive", options=["true", "false"], default="false", tooltip="Use curvature-adaptive edge lengths."),
                        io.Combo.Input("reproject", options=["true", "false"], default="true", tooltip="Reproject vertices back onto the original surface after each iteration (Botsch back-projection). true = stay faithful to the input surface (recommended); false = pure tangential smoothing, which lets vertices drift off the surface."),
                    ]),
                    io.DynamicCombo.Option("instant_meshes", [
                        io.Int.Input("target_vertex_count", default=5000, min=100, max=1000000, step=100, tooltip="Target vertex count. Creates field-aligned quad-dominant mesh."),
                        io.Combo.Input("deterministic", options=["true", "false"], default="true", tooltip="Use deterministic algorithm for reproducible results."),
                        io.Float.Input("crease_angle", default=0.0, min=0.0, max=180.0, step=1.0, tooltip="Angle threshold for preserving sharp edges. 0 = no preservation."),
                    ]),
                    io.DynamicCombo.Option("quadriflow", [
                        io.Int.Input("target_face_count", default=5000, min=100, max=5000000, step=100, tooltip="Target output faces (quad-dominant). QuadriFlow hits this fairly accurately."),
                        io.Combo.Input("preserve_sharp", options=["true", "false"], default="false", tooltip="Align quads to sharp edges and keep them crisp. QuadriFlow's sharp threshold is HARDCODED at 60 deg (an edge is 'sharp' if adjacent face normals deviate > 60 deg) -- not adjustable from the binding. Turn ON for CAD/mechanical parts."),
                        io.Combo.Input("preserve_boundary", options=["true", "false"], default="true", tooltip="Keep mesh boundary/open edges fixed (for open meshes)."),
                        io.Combo.Input("adaptive_scale", options=["false", "true"], default="false", tooltip="Curvature-adaptive quad sizing: smaller quads where curvature is high, larger on flats. Great for CAD; spends faces where detail is."),
                        io.Combo.Input("minimum_cost_flow", options=["false", "true"], default="false", tooltip="Min-cost-flow solver for the integer step -> cleaner connectivity / better singularities. Slower, more regular."),
                        io.Combo.Input("aggressive_sat", options=["false", "true"], default="false", tooltip="SAT solver for a fully-integer, seamless result with fewest singularities (highest quality). Slowest."),
                        io.Int.Input("seed", default=0, min=0, max=2000000000, step=1, tooltip="Random seed for field initialization (reproducible results)."),
                        # QuadriFlow cost scales with INPUT faces, not the
                        # target -- decimating dense inputs to ~2-3x the target
                        # first is the standard speedup (output barely changes;
                        # QuadriFlow rebuilds topology from scratch anyway).
                        # Plain Boolean + Int, NOT a nested DynamicCombo: a
                        # nested combo's assembled dict cannot cross the
                        # GraphBuilder dispatch to the backend node (the
                        # expander matches it against "on"/"off" option keys,
                        # fails, and silently drops the input -- the toggle
                        # arrived as off no matter what the user set).
                        io.Boolean.Input("pre_decimate", default=False, tooltip="Quadric-collapse the input before remeshing. Big speedup on dense inputs; output quality nearly unchanged (QuadriFlow rebuilds topology from scratch anyway)."),
                        io.Int.Input("pre_decimate_faces", default=40000, min=1000, max=5000000, step=1000, tooltip="Only used when pre_decimate is on: reduce the input to this many faces first. Rule of thumb: 2-3x target_face_count. No-op if the input is already at or below this count."),
                    ]),
                    # MMG surface remesher (mmgs). DEFAULT sizing is Hausdorff-DRIVEN (curvature-
                    # adaptive): hausd sets a geometric error bound, hmin/hmax clamp it, hgrad
                    # smooths the size field. hsiz overrides all that with a UNIFORM size. ar =
                    # ridge/feature-edge preservation. (MMGS has no 'nosurf' option.)
                    io.DynamicCombo.Option("mmg_adaptive", [
                        io.Float.Input("hausd", default=0.01, min=0.0001, max=10.0, step=0.0001, display_mode="number", tooltip=(
                            "Hausdorff distance -- THE adaptive driver: max geometric deviation of "
                            "the remeshed surface from the original. To stay within hausd of a "
                            "curved patch you need small triangles, so flats stay coarse and curves "
                            "get fine -- that's the curvature-adaptivity. SMALL = hug surface (fine, "
                            "more faces); LARGE = drift (coarse).\n"
                            "ABSOLUTE world units (NOT normalised). MMG's CLI default is ~0.01 x "
                            "bbox-diagonal (~4.0 on a 400-unit part), so the literal 0.01 here is "
                            "~400x finer and explodes the face count. Set ~bbox_diag * 0.005-0.02.")),
                        io.Float.Input("hmin", default=0.0, min=0.0, max=10.0, step=0.001, display_mode="number", tooltip=(
                            "Minimum edge length CLAMP (world units): won't shrink below this even "
                            "where curvature/hausd wants finer. 0 = auto. Caps face count on curves.")),
                        io.Float.Input("hmax", default=0.0, min=0.0, max=100.0, step=0.01, display_mode="number", tooltip=(
                            "Maximum edge length CLAMP (world units): won't grow beyond this even on "
                            "big flats. 0 = auto. Forces a minimum density on flat faces.")),
                        io.Float.Input("hgrad", default=1.3, min=1.0, max=5.0, step=0.1, display_mode="number", tooltip=(
                            "Gradation -- max size RATIO between adjacent edges = how fast triangle "
                            "size may change. 1.0 = near-uniform (many faces); 1.3 = smooth "
                            "(default); 2-3 = abrupt jumps (fewer faces). Lower = smoother, more "
                            "triangles.")),
                        io.Float.Input("ar", default=-1.0, min=-1.0, max=180.0, step=1.0, display_mode="number", tooltip=(
                            "RIDGE (feature-edge) detection angle, DEGREES. An edge is kept as a "
                            "sharp ridge and PRESERVED when its dihedral exceeds this. LOWER = keep "
                            "more features (30 = preserve shallow chamfers/creases); HIGHER = only "
                            "very sharp survive. -1 = MMG default (~45 deg). For CAD use ~30-40 to "
                            "protect creases. The feature control plain isotropic remesh lacks.")),
                        io.Float.Input("hsiz", default=0.0, min=0.0, max=100.0, step=0.001, display_mode="number", tooltip=(
                            "Constant edge size = UNIFORM mode. 0 = OFF (use adaptive hausd). Set >0 "
                            "to OVERRIDE hmin/hmax/hausd-adaptivity and produce an even ISOTROPIC "
                            "mesh of this edge length (world units) -- MMG as a feature-preserving "
                            "isotropic remesher. Use for UNIFORM density (e.g. before L0). Absolute "
                            "length; set relative to mesh scale.")),
                        io.Combo.Input("optim", options=["false", "true"], default="false", tooltip=(
                            "Optimization mode: improve triangle QUALITY of the existing mesh "
                            "keeping sizes roughly as-is, few/no insertions -- gentle clean-up, not "
                            "a re-size. Pair with noinsert to forbid adding points.")),
                        io.Combo.Input("nreg", options=["false", "true"], default="false", tooltip=(
                            "Normal regularization: smooth the per-vertex normals MMG curves new "
                            "triangles onto -- reduces faceting on NOISY input (scan/Tripo/MC) at a "
                            "slight cost to sharp transitions. Helps image-derived meshes; off for "
                            "clean CAD.")),
                        io.Combo.Input("anisosize", options=["false", "true"], default="false", tooltip=(
                            "ANISOTROPIC sizing: long thin triangles aligned to curvature -- far "
                            "fewer faces for the same fidelity on cylinders/developables. CAVEAT: "
                            "needs a per-vertex TENSOR METRIC field this node does NOT supply; "
                            "enabling alone will no-op or error. Leave OFF until a metric input is "
                            "wired.")),
                        io.Combo.Input("noinsert", options=["false", "true"], default="false", tooltip=(
                            "Disable point INSERTION (no edge splits): only collapse/swap/move "
                            "existing vertices, never add. Forbids increasing resolution. Pairs with "
                            "optim.")),
                        io.Combo.Input("noswap", options=["false", "true"], default="false", tooltip=(
                            "Disable edge SWAP (flip). Off by default -- swaps Delaunay-ize and "
                            "remove caps/large angles. Enable only to preserve input connectivity or "
                            "debug; hurts quality.")),
                        io.Combo.Input("nomove", options=["false", "true"], default="false", tooltip=(
                            "Disable point RELOCATION (tangential smoothing). Off by default. Enable "
                            "to keep vertices exactly in place (only insert/collapse/swap).")),
                        io.Combo.Input("keep_ref", options=["false", "true"], default="false", tooltip=(
                            "Keep edge REFERENCES (keepRef): preserve MMG's ridge/boundary tags on "
                            "the output. Relevant for .sol/.mesh round-trips; harmless to leave off.")),
                    ]),
                    io.DynamicCombo.Option("geogram_smooth", [
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
                    ]),
                    io.DynamicCombo.Option("geogram_anisotropic", [
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
                    ]),
                    io.DynamicCombo.Option("pmp_uniform", [
                        io.Float.Input("pmp_edge_length", default=1.0, min=0.001, max=100.0, step=0.01, display_mode="number", tooltip="Target edge length for uniform remeshing. Used only when target_vertices is 0."),
                        io.Int.Input("pmp_target_vertices", default=0, min=0, max=20000000, step=100, tooltip="Target output vertices (0 = off, use edge_length). Back-solved from input mesh area; overrides edge_length. Approximate."),
                        io.Int.Input("pmp_iterations", default=10, min=1, max=100, step=1, tooltip="Number of remeshing iterations."),
                        io.Combo.Input("pmp_use_projection", options=["true", "false"], default="true", tooltip="Project vertices back onto input surface."),
                    ]),
                    io.DynamicCombo.Option("pmp_adaptive", [
                        io.Float.Input("pmp_min_edge", default=0.1, min=0.001, max=100.0, step=0.01, display_mode="number", tooltip="Minimum edge length (high-curvature areas)."),
                        io.Float.Input("pmp_max_edge", default=2.0, min=0.01, max=100.0, step=0.01, display_mode="number", tooltip="Maximum edge length (flat areas)."),
                        io.Float.Input("pmp_approx_error", default=0.1, min=0.001, max=10.0, step=0.01, display_mode="number", tooltip="Maximum geometric approximation error."),
                        io.Int.Input("pmp_adaptive_iterations", default=10, min=1, max=100, step=1, tooltip="Number of remeshing iterations."),
                        io.Combo.Input("pmp_adaptive_projection", options=["true", "false"], default="true", tooltip="Project vertices back onto input surface."),
                    ]),
                    io.DynamicCombo.Option("quadwild", [
                        io.Float.Input("qw_sharp_angle", default=35.0, min=0.0, max=180.0, step=1.0, tooltip="Dihedral angle threshold for sharp feature detection."),
                        io.Float.Input("qw_alpha", default=0.02, min=0.005, max=0.1, step=0.005, display_mode="number", tooltip="Balances quad-grid REGULARITY vs feature ALIGNMENT (QuadWild's BiMDF alpha weight). Lower (~0.005) = a more uniform, regular quad grid with fewer singularities (irregular vertices), but the quad edges follow surface features/curvature less. Higher (~0.1) = allows more singularities so the quad flow bends to align with curvature and sharp edges, at the cost of grid uniformity. Typical 0.005-0.1; default 0.02."),
                        io.Float.Input("qw_scale_factor", default=1.0, min=0.1, max=10.0, step=0.1, tooltip="Quad size multiplier. Larger = bigger quads, fewer faces. Used only when target_vertices and target_edge_length are both 0."),
                        io.Int.Input("qw_target_vertices", default=0, min=0, max=20000000, step=100, tooltip="Target output vertices (0 = off, use scale_factor). Back-solved from input mesh area; overrides scale_factor."),
                        io.Float.Input("qw_target_edge_length", default=0.0, min=0.0, max=1000.0, step=0.001, display_mode="number", tooltip="Target quad edge length (0 = off). Overrides BOTH scale_factor and target_vertices."),
                        io.Combo.Input("qw_remesh", options=["true", "false"], default="true", tooltip="Pre-remesh input for better triangle quality."),
                        io.Combo.Input("qw_smooth", options=["true", "false"], default="true", tooltip="Smooth output mesh topology after quadrangulation."),
                    ]),
                    # ---- CGAL backend ----
                    io.DynamicCombo.Option("cgal_isotropic", [
                        io.Float.Input("cgal_edge_length", default=1.00, min=0.001, max=10.0, step=0.01, display_mode="number", tooltip="Target edge length for CGAL isotropic remeshing."),
                        io.Int.Input("cgal_iterations", default=3, min=1, max=20, step=1, tooltip="Number of remeshing passes."),
                        io.Combo.Input("protect_boundaries", options=["true", "false"], default="true", tooltip="Lock boundary/open edges in place during remeshing."),
                    ]),
                    # ---- Blender backends ----
                    io.DynamicCombo.Option("blender_voxel", [
                        # Range/default follow Blender's remesh_voxel_size (default
                        # 0.1, hard range 0.0001..FLT_MAX); the old max=1.0 cap
                        # blocked coarse remeshing of mm-unit CAD parts.
                        io.Float.Input("voxel_size", default=0.1, min=0.0001, max=100.0, step=0.01, display_mode="number", tooltip="Voxel size in mesh units. Smaller = more detail. Output is always watertight. Blender default 0.1; for large (mm-unit) CAD parts use bbox_diagonal/100 as a starting point."),
                        io.Float.Input("adaptivity", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Post-remesh simplification: collapses faces in flat regions while keeping detail. 0 = uniform density (heaviest), 1 = maximum reduction."),
                        io.Boolean.Input("fix_poles", default=False, tooltip="Produce cleaner topology around poles at some extra cost."),
                        io.Boolean.Input("preserve_volume", default=True, tooltip="Project the remeshed surface back onto the original so thin features and small parts do not shrink. Blender defaults this on."),
                    ]),
                    io.DynamicCombo.Option("blender_sharp", [
                        io.Int.Input("octree_depth", default=6, min=1, max=12, step=1, tooltip="Octree resolution -- the detail knob. Power of 2: each +1 roughly QUADRUPLES face count and halves voxel size. 6 is a sane start; 8-9 is high detail; 10+ can be very heavy."),
                        io.Float.Input("scale", default=0.9, min=0.0, max=0.99, step=0.05, display_mode="number", tooltip="Octree fit relative to the bounding box (0-0.99). Higher = grid hugs the mesh tighter = finer effective resolution; too close to 1.0 can clip the outer shell. 0.9 default."),
                        io.Float.Input("sharpness", default=1.0, min=0.0, max=2.0, step=0.1, display_mode="number", tooltip="How aggressively dual-contouring snaps to sharp edges/corners. Higher = crisper edges but can spike on noisy input; lower = rounder. 0-2 is Blender's normal slider range (1.0 default); capped at 2 here since higher just over-sharpens."),
                        io.Combo.Input("remove_disconnected", options=["true", "false"], default="true", tooltip="Delete small disconnected (floating) pieces after remeshing. ON by default (matches Blender)."),
                        io.Float.Input("disconnected_threshold", default=1.0, min=0.0, max=1.0, step=0.05, display_mode="number", tooltip="Size cutoff for removal, relative to the largest component. Higher = remove more aggressively (1.0 ~ keep only the main body); lower = keep more; 0 = keep everything. Only used when remove_disconnected is on."),
                    ]),
                    io.DynamicCombo.Option("blender_blocks", [
                        io.Int.Input("octree_depth", default=6, min=1, max=10, step=1, tooltip="Resolution. Higher = more detail, more faces."),
                        io.Float.Input("scale", default=0.9, min=0.0, max=1.0, step=0.05, display_mode="number", tooltip="Ratio of output size to input bounding box."),
                    ]),
                    # ---- GPU backend ----
                    io.DynamicCombo.Option("gpu_cumesh", [
                        io.Int.Input("gpu_grid_resolution", default=512, min=32, max=2048, step=16, tooltip="Dual-contouring grid resolution -- the main detail knob. Higher = finer detail + more faces + slower/more VRAM; lower = coarser/faster."),
                        io.Int.Input("gpu_target_face_count", default=500000, min=1000, max=5000000, step=1000, tooltip="Target faces after simplification."),
                        io.Float.Input("remesh_band", default=1.0, min=0.1, max=5.0, step=0.1, tooltip="Thickness of the voxel shell dual-contouring evaluates around the surface, in VOXELS (real width = band x scale/grid_resolution). Robustness-vs-detail knob, NOT the main detail knob (that's grid_resolution). 1.0 (default) = one-voxel shell, right almost always. 0.5 = hugs the surface: keeps thin walls + sharp detail but can leave holes if input is noisy / resolution too low. 2-3 = thicker: fills gaps, robust on messy/non-watertight input, smoother -- but fuses nearby thin walls and rounds detail; slower. Above ~3 rarely helps. Holes -> raise band; thin features vanishing -> lower band; crisper detail -> raise grid_resolution."),
                        io.Float.Input("gpu_project_back", default=0.0, min=0.0, max=2.0, step=0.05, tooltip="Re-project DC vertices back onto the input surface for sharper fidelity (0 = off)."),
                        io.Combo.Input("remove_degenerate_faces", options=["true", "false"], default="false", tooltip="Cleanup: drop zero-area / sliver faces."),
                        io.Combo.Input("remove_duplicate_faces", options=["true", "false"], default="false", tooltip="Cleanup: drop exact-duplicate faces."),
                        io.Combo.Input("repair_non_manifold_edges", options=["true", "false"], default="false", tooltip="Cleanup: mend non-manifold edges (fix variant)."),
                        io.Combo.Input("remove_non_manifold_faces", options=["true", "false"], default="false", tooltip="Cleanup: drop faces that create non-manifold edges."),
                        io.Float.Input("remove_small_components_min_area", default=0.0, min=0.0, max=1.0, step=0.001, display_mode="number", tooltip="Cleanup: drop floating components below this area (0 = off). Great for recon/Tripo crumbs."),
                        io.Combo.Input("remove_unreferenced_vertices", options=["true", "false"], default="false", tooltip="Cleanup: drop orphan vertices (no face uses them)."),
                    ]),
                    io.DynamicCombo.Option("faithc", [
                        io.Int.Input("faithc_max_level", default=7, min=4, max=10, step=1, tooltip="FaithC octree depth = grid resolution (2^level: 7=128, 8=256, 9=512). The main density knob; power-of-2 only, so this IS the grid resolution. Higher = finer + more faces + more VRAM."),
                        io.Int.Input("faithc_min_level", default=4, min=1, max=7, step=1, tooltip="Coarsest octree level where traversal starts (<= max_level). Default 4; rarely changed."),
                        io.Float.Input("faithc_lambda_n", default=1.0, min=0.0, max=10.0, step=0.1, tooltip="QEF normal-alignment weight: higher = anchors snap onto surface tangent planes -> sharper, more faithful edges. Default 1.0."),
                        io.Float.Input("faithc_lambda_d", default=0.1, min=0.0, max=10.0, step=0.05, tooltip="QEF distance/regularization weight: higher = anchors pulled to voxel centers -> more regular, blunts detail. Default 0.1."),
                        io.Float.Input("faithc_weight_power", default=1.0, min=0.1, max=4.0, step=0.1, tooltip="Exponent on per-constraint QEF weights. >1 emphasizes the most confident surface samples. Advanced; leave 1.0."),
                        io.Combo.Input("faithc_clamp_anchors", options=["false", "true"], default="false", tooltip="Clamp anchors to voxel bounds + reproject onto the surface. Removes spikes on noisy/thin geometry at slight cost to sharpness."),
                        io.Combo.Input("faithc_triangulation", options=["auto", "length", "angle", "normal_abs", "normal", "simple_02", "simple_13"], default="auto", tooltip="Quad->triangle split. auto (recommended), length=shorter diagonal, angle=best triangle quality, normal/normal_abs=align to surface normal, simple_02/13=fixed diagonal."),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, backend):
        from comfy_execution.graph_utils import GraphBuilder

        # Ensure SCHEMA is initialized (worker subprocess doesn't call GET_SCHEMA)
        if cls.SCHEMA is None:
            cls.GET_SCHEMA()

        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]

        log.info("Remesh dispatch: %s -> %s", selected, node_id)

        # Build kwargs for the backend node: mesh + backend-specific params
        kwargs = {"trimesh": trimesh}
        for k, v in backend.items():
            if k == "backend":
                continue
            # A nested DynamicCombo (e.g. quadriflow's pre_decimate toggle)
            # arrives here as an assembled dict. It CANNOT be forwarded
            # verbatim: the expanded backend node's schema parser compares the
            # live value against the option keys ("on"/"off"), a dict matches
            # neither, and the input is silently dropped -- the user's toggle
            # became off. Unpack to the backend node's plain scalar inputs
            # (<name> bool + the option's own params) instead.
            if isinstance(v, dict):
                sel = v.get(k)
                kwargs[cls.PARAM_REMAP.get(k, k)] = (
                    sel is True or sel == "on" or sel == "true")
                for nk, nv in v.items():
                    if nk == k:
                        continue
                    kwargs[cls.PARAM_REMAP.get(nk, nk)] = nv
                continue
            # Remap param names if needed (frontend name -> backend name)
            backend_key = cls.PARAM_REMAP.get(k, k)
            kwargs[backend_key] = v

        graph = GraphBuilder()
        backend_node = graph.node(node_id, **kwargs)

        return {
            "result": (backend_node.out(0), backend_node.out(1)),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "GeomPackRemesh": RemeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemesh": "Remesh",
}
