# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Fix Self Intersections node - single frontend with backend selector.

Dispatches to the existing standalone self-intersection fixers via node expansion
(GraphBuilder), auto-chaining GeomPackDetectSelfIntersections internally: every backend
gets a fresh field (face_attributes['self_intersecting'] / vertex_attributes
['intersection_flag']/['intersection_count']) recomputed on its OUTPUT, unconditionally
-- not left to each backend's own optional re-detect toggle. Removal/Perturbation
additionally get Detect run BEFORE them too, since they require the field as input
(no need to manually wire a separate Detect node upstream anymore). Remesh works from
raw geometry, so it only needs the after-step. GeomPackDetectSelfIntersections itself
stays a standalone, visible node -- it's inspection, not a fix.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class FixSelfIntersectionsNode(io.ComfyNode):
    """Fix Self Intersections - unified self-intersection repair with backend selection."""

    BACKEND_MAP = {
        "removal": "GeomPackFixSelfIntersectionsByRemoval",
        "perturbation": "GeomPackFixSelfIntersectionsByPerturbation",
        "remesh": "GeomPackRemeshSelfIntersections",
    }

    # Each backend's mesh input parameter has a different name (inherited unchanged
    # from the standalone nodes being dispatched to).
    MESH_PARAM = {
        "removal": "trimesh",
        "perturbation": "trimesh",
        "remesh": "mesh",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackFixSelfIntersections",
            display_name="Fix Self Intersections",
            category="geompack/repair",
            description=(
                "Fix self-intersecting geometry. Pick a backend; each exposes its own toggles. "
                "Self-intersection detection runs automatically (before and/or after, as needed) "
                "-- no need to wire a separate 'Detect Self Intersections' node.\n"
                "\n"
                "removal: deletes intersecting faces, then optionally fills the resulting holes "
                "and fixes normals.\n"
                "perturbation: nudges vertices adjacent to intersections along their normals -- "
                "non-destructive, preserves topology.\n"
                "remesh: subdivides intersecting triangles via libigl/CGAL so intersections lie "
                "exactly on edges."
            ),
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.DynamicCombo.Input("backend", tooltip="Self-intersection fix algorithm and backend", options=[
                    io.DynamicCombo.Option("removal", [
                        io.Boolean.Input("fill_holes", default=True, tooltip=(
                            "After deleting the faces flagged as self-intersecting, the removed patch "
                            "leaves an open hole in the mesh. When enabled, trimesh's fill_holes() "
                            "re-triangulates that gap so the surface stays closed. Disable if you'd "
                            "rather inspect the resulting hole, or if you plan to fill it yourself with "
                            "a more controllable node (the dedicated Fill Holes dispatcher offers "
                            "CGAL/pymeshfix/igl_fan backends with more quality control than this quick fill).")),
                        io.Boolean.Input("fix_normals", default=True, tooltip=(
                            "Recomputes/reorients face normals after the removal+fill step, since "
                            "deleting faces and adding new triangulated ones can leave inconsistent "
                            "winding at the seam. Cheap and safe to leave on; only disable if you're "
                            "already running a dedicated Fix Normals pass afterward anyway.")),
                        io.Int.Input("max_hole_size", default=100, min=3, max=10000, tooltip=(
                            "Upper bound, in boundary edges, on how large a hole this node's own "
                            "fill_holes step will attempt to close. Removing a cluster of intersecting "
                            "faces can leave a large, irregular hole; big holes are expensive and often "
                            "triangulate poorly with this simple fill. Holes larger than this are left "
                            "open -- feed the result into the dedicated Fill Holes node afterward for a "
                            "higher-quality fill on anything that didn't get closed here.")),
                    ]),
                    io.DynamicCombo.Option("perturbation", [
                        io.Float.Input("epsilon", default=0.001, min=1e-8, max=1.0, step=0.0001, tooltip=(
                            "Base displacement distance, in the mesh's own world units (not a "
                            "percentage), applied along each affected vertex's normal. Too small and "
                            "the intersection may persist; too large and you visibly deform the "
                            "surface near the fix site. 0.001 is a reasonable starting point for a mesh "
                            "roughly normalized to unit scale -- for CAD-derived meshes in mm or with a "
                            "much larger/smaller bounding box, scale this relative to the mesh's "
                            "typical edge length rather than using the default blindly.")),
                        io.Int.Input("max_iterations", default=10, min=1, max=100, tooltip=(
                            "Number of perturbation passes. Displacement ramps up linearly across "
                            "iterations (iteration 1 moves epsilon/max_iterations, the final iteration "
                            "moves the full epsilon) -- so more iterations means a gentler ramp toward "
                            "the same final displacement, not more total movement. Increase if a single "
                            "pass still leaves faces marginally intersecting; each extra iteration also "
                            "recomputes vertex normals, so cost scales roughly linearly with this value.")),
                        io.Combo.Input("direction", options=["outward", "inward", "adaptive"], default="adaptive", tooltip=(
                            "Which way to nudge affected vertices along their normals. 'outward' pushes "
                            "vertices away from the surface (grows the mesh slightly at the intersection "
                            "site); 'inward' pulls them in (shrinks it slightly); 'adaptive' is intended "
                            "to choose per-vertex based on local geometry, but this backend's current "
                            "implementation falls back to the same behavior as 'outward' -- functionally "
                            "an alias for now. Kept as the default anyway since it's the intended "
                            "long-term behavior once true per-vertex direction selection lands; switch "
                            "to 'inward' explicitly if you specifically need the mesh to shrink rather "
                            "than grow at the fix site.")),
                        io.Boolean.Input("scale_by_intersection_count", default=True, tooltip=(
                            "When enabled, vertices touched by more intersecting faces get "
                            "proportionally larger displacement (scaled 0-1 against the mesh's maximum "
                            "per-vertex intersection count from the internal Detect pass), so heavily "
                            "tangled regions get nudged harder than lightly-affected ones. Disable for a "
                            "uniform epsilon everywhere, regardless of how many intersections each "
                            "vertex participates in -- useful if you want a predictable, constant "
                            "displacement rather than one that varies across the mesh.")),
                    ]),
                    io.DynamicCombo.Option("remesh", [
                        io.Boolean.Input("remove_unreferenced", default=True, tooltip=(
                            "After CGAL subdivides intersecting triangles along their intersection "
                            "curves, some original vertices may end up referenced by no face at all. "
                            "Enabling this strips them out (via igl.remove_unreferenced) so the output "
                            "mesh's vertex list stays compact. Safe to leave on; only matters if "
                            "something downstream depends on vertex indices/count staying untouched "
                            "from the input mesh.")),
                        io.Boolean.Input("extract_outer_hull", default=False, tooltip=(
                            "Attempts to extract just the outer, manifold shell of the remeshed result "
                            "(igl.outer_hull_legacy), discarding any interior geometry the subdivision "
                            "created at the intersection sites. Produces a cleaner, genuinely manifold "
                            "mesh suitable for boolean operations or 3D printing, but is significantly "
                            "slower than the base remesh step and can fail or silently no-op on complex "
                            "topology. Leave off for a quick fix; enable when you specifically need a "
                            "watertight, printable result and are willing to pay the extra cost.")),
                        io.Boolean.Input("stitch_all", default=True, tooltip=(
                            "Passed straight through to CGAL's remesh_self_intersections call. When "
                            "enabled, CGAL attempts to stitch together the boundaries created by "
                            "subdividing intersecting triangles, keeping the result as connected/closed "
                            "as possible rather than leaving separate disconnected patches at each "
                            "intersection site. Recommended on by default; disabling it can leave more "
                            "fragmented geometry that's harder to clean up afterward.")),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="fixed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    DETECT_NODE_ID = "GeomPackDetectSelfIntersections"

    @classmethod
    def execute(cls, mesh, backend):
        from comfy_execution.graph_utils import GraphBuilder

        if cls.SCHEMA is None:
            cls.GET_SCHEMA()

        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]
        mesh_param = cls.MESH_PARAM[selected]

        log.info("Fix Self Intersections dispatch: %s -> %s", selected, node_id)

        kwargs = {k: v for k, v in backend.items() if k != "backend"}

        graph = GraphBuilder()

        if selected == "remesh":
            # Works from raw geometry -- no pre-detection needed, always a real fix pass.
            kwargs["detect_only"] = False
            fix_node = graph.node(node_id, mesh=mesh, **kwargs)
        else:
            # Removal/perturbation read the field to know which faces/vertices to touch,
            # so detect first. The dispatcher re-detects after unconditionally (below),
            # so the backend's own re-detect is redundant here -- skip it.
            detect_before = graph.node(cls.DETECT_NODE_ID, trimesh=mesh)
            kwargs["re_detect_after_fix"] = False
            fix_node = graph.node(node_id, **{mesh_param: detect_before.out(0)}, **kwargs)

        # Always recompute the field on the final output, regardless of backend, so the
        # result never needs a manually-wired Detect node to inspect what (if anything) remains.
        detect_after = graph.node(cls.DETECT_NODE_ID, trimesh=fix_node.out(0))

        return {
            "result": (detect_after.out(0), fix_node.out(1)),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {"GeomPackFixSelfIntersections": FixSelfIntersectionsNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackFixSelfIntersections": "Fix Self Intersections"}
