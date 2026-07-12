# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Mesh Repair node - single frontend with backend selector.

Cleans up degenerate / sliver / duplicate / non-manifold elements. Dispatches to
hidden backend nodes via node expansion (GraphBuilder), each in its own isolation
env. Self-intersection fixing lives in its own dispatcher (GeomPackFixSelfIntersections,
nodes/repair_cgal/fix_self_intersections.py) -- a distinct topology problem, not
degeneracy/duplication.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class MeshRepairNode(io.ComfyNode):
    """Mesh Repair - unified degenerate/sliver cleanup with backend selection."""

    BACKEND_MAP = {
        "trimesh": "GeomPackMeshRepair_Trimesh",
        "pymeshlab": "GeomPackMeshRepair_PyMeshLab",
        "pymeshfix": "GeomPackMeshRepair_PyMeshFix",
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackMeshRepair",
            display_name="Mesh Repair",
            category="geompack/repair",
            description=(
                "Clean up bad geometry: degenerate / sliver / folded / duplicate faces, "
                "duplicate vertices, T-vertices, non-manifold edges, and small floating "
                "components. Pick a backend; each exposes its own toggles.\n"
                "\n"
                "Self-intersections are NOT handled here -- use the separate 'Fix Self "
                "Intersections' dispatcher for that (a distinct topology problem: overlapping "
                "geometry, not degeneracy/duplication).\n"
                "\n"
                "trimesh (lightweight, no extra deps): a 2-step pipeline -- merge "
                "near-duplicate vertices within a tolerance, THEN drop degenerate/cap-sliver "
                "faces (which the merge step can itself create). Good default when you don't "
                "want an extra dependency, or as a quick pre-pass before a heavier backend.\n"
                "\n"
                "pymeshlab (per-operation cleanup): the most complete, most surgical toggle "
                "set -- MeshLab's own meshing_remove_* filters, each independently "
                "switchable. Best when you want fine control over exactly which defect "
                "categories get touched, or when you need non-manifold-edge repair "
                "specifically (the other two backends don't offer it).\n"
                "\n"
                "pymeshfix (component + intersection focus): small-component removal/joining "
                "plus PyTMesh's own clean() pass, which targets self-intersections AND "
                "degenerate faces together in one iterative loop. Complements pymeshlab -- "
                "reach for this when the mesh has floating debris you want either dropped or "
                "stitched back together, or lingering self-intersections a pure degenerate-"
                "face pass won't touch. (Hole-filling is intentionally NOT exposed here -- "
                "that's the dedicated 'Fill Holes' dispatcher's job, including its own "
                "pymeshfix backend.)"
            ),
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.DynamicCombo.Input("backend", tooltip=(
                    "Repair algorithm and backend. trimesh = lightweight vertex-merge + "
                    "degenerate/cap-sliver cleanup, no extra deps. pymeshlab = the most "
                    "complete per-operation toggle set (duplicate verts/faces, folded faces, "
                    "T-vertices, non-manifold edges, small components) -- the only backend "
                    "here that repairs non-manifold edges. pymeshfix = small-component "
                    "removal/joining plus PyTMesh's combined self-intersection + degenerate-"
                    "face clean() pass. See the node description for a fuller breakdown of "
                    "when to reach for each."
                ), options=[
                    io.DynamicCombo.Option("trimesh", [
                        io.Float.Input("tolerance", default=1e-5, min=1e-8, max=1e-2, step=1e-6,
                                       tooltip=(
                            "Distance tolerance (in the mesh's own world units, NOT a percentage) "
                            "for merging near-duplicate vertices, run FIRST before the "
                            "degenerate-face cleanup below. Vertices that are SUPPOSED to be the "
                            "same point -- two triangles meeting at an edge, a boolean seam, a "
                            "re-exported CAD boundary -- often differ by tiny floating-point noise "
                            "from meshing/export, so they end up as separate, unwelded vertices "
                            "instead of one shared one. This rounds coordinates to roughly "
                            "-log10(tolerance) decimal digits and merges whatever lands on the same "
                            "rounded position.\n\n"
                            "HOW TO PICK A VALUE: start at the default, 1e-5. Then look at the "
                            "result: if the mesh still looks visibly fragmented into separate "
                            "floating pieces that should obviously be one connected part, the "
                            "tolerance was too TIGHT -- go up 10x at a time (1e-4, then 1e-3) until "
                            "it connects. If instead you see thin walls, small gaps, or fine details "
                            "vanishing/pinching shut that were there before, you went too LOOSE -- "
                            "go back down 10x. 1e-5 assumes a mesh roughly 1-100 units across "
                            "(typical for a CAD part in meters or a normalized model); for something "
                            "in millimeters with a bounding box in the thousands, or something "
                            "pre-scaled to a tiny fraction of a unit, think of this value as a "
                            "FRACTION OF THE MESH'S OWN TYPICAL EDGE LENGTH rather than trusting "
                            "1e-5 verbatim -- a mesh 1000x bigger than 'normal' needs a tolerance "
                            "roughly 1000x bigger too, not the same absolute number.")),
                        io.Float.Input("min_area", default=1e-10, min=0.0, max=1.0, step=1e-10,
                                       tooltip=(
                            "After the vertex merge above, faces with area below this ABSOLUTE "
                            "threshold are deleted outright. A literally-zero-area triangle (its "
                            "three corners are collinear, or two of them are now the same point "
                            "after merging) contributes nothing to volume or rendering and can "
                            "break downstream algorithms that divide by face area -- normal "
                            "weighting, curvature estimation, hole-fill triangulation, mass "
                            "properties, etc. This specifically also catches faces the MERGE step "
                            "above may have just created a moment ago (two of a triangle's three "
                            "corners collapsing onto the same vertex during merge instantly makes "
                            "that triangle degenerate), on top of anything already degenerate in "
                            "the input.\n\n"
                            "HOW TO PICK A VALUE: leave this at the default (1e-10) for almost every "
                            "mesh -- it's deliberately microscopic so it only kills TRUE zero-area "
                            "slivers, never real geometry. 0 turns the check off completely (rarely "
                            "useful; there's essentially no downside to leaving it on at the "
                            "default). Do NOT reach for this if your mesh still 'looks' bad after a "
                            "run -- most visually-obvious sliver problems are actually CAP triangles "
                            "(see max_angle_deg below), which can have totally normal area and this "
                            "check will never touch them. Only raise min_area above default if "
                            "you've explicitly confirmed (e.g. by comparing face counts before/"
                            "after) that it's removing real slivers and not eating legitimate small "
                            "detail -- raising it much past 1e-10 starts risking exactly that.")),
                        io.Float.Input("max_angle_deg", default=180.0, min=90.0, max=180.0, step=0.5,
                                       tooltip=(
                            "Cap-sliver removal by COLLAPSE (runs before the min_area deletion "
                            "step, on the merged mesh). A 'cap' is an extremely obtuse triangle -- "
                            "one interior angle close to 180 degrees -- whose apex sits almost "
                            "exactly ON the opposite edge. Visually and topologically this behaves "
                            "like a crack or a near-hole even though it can have a perfectly "
                            "ordinary, non-zero AREA, so min_area above will NOT catch it. These "
                            "typically come from OCC/CAD meshing at sharp face boundaries, or from "
                            "image/heightmap-derived meshes (e.g. Depth Map to Mesh) where a pixel "
                            "grid produces near-degenerate triangles along steep gradients.\n\n"
                            "A face whose LARGEST interior angle is >= this threshold gets its apex "
                            "COLLAPSED onto the nearer of its two base vertices (rather than deleted "
                            "outright, which would leave an actual hole) -- this closes the gap "
                            "smoothly and keeps the mesh connected, at the cost of merging that "
                            "vertex into its neighbor.\n\n"
                            "HOW TO PICK A VALUE: 180 = COMPLETELY OFF (no real triangle reaches "
                            "exactly 180 degrees, so nothing is ever touched) -- if you're not sure "
                            "your mesh has cap slivers, leave it at 180 first and only come back to "
                            "this if you see thin cracks or slit-like artifacts in an otherwise "
                            "'watertight' mesh. If you do suspect caps (common after Depth Map to "
                            "Mesh, or OCC-meshed CAD with lots of thin/sharp faces), start "
                            "CONSERVATIVE at 179 and step DOWN one or two degrees at a time (178, "
                            "177, 176...), re-checking the mesh after each step -- lower numbers are "
                            "MORE aggressive (they treat less-extreme, fatter triangles as caps "
                            "too). Most real fixes land somewhere in 175-179; below about 170 you "
                            "start collapsing triangles that are just naturally sharp/thin by "
                            "design, not actually broken, so treat anything under 170 as a red flag "
                            "that you've gone too far. 90 (the minimum) would flag nearly every "
                            "triangle as a cap and is never what you want.")),
                    ]),
                    io.DynamicCombo.Option("pymeshlab", [
                        io.Combo.Input("remove_duplicate_vertices", options=["true", "false"], default="true", tooltip=(
                            "Merge coincident vertices FIRST, before every other filter below runs "
                            "(MeshLab's meshing_remove_duplicate_vertices). Uses MeshLab's own "
                            "internal epsilon for 'coincident' -- unlike the trimesh backend's "
                            "explicit, user-tunable tolerance, there's no knob to loosen or tighten "
                            "here. This has to run first because every filter below relies on "
                            "connectivity being correct: two vertices at the exact same 3D position "
                            "but still carrying separate indices will look 'disconnected' to the "
                            "folded-face, duplicate-face, and non-manifold-edge checks below, which "
                            "silently hides real defects from them.\n\n"
                            "DECISION: this is not really a judgment call -- leave it ON essentially "
                            "always. The only time to turn it off is if you've already merged "
                            "vertices upstream (e.g. ran the trimesh backend first with your own "
                            "tolerance) and want to skip redundant work.")),
                        io.Combo.Input("remove_null_faces", options=["true", "false"], default="true", tooltip=(
                            "MeshLab's meshing_remove_null_faces filter -- the core sliver/"
                            "degenerate-face remover, and the closest pymeshlab equivalent to the "
                            "trimesh backend's min_area check. A 'null face' is a triangle with "
                            "zero or near-zero area: either its three corners are collinear, or two "
                            "of them coincide (often freshly created a moment ago by the duplicate-"
                            "vertex merge above). Uses MeshLab's own internal epsilon, not an exposed "
                            "number -- if you need explicit numeric control over how aggressive the "
                            "area cutoff is, use the trimesh backend's min_area input instead.\n\n"
                            "DECISION: leave ON for essentially any mesh with meshing/boolean/export "
                            "artifacts -- there is no realistic downside, it never removes anything "
                            "with actual area.")),
                        io.Combo.Input("remove_folded_faces", options=["true", "false"], default="true", tooltip=(
                            "Remove 'folded' faces: a triangle whose normal points almost exactly "
                            "opposite to its neighbor's, meaning it has doubled back and folded flat "
                            "against the adjacent surface (a near-180-degree dihedral fold) instead "
                            "of extending outward normally. Visually these look like a thin spike or "
                            "crease pressed flat against the mesh, and they're a common byproduct of "
                            "bad boolean operations, self-intersecting extrusions, or numerically "
                            "unstable meshing near sharp features. Distinct from remove_null_faces "
                            "above: a folded face can have completely ordinary area and still be "
                            "geometrically wrong, since the defect is in its ORIENTATION relative to "
                            "its neighbor, not its size.\n\n"
                            "DECISION: leave ON by default -- it only removes triangles that were "
                            "already contributing nothing but visual noise. Specifically worth double-"
                            "checking is ON if your mesh came out of a boolean operation or has "
                            "visible thin spikes/creases.")),
                        io.Combo.Input("remove_duplicate_faces", options=["true", "false"], default="true", tooltip=(
                            "Remove exact-duplicate faces: two triangles referencing the identical "
                            "set of 3 vertices (regardless of winding order). Happens when "
                            "overlapping surfaces get merged, a mesh gets accidentally appended to "
                            "itself, or an export/import round-trip duplicates geometry. Easy to "
                            "miss visually in solid shading (the duplicate sits exactly on top of the "
                            "original) but silently doubles triangle count, can cause z-fighting in "
                            "rendering, and throws off any volume/area/mass computation that assumes "
                            "each surface patch is counted once.\n\n"
                            "DECISION: leave ON -- pure redundancy removal, no real downside.")),
                        io.Combo.Input("remove_t_vertices", options=["true", "false"], default="false", tooltip=(
                            "Remove T-vertices: a vertex that lies exactly ON another face's edge "
                            "without being a proper shared corner of that face -- picture a 'T' "
                            "junction where one triangle's edge midpoint touches, but isn't welded "
                            "to, a neighboring triangle's edge. Geometrically the two surfaces touch "
                            "with zero gap, but topologically they don't share a true edge, which can "
                            "produce visible cracks under subdivision, displacement, or certain "
                            "boolean/remeshing algorithms that walk edge-by-edge rather than "
                            "point-in-space.\n\n"
                            "OFF by default for a real reason: fixing this RE-TRIANGULATES the "
                            "affected neighboring faces (splitting them so the T-vertex becomes a "
                            "real shared corner), which changes topology and face count in a way "
                            "none of the other toggles here do, and can disturb any per-face/per-"
                            "vertex data (UVs, custom scalar fields, cad_face_id-style metadata) tied "
                            "to the old triangulation.\n\n"
                            "DECISION: only turn ON if you're specifically seeing cracks/gaps appear "
                            "under subdivision or displacement mapping, and you don't have per-face "
                            "attributes you need to keep exactly as-is. If you're not chasing that "
                            "specific symptom, leave it off.")),
                        io.Combo.Input("repair_non_manifold_edges", options=["true", "false"], default="true", tooltip=(
                            "Repair non-manifold edges: an edge shared by 3 or more faces, instead "
                            "of the 2 (interior) or 1 (open boundary) that a proper manifold surface "
                            "allows. This is the ONE defect category unique to this pymeshlab backend "
                            "-- neither the trimesh nor pymeshfix backends in this dispatcher touch "
                            "non-manifold edges at all. Physically nonsensical for a 'solid' -- you "
                            "can't 3D print or run a clean boolean operation against a mesh that has "
                            "them -- and typically arises from merging separate parts along a shared "
                            "seam, self-intersecting geometry, or duplicate overlapping faces the "
                            "filters above didn't fully clean up. The repair disconnects the extra "
                            "fan of faces at each offending edge so every edge ends up shared by at "
                            "most 2 faces again, which CAN locally split the mesh into more pieces "
                            "than it started with.\n\n"
                            "DECISION: turn ON whenever the mesh needs to be watertight, 3D-"
                            "printable, or boolean-safe -- non-manifold edges break those use cases "
                            "outright, there's no partial fix. If all you care about is visual "
                            "rendering and you want to preserve face count as closely as possible, "
                            "you can leave it off, but there's rarely a real downside to leaving it "
                            "on. If turning this on noticeably fragments the mesh, follow up with "
                            "remove_small_components_pct below (or the pymeshfix backend) to clean up "
                            "the resulting pieces.")),
                        io.Float.Input("remove_small_components_pct", default=0.0, min=0.0, max=100.0, step=0.5, display_mode="number", tooltip=(
                            "Drop floating connected components whose bounding-box diagonal is below "
                            "this PERCENT of the WHOLE MESH's bounding-box diagonal -- NOT a "
                            "percentage of face count, vertex count, or volume, and not an absolute "
                            "distance. This is pymeshlab's own PercentageValue mechanism "
                            "(meshing_remove_connected_component_by_diameter), so the same number "
                            "automatically means something proportionally similar whether the mesh "
                            "is a 10cm part or a 10m building.\n\n"
                            "Classic use case: stray debris -- photogrammetry/3D-scan noise floating "
                            "near the real surface, tiny disconnected fragments left behind by a "
                            "boolean operation, or leftover crumbs after repair_non_manifold_edges "
                            "above splits something apart.\n\n"
                            "HOW TO PICK A VALUE: 0 = OFF, nothing removed, matches input exactly. "
                            "This is NOT a 'when in doubt crank it up' slider -- start small, 1 to 2, "
                            "and actually LOOK at the result before going further. If real small "
                            "parts disappeared (a bolt, a screw, a small standalone detail that's "
                            "genuinely supposed to be its own piece), you went too far -- drop back "
                            "to 0.5-1. If debris is still floating around, go up in small steps (2, "
                            "3, 4, 5) rather than jumping straight to a big number. Once you're above "
                            "roughly 5-10 you're removing components a THIRD the size of major mesh "
                            "features, which is rarely what anyone actually wants -- if you need to "
                            "go that high to clear the debris, the debris and the real geometry are "
                            "probably too close in size for this filter to tell apart automatically, "
                            "and you should inspect the mesh manually instead of trusting a single "
                            "number.")),
                        io.Combo.Input("remove_unreferenced_vertices", options=["true", "false"], default="true", tooltip=(
                            "Drop orphan vertices -- ones no longer referenced by any face -- as the "
                            "LAST step, after every face-deleting filter above has run (null faces, "
                            "folded faces, duplicate faces, non-manifold-edge repair, small-component "
                            "removal all leave behind vertices that used to belong to deleted faces "
                            "but are now unused).\n\n"
                            "DECISION: leave ON always. Purely a housekeeping pass -- changes nothing "
                            "about the surviving surface, just trims the vertex array. No real "
                            "reason to ever turn this off.")),
                    ]),
                    io.DynamicCombo.Option("pymeshfix", [
                        io.Combo.Input("remove_small_components", options=["true", "false"], default="true", tooltip=(
                            "Remove small isolated mesh fragments, using PyTMesh's own built-in "
                            "small-component heuristic (a DIFFERENT algorithm from pymeshlab's "
                            "remove_small_components_pct, and there's no user-tunable percentage/"
                            "threshold exposed here -- pymeshfix decides internally what counts as "
                            "'small' relative to the rest of the mesh, you don't get a dial for it).\n\n"
                            "DECISION: leave ON if you have obvious debris/floating crumbs and don't "
                            "need fine control over the threshold. Because the heuristic differs "
                            "from pymeshlab's, this backend can end up keeping or dropping DIFFERENT "
                            "components than pymeshlab would on the exact same mesh -- if this "
                            "doesn't remove debris you can clearly see, don't just re-run it hoping "
                            "for a different result; switch to the pymeshlab backend's "
                            "remove_small_components_pct instead, where you can actually tune the "
                            "aggressiveness.")),
                        io.Combo.Input("join_components", options=["true", "false"], default="false", tooltip=(
                            "Attempt to STITCH separate nearby components together into one "
                            "connected mesh by adding bridging geometry between them, rather than "
                            "deleting anything -- the opposite intent from remove_small_components "
                            "above.\n\n"
                            "DECISION: only turn this ON if you've specifically confirmed the extra "
                            "pieces are meant to be the SAME object that got split apart (e.g. a "
                            "thin neck or bridge that scanned/exported as two near-touching-but-not-"
                            "quite surfaces) -- use remove_small_components instead when the extra "
                            "pieces are just debris/noise you want gone. Leave OFF (the default) if "
                            "you're not sure which situation you're in: accidentally welding together "
                            "parts that were meant to stay separate is a much harder mistake to "
                            "notice and undo afterward than leaving them apart, whereas leaving real "
                            "debris in place is at least visually obvious and easy to fix later.")),
                        io.Combo.Input("clean_mesh", options=["true", "false"], default="true", tooltip=(
                            "PyTMesh's own clean() pass -- unique among every option in this whole "
                            "node in that it targets BOTH self-intersecting geometry AND degenerate "
                            "faces together, iteratively, in one combined loop (not as separate one-"
                            "shot passes like the other backends' toggles). This is a DIFFERENT "
                            "algorithm from the dedicated 'Fix Self Intersections' dispatcher (which "
                            "uses CGAL-based detection/removal/perturbation/remeshing) -- think of it "
                            "as a complementary general-purpose pass, not a replacement for that node.\n\n"
                            "DECISION: leave ON as a good default catch-all, especially when you're "
                            "not sure exactly what's wrong with a messy/tangled mesh and just want "
                            "one pass that tightens things up. If it doesn't fully resolve visible "
                            "self-intersections, don't just crank clean_iterations indefinitely -- go "
                            "use the dedicated Fix Self Intersections node instead, which has "
                            "purpose-built strategies for that specific problem.")),
                        io.Int.Input("clean_iterations", default=10, min=1, max=100, step=1, tooltip=(
                            "Maximum OUTER iterations for the clean_mesh pass above (PyTMesh."
                            "clean's max_iters). Each outer iteration re-detects and re-fixes "
                            "self-intersections/degeneracies that the PREVIOUS iteration's own fixes "
                            "may have just introduced -- cleaning up one tangled region can sometimes "
                            "create a new, smaller problem right next to it, so this loop keeps going "
                            "until it converges or hits this cap.\n\n"
                            "HOW TO PICK A VALUE: 1 iteration is USUALLY NOT ENOUGH for a genuinely "
                            "messy mesh -- it fixes what it sees in a single pass but leaves any "
                            "knock-on problems its own fix just created. The default, 10, is a "
                            "reasonable middle ground for typical real-world messy meshes and is "
                            "plenty for most cases. If the info output afterward STILL reports "
                            "unresolved intersections or degeneracies, don't creep up by 1 at a time "
                            "-- jump straight to 20-30 and see if it converges there. If it's still "
                            "not resolved by somewhere around 30-50, more iterations are unlikely to "
                            "help at all -- that's a sign to switch to a different backend or the "
                            "dedicated Fix Self Intersections node instead of continuing to raise "
                            "this. Cranking it to the max (100) 'just in case' mostly just burns "
                            "runtime for little to no additional benefit beyond that point, since "
                            "each iteration re-scans the whole mesh.")),
                        io.Int.Input("inner_loops", default=3, min=1, max=10, step=1, tooltip=(
                            "Inner sub-loop count PER outer clean_iterations iteration (PyTMesh."
                            "clean's inner_loops) -- a finer-grained convergence pass that tries to "
                            "fully resolve the specific trouble spots found in ONE outer iteration "
                            "before moving to the next. Think of clean_iterations as 'how many times "
                            "we re-scan the whole mesh' and inner_loops as 'how hard we grind on the "
                            "problems found in one scan.'\n\n"
                            "HOW TO PICK A VALUE: the default, 3, is enough for nearly every mesh and "
                            "is rarely worth changing either direction. If problems remain after a "
                            "run, raise clean_iterations (above) first, not this -- the outer loop is "
                            "almost always the more effective knob for actually resolving MORE "
                            "distinct problem regions, while this one just polishes harder on the "
                            "ones already found in a given pass. Only consider raising this if you've "
                            "already pushed clean_iterations up and specific, isolated trouble spots "
                            "are still not fully resolving.")),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="repaired_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, backend):
        from comfy_execution.graph_utils import GraphBuilder

        if cls.SCHEMA is None:
            cls.GET_SCHEMA()

        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]

        log.info("Mesh Repair dispatch: %s -> %s", selected, node_id)

        kwargs = {"mesh": mesh}
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


NODE_CLASS_MAPPINGS = {"GeomPackMeshRepair": MeshRepairNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackMeshRepair": "Mesh Repair"}
