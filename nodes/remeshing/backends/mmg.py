# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""MMG adaptive surface remeshing backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshMMGNode(io.ComfyNode):
    """MMG curvature-adaptive surface remeshing backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh_MMG",
            display_name="Remesh MMG (backend)",
            category="geompack/remeshing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("hausd", default=0.01, min=0.0001, max=10.0, step=0.0001, display_mode="number", tooltip=(
                    "Hausdorff distance -- THE adaptive driver: the max geometric deviation the "
                    "remeshed surface may have from the original. MMG sizes triangles so this bound "
                    "holds everywhere, which is WHY it's curvature-adaptive: to stay within hausd of "
                    "a curved patch you need small triangles, while flats can be huge. SMALL hausd = "
                    "hug the surface (fine on curves, more faces, higher fidelity); LARGE = drift "
                    "(coarser, fewer faces).\n"
                    "ABSOLUTE world units (NOT normalised). MMG's CLI default is ~0.01 x "
                    "bbox-diagonal, so on a 400-unit part that's ~4.0 -- the literal 0.01 here is "
                    "~400x finer and will explode the face count. Set relative to mesh scale "
                    "(~bbox_diag * 0.005-0.02).")),
                io.Float.Input("hmin", default=0.0, min=0.0, max=10.0, step=0.001, display_mode="number", tooltip=(
                    "Minimum edge length CLAMP (world units): triangles won't shrink below this even "
                    "where curvature/hausd wants them finer. 0 = auto (MMG picks from bbox). Use to "
                    "cap face count in highly-curved regions.")),
                io.Float.Input("hmax", default=0.0, min=0.0, max=100.0, step=0.01, display_mode="number", tooltip=(
                    "Maximum edge length CLAMP (world units): triangles won't grow beyond this even "
                    "on big flats. 0 = auto. Use to force a minimum density on flat faces (otherwise "
                    "MMG makes them very coarse).")),
                io.Float.Input("hgrad", default=1.3, min=1.0, max=5.0, step=0.1, display_mode="number", tooltip=(
                    "Gradation -- max allowed size RATIO between adjacent edges, i.e. how fast "
                    "triangle size may change across the mesh. 1.0 = no growth (near-uniform, many "
                    "faces); 1.3 = smooth transitions (default); 2-3 = abrupt jumps allowed (fewer "
                    "faces, raggeder density). Lower = smoother size field, more triangles.")),
                io.Float.Input("ar", default=-1.0, min=-1.0, max=180.0, step=1.0, display_mode="number", optional=True, tooltip=(
                    "RIDGE (feature-edge) detection angle, in DEGREES. MMG marks an edge as a sharp "
                    "'ridge' and PRESERVES it through remeshing when the dihedral between its two "
                    "faces exceeds this. LOWER = more edges kept as features (30 preserves shallow "
                    "chamfers/creases); HIGHER = only very sharp edges survive (milder ones get "
                    "remeshed/smoothed across). -1 = MMG default (~45 deg).\n"
                    "For CAD set ~30-40 to protect creases/chamfers; raise toward 60-90 to let soft "
                    "creases reflow. This is the feature-preservation control plain isotropic lacks.")),
                io.Float.Input("hsiz", default=0.0, min=0.0, max=100.0, step=0.001, display_mode="number", optional=True, tooltip=(
                    "Constant target edge size = UNIFORM mode. 0 = OFF (use adaptive hausd sizing, "
                    "MMG's whole point). Set >0 to OVERRIDE hmin/hmax/hausd-adaptivity and make an "
                    "even ISOTROPIC mesh of this edge length (world units) -- turns MMG into a "
                    "feature-preserving isotropic remesher. Use when you want UNIFORM density (e.g. "
                    "before L0, which is consistent only on uniform meshes) rather than "
                    "curvature-adaptive. Absolute length -- set relative to mesh scale.")),
                io.Combo.Input("optim", options=["false", "true"], default="false", optional=True, tooltip=(
                    "Optimization mode. true = improve triangle QUALITY (angles/shape) of the "
                    "EXISTING mesh while keeping sizes roughly as-is, few/no insertions -- a gentle "
                    "clean-up, not a re-size. Use to fix sliver quality without changing resolution. "
                    "Pair with noinsert to forbid adding points.")),
                io.Combo.Input("nreg", options=["false", "true"], default="false", optional=True, tooltip=(
                    "Normal regularization. true = smooth the per-vertex normals MMG uses to curve "
                    "new triangles onto the surface -- reduces faceting/oscillation on NOISY input "
                    "(scan / Tripo / marching-cubes) at the cost of slightly softening sharp "
                    "transitions. Helps image-derived meshes; leave off for clean CAD.")),
                io.Combo.Input("anisosize", options=["false", "true"], default="false", optional=True, tooltip=(
                    "ANISOTROPIC sizing: allow long, thin triangles ALIGNED to curvature (stretched "
                    "along low-curvature directions, refined across high) -- far fewer triangles for "
                    "the same fidelity on cylinders/developable CAD surfaces.\n"
                    "CAVEAT: MMG's anisotropic mode needs a per-vertex TENSOR METRIC field, which "
                    "this node does NOT currently supply. Enabling it alone will likely no-op or "
                    "error -- leave OFF until a metric-field input is wired. (Isotropic "
                    "hausd-adaptive is the default and works standalone.)")),
                io.Combo.Input("noinsert", options=["false", "true"], default="false", optional=True, tooltip=(
                    "Disable point INSERTION (no edge splits): MMG may only collapse/swap/move "
                    "existing vertices, never add new ones. Use to forbid increasing resolution "
                    "(pure cleanup / decimation). Pairs with optim.")),
                io.Combo.Input("noswap", options=["false", "true"], default="false", optional=True, tooltip=(
                    "Disable edge SWAP (flip). Off by default -- swaps are how MMG Delaunay-izes and "
                    "removes caps/large angles. Enable only to preserve input connectivity exactly "
                    "or to debug; it will hurt triangle quality.")),
                io.Combo.Input("nomove", options=["false", "true"], default="false", optional=True, tooltip=(
                    "Disable point RELOCATION (tangential smoothing). Off by default. Enable to keep "
                    "vertices exactly where they are (only insert/collapse/swap), avoiding any "
                    "tangential drift of existing points.")),
                io.Combo.Input("keep_ref", options=["false", "true"], default="false", optional=True, tooltip=(
                    "Keep edge REFERENCES in the output (forwarded as keepRef): preserves MMG's "
                    "internal edge/region reference tags (marked ridges/boundaries) on the result. "
                    "Relevant if you round-trip through .sol/.mesh or consume ridge tags downstream; "
                    "harmless to leave off.")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, hausd=0.01, hmin=0.0, hmax=0.0, hgrad=1.3, ar=-1.0, hsiz=0.0,
                optim="false", nreg="false", anisosize="false", noinsert="false",
                noswap="false", nomove="false", keep_ref="false"):
        try:
            import mmgpy
            # conda-forge mmgpy's __init__ omits Mesh from public exports; the class
            # itself is present at mmgpy._mesh.Mesh on both pypi and conda builds.
            from mmgpy._mesh import Mesh as _MmgMesh
        except ImportError:
            log.warning("mmgpy not available on Windows — returning input mesh unchanged")
            info = "Remesh (MMG Adaptive): skipped — mmgpy not available on this platform"
            return io.NodeOutput(trimesh, info, ui={"text": [info]})

        log.info("Backend: mmg_adaptive")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))

        vertices = np.array(trimesh.vertices, dtype=np.float64)
        faces = np.array(trimesh.faces, dtype=np.int32)
        mmg_mesh = _MmgMesh(vertices, faces)

        opts_kwargs = {"hausd": hausd, "hgrad": hgrad, "verbose": -1}
        if float(hsiz) > 0:
            opts_kwargs["hsiz"] = float(hsiz)        # uniform mode -- overrides hmin/hmax
        else:
            if hmin > 0:
                opts_kwargs["hmin"] = hmin
            if hmax > 0:
                opts_kwargs["hmax"] = hmax
        if float(ar) >= 0:
            opts_kwargs["ar"] = float(ar)
        for name, val in (("optim", optim), ("nreg", nreg), ("anisosize", anisosize),
                          ("noinsert", noinsert), ("noswap", noswap), ("nomove", nomove),
                          ("keep_ref", keep_ref)):
            if val == "true":
                opts_kwargs[name] = True
        log.info("Parameters: %s", opts_kwargs)

        opts = mmgpy.MmgSOptions(**opts_kwargs)

        log.info("Running mmgs surface remeshing...")
        result = mmg_mesh.remesh(opts)

        if not result.success:
            raise ValueError(f"MMG remeshing failed (return code {result.return_code})")

        out_vertices = mmg_mesh.get_vertices()
        out_faces = mmg_mesh.get_triangles()

        remeshed_mesh = trimesh_module.Trimesh(vertices=out_vertices, faces=out_faces, process=False)
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'mmg_adaptive',
            'hausd': hausd, 'hmin': hmin, 'hmax': hmax, 'hgrad': hgrad,
        }

        log.info("Output: %d vertices, %d faces", len(remeshed_mesh.vertices), len(remeshed_mesh.faces))

        info = (f"Remesh (MMG Adaptive): "
                f"{len(trimesh.vertices):,}v/{len(trimesh.faces):,}f -> "
                f"{len(remeshed_mesh.vertices):,}v/{len(remeshed_mesh.faces):,}f | "
                f"hausd={hausd}")

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackRemesh_MMG": RemeshMMGNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRemesh_MMG": "Remesh MMG (backend)"}
