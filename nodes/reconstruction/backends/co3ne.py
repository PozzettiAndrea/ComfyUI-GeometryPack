# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""CO3NE (CoCone-family) surface reconstruction backend node.

Unlike the Poisson backend (an implicit-function / energy-minimization fit that
always produces a smooth, watertight result), CO3NE is a Delaunay/Voronoi-driven
combinatorial reconstruction in the CoCone family (Amenta/Choi/Kolluri) -- it
builds the surface directly out of the point cloud's own local Delaunay/Voronoi
structure, with sampling-theoretic guarantees on well-sampled input, rather than
fitting a smooth implicit function to it. Genuinely different algorithm family
from every other backend on this dispatcher.
"""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class ReconstructCo3neNode(io.ComfyNode):
    """CO3NE (CoCone-family) surface reconstruction from a point cloud."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackReconstruct_Co3ne",
            display_name="Reconstruct CO3NE (backend)",
            category="geompack/reconstruction",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("points"),
                io.Int.Input("nb_neighbors", default=30, min=3, max=500, step=1, tooltip=(
                    "Number of nearest neighbors used to estimate each point's local surface "
                    "structure (tangent plane / local Delaunay neighborhood), before the "
                    "combinatorial CoCone reconstruction runs. Too FEW neighbors and the local "
                    "surface estimate becomes noisy/unreliable on sparse or noisy point clouds -- "
                    "you'll see spurious holes or incorrectly connected regions. Too MANY "
                    "neighbors over-smooths the local estimate (can blur genuinely sharp features "
                    "or thin parts) and costs more time per point. The default, 30, is a reasonable "
                    "starting point for moderately dense scans; drop toward 15-20 for very dense, "
                    "clean point clouds where local neighborhoods are already tight and reliable, "
                    "or raise toward 50-100 for sparse or noisy scans where you need to average "
                    "over more neighbors to get a trustworthy local estimate.")),
                io.Int.Input("nb_iterations", default=3, min=1, max=20, step=1, tooltip=(
                    "Number of refinement passes the reconstruction runs to clean up and complete "
                    "the initial CoCone surface (closing small gaps, resolving locally ambiguous "
                    "regions the first pass couldn't fully determine). 1 iteration is often not "
                    "enough on real-world (imperfectly/non-uniformly sampled) scans -- you'll see "
                    "leftover small holes or rough patches. The default, 3, is enough for most "
                    "reasonably-sampled point clouds. If the output still has visible gaps after a "
                    "run, try 5-8 before assuming the point cloud itself is too sparse in that "
                    "region for ANY reconstruction to bridge -- going much beyond ~10 rarely "
                    "resolves gaps that persisted through the first several passes, since those are "
                    "usually genuine sampling gaps rather than something more refinement can fix.")),
                io.Float.Input("radius", default=5.0, min=0.001, max=1000.0, step=0.1, display_mode="number", tooltip=(
                    "Search radius for the neighbor queries above -- an ABSOLUTE distance in the "
                    "point cloud's own coordinate units, NOT automatically scaled to your point "
                    "cloud's size (unlike this dispatcher's ball_pivoting backend, which has a "
                    "0=auto option; this backend has no auto mode). The library default, 5.0, "
                    "assumes a point cloud roughly on the order of tens of units across -- for a "
                    "point cloud normalized to a unit bounding box, or one in millimeters with "
                    "coordinates in the thousands, 5.0 will almost certainly be badly wrong "
                    "(either finding far too few or far too many neighbors within the radius). "
                    "ALWAYS set this relative to your point cloud's own average point spacing "
                    "before running -- as a starting point, try roughly 3-5x the typical distance "
                    "between neighboring points, then adjust based on whether nb_neighbors above "
                    "is finding a sensible number of points within that radius.")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="reconstructed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, points, nb_neighbors=30, nb_iterations=3, radius=5.0):
        import pygeogram

        log.info("Backend: co3ne")
        vertices = np.ascontiguousarray(points.vertices, dtype=np.float64)
        log.info("Input: %d points | nb_neighbors=%s, nb_iterations=%s, radius=%s",
                 len(vertices), nb_neighbors, nb_iterations, radius)

        V_out, F_out = pygeogram.co3ne_reconstruct(
            vertices, nb_neighbors=int(nb_neighbors), nb_iterations=int(nb_iterations),
            radius=float(radius),
        )

        result = trimesh_module.Trimesh(vertices=V_out, faces=F_out, process=False)

        if hasattr(points, 'metadata') and points.metadata:
            result.metadata = points.metadata.copy()
        else:
            result.metadata = {}
        result.metadata['reconstruction'] = {
            'method': 'co3ne',
            'input_points': len(vertices),
            'output_vertices': len(result.vertices),
            'output_faces': len(result.faces),
            'nb_neighbors': nb_neighbors,
            'nb_iterations': nb_iterations,
            'radius': radius,
        }

        info = f"""Reconstruct Surface Results (CO3NE):

Engine: Geogram (CoCone family)
nb_neighbors: {nb_neighbors}
nb_iterations: {nb_iterations}
radius: {radius}

Input Points: {len(vertices):,}
Output Vertices: {len(result.vertices):,}
Output Faces: {len(result.faces):,}

Watertight: {result.is_watertight}

CO3NE is a Delaunay/Voronoi-driven combinatorial reconstruction (distinct from
Poisson's smooth implicit-function fit) -- best on well-sampled point clouds.
"""
        log.info("Output: %d vertices, %d faces", len(result.vertices), len(result.faces))
        return io.NodeOutput(result, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackReconstruct_Co3ne": ReconstructCo3neNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackReconstruct_Co3ne": "Reconstruct CO3NE (backend)"}
