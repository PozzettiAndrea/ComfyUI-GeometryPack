# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Alpha Wrap Node - Shrink wrap mesh generation using CGAL's Alpha Wrap algorithm.

Creates a watertight mesh that tightly wraps around input geometry.
Useful for non-manifold meshes or polygon soups.
"""

import logging
import os
import threading
import time

import numpy as np
import trimesh
import comfy.utils
from comfy_api.latest import io

from CGAL import CGAL_Alpha_wrap_3
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3

log = logging.getLogger("geometrypack")

# Progress bar layout (percentage of total)
_PHASE_BUILD = 15       # 0-15%: building CGAL input
_PHASE_WRAP_START = 15  # 15%: wrap starts
_PHASE_WRAP_END = 85    # ..85%: wrap ends (time-based fill)
_PHASE_EXTRACT = 100    # 85-100%: extracting result


def _get_rss_mb():
    """Get current process RSS in MB (Linux)."""
    try:
        with open(f"/proc/{os.getpid()}/statm", "r") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return 0.0


def _polyhedron_to_arrays(poly):
    """Extract vertices and faces from a CGAL Polyhedron_3."""
    verts = []
    vertex_map = {}
    for i, vertex in enumerate(poly.vertices()):
        point = vertex.point()
        verts.append([float(point.x()), float(point.y()), float(point.z())])
        vertex_map[vertex] = i

    faces = []
    for facet in poly.facets():
        halfedge = facet.halfedge()
        face_verts = []
        start_he = halfedge
        current = start_he
        while True:
            face_verts.append(vertex_map[current.vertex()])
            current = current.next()
            if current == start_he:
                break
        if len(face_verts) == 3:
            faces.append(face_verts)

    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


class AlphaWrapNode(io.ComfyNode):
    """
    Alpha Wrap - Generate a watertight shrink-wrapped mesh.

    Uses CGAL's Alpha Wrap algorithm to create a tight-fitting
    watertight mesh around input geometry.

    Works with:
    - Non-manifold meshes
    - Meshes with holes
    - Polygon soups (overlapping/intersecting geometry)

    Parameters:
    - alpha: Controls wrap tightness. Smaller = tighter wrap, more detail.
             Relative to bounding box diagonal (0.01 = 1% of bbox diagonal).
    - offset: Surface offset distance. Smaller = closer to original surface.
              Relative to bounding box diagonal.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackAlphaWrap",
            display_name="Alpha Wrap (Shrink Wrap)",
            category="geompack/reconstruction",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("input_mesh"),
                io.Float.Input("alpha_percent", default=2.0, min=0.001, max=50.0, step=0.1, tooltip="Wrap tightness as % of bounding box diagonal. Smaller = tighter wrap with more detail, but MUCH slower (runtime ~ 1/alpha\u00b3). Start at 2-5% and tighten if needed.", optional=True),
                io.Float.Input("offset_percent", default=2.0, min=0.01, max=10.0, step=0.1, tooltip="Surface offset as % of bounding box diagonal. Smaller = closer to original surface.", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="wrapped_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, input_mesh, alpha_percent=2.0, offset_percent=2.0):
        vertices = np.asarray(input_mesh.vertices, dtype=np.float64)
        input_vertex_count = len(vertices)

        is_point_cloud = (
            not hasattr(input_mesh, 'faces') or
            input_mesh.faces is None or
            len(input_mesh.faces) == 0
        )

        if is_point_cloud:
            raise ValueError(
                "Alpha Wrap requires a mesh with faces (triangle soup), not a point cloud.\n"
                "For point clouds, first use 'Reconstruct Surface' node (e.g., Poisson or Ball Pivoting) "
                "to create a mesh, then apply Alpha Wrap."
            )

        input_face_count = len(input_mesh.faces)
        input_type = "mesh"

        # Compute bounding box diagonal for relative parameters
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)

        # Convert percentages to absolute values
        alpha = (alpha_percent / 100.0) * bbox_diagonal
        offset = (offset_percent / 100.0) * bbox_diagonal

        log.info("Input: %s vertices, %s faces (%s)", f"{input_vertex_count:,}", f"{input_face_count:,}", input_type)
        log.info("Bounding box diagonal: %.4f", bbox_diagonal)
        log.info("Alpha: %.6f (%s%% of bbox)", alpha, alpha_percent)
        log.info("Offset: %.6f (%s%% of bbox)", offset, offset_percent)

        pbar = comfy.utils.ProgressBar(100)

        # --- Phase 1: Build CGAL input data (0-15%) ---
        t0 = time.monotonic()
        cgal_points = CGAL_Alpha_wrap_3.Point_3_Vector()
        cgal_points.reserve(input_vertex_count)
        report_every = max(1, input_vertex_count // 10)
        for i, v in enumerate(vertices):
            cgal_points.append(Point_3(float(v[0]), float(v[1]), float(v[2])))
            if i % report_every == 0:
                pbar.update_absolute(int(_PHASE_BUILD * i / input_vertex_count))

        cgal_faces = [[int(idx) for idx in face] for face in input_mesh.faces]
        pbar.update_absolute(_PHASE_BUILD)
        log.info("Built CGAL input data in %.1fs", time.monotonic() - t0)

        # --- Phase 2: Run alpha wrap (15-85%) ---
        output_poly = Polyhedron_3()
        exc_holder = [None]
        rss_before = _get_rss_mb()

        def _run():
            try:
                CGAL_Alpha_wrap_3.alpha_wrap_3(cgal_points, cgal_faces, alpha, offset, output_poly)
            except Exception as e:
                exc_holder[0] = e

        log.info("Running CGAL Alpha Wrap...")
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        wrap_start = time.monotonic()
        interval = 5
        wrap_range = _PHASE_WRAP_END - _PHASE_WRAP_START
        while thread.is_alive():
            thread.join(timeout=interval)
            elapsed = time.monotonic() - wrap_start
            rss = _get_rss_mb()
            delta = rss - rss_before
            if thread.is_alive():
                # Asymptotic progress: approaches _PHASE_WRAP_END but never reaches it
                # Uses 1 - 1/(1 + t/30) so it fills ~50% at 30s, ~75% at 90s, etc.
                frac = 1.0 - 1.0 / (1.0 + elapsed / 30.0)
                progress = int(_PHASE_WRAP_START + wrap_range * frac)
                pbar.update_absolute(progress)
                log.info("Alpha Wrap running... %.0fs elapsed, RSS %.0f MB (+%.0f MB)",
                         elapsed, rss, delta)

        elapsed_total = time.monotonic() - wrap_start
        rss_final = _get_rss_mb()
        pbar.update_absolute(_PHASE_WRAP_END)
        log.info("Alpha Wrap finished in %.1fs, RSS %.0f MB (+%.0f MB)",
                 elapsed_total, rss_final, rss_final - rss_before)

        if exc_holder[0] is not None:
            raise exc_holder[0]

        # --- Phase 3: Extract result mesh (85-100%) ---
        t0 = time.monotonic()
        result_vertices, result_faces = _polyhedron_to_arrays(output_poly)
        pbar.update_absolute(_PHASE_EXTRACT)
        log.info("Extracted result mesh in %.1fs", time.monotonic() - t0)

        result_mesh = trimesh.Trimesh(
            vertices=result_vertices,
            faces=result_faces,
            process=False
        )

        # Copy metadata if present
        if hasattr(input_mesh, 'metadata') and input_mesh.metadata:
            result_mesh.metadata = input_mesh.metadata.copy()
        else:
            result_mesh.metadata = {}

        result_mesh.metadata['alpha_wrap'] = {
            'alpha': alpha,
            'alpha_percent': alpha_percent,
            'offset': offset,
            'offset_percent': offset_percent,
            'bbox_diagonal': bbox_diagonal,
            'input_type': input_type
        }

        output_vertex_count = len(result_mesh.vertices)
        output_face_count = len(result_mesh.faces)
        is_watertight = result_mesh.is_watertight

        log.info("Result: %s vertices, %s faces", f"{output_vertex_count:,}", f"{output_face_count:,}")
        log.info("Watertight: %s", is_watertight)

        report = f"""Alpha Wrap Report
{'='*40}

Input:
  Type: {input_type}
  Vertices: {input_vertex_count:,}
  Faces: {input_face_count:,}

Parameters:
  Alpha: {alpha:.6f} ({alpha_percent}% of bbox diagonal)
  Offset: {offset:.6f} ({offset_percent}% of bbox diagonal)
  BBox Diagonal: {bbox_diagonal:.4f}

Output:
  Vertices: {output_vertex_count:,}
  Faces: {output_face_count:,}
  Watertight: {'Yes' if is_watertight else 'No'}

Tips:
  - Decrease alpha_percent for tighter wrap / more detail
  - Decrease offset_percent to get closer to original surface
  - Lower values = slower computation
"""

        return io.NodeOutput(result_mesh, report, ui={"text": [report]})


NODE_CLASS_MAPPINGS = {
    "GeomPackAlphaWrap": AlphaWrapNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackAlphaWrap": "Alpha Wrap (Shrink Wrap)",
}
