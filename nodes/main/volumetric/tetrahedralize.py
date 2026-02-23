# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Tetrahedralize Node - Convert a watertight surface mesh to a tetrahedral volume mesh.

Uses pytetwild (fTetWild). Best results with watertight input; chain MeshFix first if needed.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import trimesh

from ._tetmesh import TetMesh


def _boundary_surface_from_tets(
    vertices: np.ndarray, cells: np.ndarray
) -> np.ndarray:
    """
    Extract boundary triangular faces from a tetrahedral mesh.

    A face is on the boundary iff it belongs to exactly one tetrahedron.
    """
    face_count: dict[tuple[int, int, int], int] = defaultdict(int)
    for i in range(cells.shape[0]):
        a, b, c, d = cells[i, 0], cells[i, 1], cells[i, 2], cells[i, 3]
        for tri in (
            (a, b, c),
            (a, b, d),
            (a, c, d),
            (b, c, d),
        ):
            key = tuple(sorted(tri))
            face_count[key] += 1
    boundary = np.array(
        [list(k) for k, count in face_count.items() if count == 1],
        dtype=np.int32,
    )
    return boundary


class TetrahedralizeNode:
    """
    Tetrahedralize a watertight surface mesh using fTetWild (pytetwild).

    Produces a tetrahedral volume mesh for FEM/simulation workflows.
    Run repair (e.g. MeshFix) first if the mesh has holes or self-intersections.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "input_mesh": ("TRIMESH",),
            },
            "optional": {
                "edge_length_fac": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Ideal edge length as fraction of bounding box diagonal (ignored if edge_length_abs > 0)",
                }),
                "edge_length_abs": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1e6,
                    "step": 0.001,
                    "tooltip": "Absolute ideal edge length; if > 0 overrides edge_length_fac",
                }),
                "optimize": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Improve cell quality (slower)",
                }),
                "simplify": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Simplify input surface before tetrahedralization",
                }),
                "epsilon": ("FLOAT", {
                    "default": 1e-3,
                    "min": 1e-6,
                    "max": 0.1,
                    "step": 1e-4,
                    "tooltip": "Envelope size (max distance of output surface from input, relative to bbox)",
                }),
                "stop_energy": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.5,
                    "tooltip": "Stop optimization when conformal AMIPS energy reaches this",
                }),
                "num_opt_iter": ("INT", {
                    "default": 80,
                    "min": 1,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Max optimization iterations when optimize=true",
                }),
                "coarsen": (["true", "false"], {
                    "default": "false",
                    "tooltip": "Coarsen output while keeping quality",
                }),
                "num_threads": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of threads (0 = all cores)",
                }),
                "quiet": (["true", "false"], {
                    "default": "false",
                    "tooltip": "Suppress pytetwild log output",
                }),
            },
        }

    RETURN_TYPES = ("TETMESH", "TRIMESH", "STRING")
    RETURN_NAMES = ("tet_mesh", "surface", "info")
    FUNCTION = "tetrahedralize"
    CATEGORY = "geompack/volumetric"
    OUTPUT_NODE = True

    def tetrahedralize(
        self,
        input_mesh: trimesh.Trimesh,
        edge_length_fac: float = 0.05,
        edge_length_abs: float = 0.0,
        optimize: str = "true",
        simplify: str = "true",
        epsilon: float = 1e-3,
        stop_energy: float = 10.0,
        num_opt_iter: int = 80,
        coarsen: str = "false",
        num_threads: int = 0,
        quiet: str = "false",
    ) -> dict:
        """
        Run fTetWild tetrahedralization and return tet mesh, boundary surface, and report.
        """
        try:
            import pytetwild
        except (ImportError, OSError) as e:
            raise ImportError(
                "pytetwild is required for tetrahedralization. "
                "Install with: pip install pytetwild"
            ) from e

        v = np.asarray(input_mesh.vertices, dtype=np.float64)
        f = np.asarray(input_mesh.faces, dtype=np.int32)
        if v.size == 0 or f.size == 0:
            raise ValueError("Input mesh has no vertices or faces")

        n_verts_in = len(v)
        n_faces_in = len(f)
        print(f"[Tetrahedralize] Input: {n_verts_in:,} vertices, {n_faces_in:,} faces")

        kwargs: dict = {
            "optimize": optimize == "true",
            "simplify": simplify == "true",
            "epsilon": epsilon,
            "stop_energy": stop_energy,
            "num_opt_iter": num_opt_iter,
            "coarsen": coarsen == "true",
            "num_threads": num_threads,
            "quiet": quiet == "true",
        }
        if edge_length_abs > 0:
            kwargs["edge_length_abs"] = float(edge_length_abs)
        else:
            kwargs["edge_length_fac"] = float(edge_length_fac)

        t0 = time.perf_counter()
        try:
            v_out, tetra = pytetwild.tetrahedralize(v, f, **kwargs)
        except Exception as e:
            raise RuntimeError(
                "Tetrahedralization failed. Mesh may be non-watertight or degenerate. "
                "Try running MeshFix (or other repair) first."
            ) from e
        elapsed = time.perf_counter() - t0

        v_out = np.asarray(v_out, dtype=np.float64)
        tetra = np.asarray(tetra, dtype=np.int32)
        if tetra.ndim == 1 or tetra.shape[0] == 0:
            raise RuntimeError("pytetwild returned no tetrahedra")

        if tetra.shape[1] != 4:
            tetra = np.reshape(tetra, (-1, 4))

        tet_mesh = TetMesh(vertices=v_out, cells=tetra)
        boundary_faces = _boundary_surface_from_tets(v_out, tetra)
        surface_mesh = trimesh.Trimesh(
            vertices=v_out,
            faces=boundary_faces,
            process=False,
        )
        if hasattr(input_mesh, "metadata") and input_mesh.metadata:
            surface_mesh.metadata = input_mesh.metadata.copy()

        n_verts_out = len(v_out)
        n_tets = len(tetra)
        n_boundary = len(boundary_faces)
        info = f"""Tetrahedralize (TetWild) Report
{'='*40}
Input:  {n_verts_in:,} vertices, {n_faces_in:,} faces
Output: {n_verts_out:,} vertices, {n_tets:,} tetrahedra, {n_boundary:,} boundary triangles
Time:   {elapsed:.2f}s
"""
        print(f"[Tetrahedralize] Result: {n_verts_out:,} vertices, {n_tets:,} tets in {elapsed:.2f}s")
        return {
            "ui": {"text": [info]},
            "result": (tet_mesh, surface_mesh, info),
        }


NODE_CLASS_MAPPINGS = {
    "GeomPackTetrahedralize": TetrahedralizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackTetrahedralize": "Tetrahedralize (TetWild)",
}
