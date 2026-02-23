# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Tetrahedralize a closed surface mesh using TetGen.

Uses TetGen as primary with quality options; on failure retries with
permissive switches (TetGen as backup).
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import trimesh


@dataclass
class TetMesh:
    """Tetrahedral mesh: vertices (n, 3) and cells (m, 4) indices."""

    vertices: np.ndarray
    cells: np.ndarray

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.cells = np.asarray(self.cells, dtype=np.int32)


def _trimesh_to_pyvista(mesh: trimesh.Trimesh):
    """Convert trimesh.Trimesh to pyvista.PolyData."""
    import pyvista as pv

    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)
    faces_pv = np.column_stack([np.full(len(faces), 3), faces])
    return pv.PolyData(vertices, faces_pv)


def _pyvista_surface_to_trimesh(pv_mesh) -> trimesh.Trimesh:
    """Convert pyvista.PolyData (surface) to trimesh.Trimesh."""
    vertices = np.array(pv_mesh.points)
    faces = []
    if getattr(pv_mesh, "n_faces", 0) > 0:
        faces_flat = np.array(pv_mesh.faces)
        i = 0
        while i < len(faces_flat):
            n = int(faces_flat[i])
            if n == 3:
                faces.append(faces_flat[i + 1 : i + 4].tolist())
            elif n == 4:
                faces.append([faces_flat[i + 1], faces_flat[i + 2], faces_flat[i + 3]])
                faces.append([faces_flat[i + 1], faces_flat[i + 3], faces_flat[i + 4]])
            i += n + 1
    if faces:
        faces = np.array(faces, dtype=np.int32)
    else:
        faces = np.zeros((0, 3), dtype=np.int32)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _run_tetgen(
    pv_mesh,
    switches: str,
) -> Tuple[Optional[Any], Optional[TetMesh], Optional[trimesh.Trimesh], str]:
    """
    Run TetGen with given switches. Returns (tgen, tet_mesh, surface_trimesh, error_msg).
    On success error_msg is empty.
    """
    try:
        import tetgen
        import pyvista as pv
    except ImportError as e:
        return None, None, None, f"TetGen required: pip install tetgen. {e}"

    try:
        tgen = tetgen.TetGen(pv_mesh)
        tgen.tetrahedralize(switches=switches)
    except Exception as e:
        return None, None, None, str(e)

    grid = tgen.grid
    if grid is None or grid.n_points == 0:
        return None, None, None, "TetGen produced empty grid"

    nodes = np.array(grid.points, dtype=np.float64)
    elem = getattr(tgen, "elem", None)
    if elem is None or len(elem) == 0:
        return None, None, None, "TetGen produced no tetrahedra"

    elem = np.asarray(elem, dtype=np.int32)
    if elem.ndim == 1:
        elem = elem.reshape(-1, 4)
    if elem.shape[1] != 4:
        return None, None, None, f"Unexpected elem shape {elem.shape}"

    tet_mesh = TetMesh(vertices=nodes, cells=elem)

    try:
        surface_pv = grid.extract_surface()
        if surface_pv.n_faces > 0:
            surface_trimesh = _pyvista_surface_to_trimesh(surface_pv)
        else:
            surface_trimesh = trimesh.Trimesh(
                vertices=nodes, faces=np.zeros((0, 3), dtype=np.int32), process=False
            )
    except Exception:
        surface_trimesh = trimesh.Trimesh(
            vertices=nodes, faces=np.zeros((0, 3), dtype=np.int32), process=False
        )

    return tgen, tet_mesh, surface_trimesh, ""


class TetrahedralizeNode:
    """
    Tetrahedralize the interior of a closed surface mesh using TetGen.

    Primary run uses quality options; on failure a backup run uses
    permissive switches so difficult meshes can still be tetrahedralized.
    """

    @classmethod
    def INPUT_TYPES(cls):
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
                    "tooltip": "Relative scale for max tet volume / quality",
                }),
                "optimize": (["true", "false"], {"default": "true"}),
                "simplify": (["true", "false"], {"default": "true"}),
            },
        }

    RETURN_TYPES = ("TETMESH", "TRIMESH", "STRING")
    RETURN_NAMES = ("tet_mesh", "surface", "info")
    FUNCTION = "tetrahedralize"
    CATEGORY = "geompack/conversion"

    def tetrahedralize(
        self,
        input_mesh,
        edge_length_fac: float = 0.05,
        optimize: str = "true",
        simplify: str = "true",
    ):
        """
        Run TetGen: primary with quality, backup with permissive switches.
        """
        # Validate input
        if not hasattr(input_mesh, "vertices") or not hasattr(input_mesh, "faces"):
            raise ValueError("Tetrahedralize requires a TRIMESH with vertices and faces")
        verts = np.asarray(input_mesh.vertices)
        faces = np.asarray(input_mesh.faces)
        if len(verts) < 4:
            raise ValueError("Mesh must have at least 4 vertices")
        if len(faces) < 4:
            raise ValueError("Mesh must have at least 4 faces")
        if faces.shape[1] != 3:
            raise ValueError("Faces must be triangles (shape (n, 3))")

        try:
            pv_mesh = _trimesh_to_pyvista(input_mesh)
        except ImportError as e:
            raise ImportError(
                "Tetrahedralize requires pyvista. Install with: pip install pyvista"
            ) from e

        # Primary switches: quality (radius-edge ratio), optional optimization
        quality = max(1.01, 1.0 + float(edge_length_fac) * 2.0)
        primary_switches = "pq" + f"{quality:.2f}"
        if optimize == "true":
            primary_switches += "O"

        tgen, tet_mesh, surface_trimesh, err = _run_tetgen(pv_mesh, primary_switches)

        if err:
            print(f"[Tetrahedralize] Primary TetGen failed: {err}, trying backup switches")
            backup_switches = "p"
            tgen, tet_mesh, surface_trimesh, backup_err = _run_tetgen(
                pv_mesh, backup_switches
            )
            if backup_err:
                raise RuntimeError(
                    f"Tetrahedralize failed. Primary: {err}. Backup: {backup_err}"
                )
            used_switches = backup_switches
            run_type = "backup"
        else:
            used_switches = primary_switches
            run_type = "primary"

        n_verts = tet_mesh.vertices.shape[0]
        n_tets = tet_mesh.cells.shape[0]
        n_surf = surface_trimesh.faces.shape[0]
        info = (
            f"Tetrahedralize ({run_type}, switches={used_switches}): "
            f"{n_verts} vertices, {n_tets} tets, surface {n_surf} faces."
        )
        print(f"[Tetrahedralize] {info}")

        return (tet_mesh, surface_trimesh, info)


NODE_CLASS_MAPPINGS = {
    "GeomPackTetrahedralize": TetrahedralizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackTetrahedralize": "Tetrahedralize",
}
