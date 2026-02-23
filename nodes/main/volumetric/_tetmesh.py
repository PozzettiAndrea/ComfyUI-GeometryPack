# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
TETMESH type for tetrahedral volume meshes.

Used by Tetrahedralize (TetWild) and future Save Tet Mesh / FEM export nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TetMesh:
    """
    Tetrahedral volume mesh: vertices and cell connectivity.

    Attributes:
        vertices: (N, 3) float64 array of vertex positions.
        cells: (M, 4) int32 array of tetrahedron vertex indices (each row is one tet).
    """

    vertices: np.ndarray
    cells: np.ndarray

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be (N, 3)")
        if self.cells.ndim != 2 or self.cells.shape[1] != 4:
            raise ValueError("cells must be (M, 4)")
