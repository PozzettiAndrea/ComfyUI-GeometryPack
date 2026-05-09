# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Shared helpers for skeleton backend nodes."""

import numpy as np


def normalize_skeleton(vertices):
    """
    Normalize skeleton vertices to [-1, 1] range.

    Args:
        vertices: Array of shape [N, 3]

    Returns:
        Normalized vertices in [-1, 1] range
    """
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (min_coords + max_coords) / 2
    vertices = vertices - center
    scale = (max_coords - min_coords).max() / 2
    if scale > 0:
        vertices = vertices / scale
    return vertices
