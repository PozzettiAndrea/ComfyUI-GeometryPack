# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Scramble integer field values to maximize contrast between adjacent faces.
Uses graph coloring to ensure neighboring segments get visually distinct values.
"""

import numpy as np
import trimesh
from collections import defaultdict


class ScrambleIntField:
    """
    Reassigns integer field values to maximize contrast between adjacent faces.
    Useful for visualizing segmentation results where adjacent segments
    might have similar label numbers by chance.

    Uses greedy graph coloring on the segment adjacency graph.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "field_name": ("STRING", {
                    "default": "seg",
                    "tooltip": "Name of the integer face field to scramble."
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                    "tooltip": "Random seed for color assignment order."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "scramble"
    CATEGORY = "geometrypack/analysis"

    def scramble(
        self,
        mesh: trimesh.Trimesh,
        field_name: str,
        seed: int = 0,
    ):
        # Get the field data
        face_data_key = f"face_{field_name}"

        if not hasattr(mesh, 'face_attributes') or face_data_key not in mesh.face_attributes:
            # Try legacy metadata access
            if hasattr(mesh, 'metadata') and face_data_key in mesh.metadata:
                labels = np.array(mesh.metadata[face_data_key])
            else:
                print(f"[ScrambleIntField] Field '{field_name}' not found, returning unchanged")
                return (mesh,)
        else:
            labels = np.array(mesh.face_attributes[face_data_key])

        print(f"[ScrambleIntField] Input field '{field_name}': {len(np.unique(labels))} unique values")

        # Build face adjacency (which faces share edges)
        face_adjacency = mesh.face_adjacency  # Nx2 array of adjacent face pairs

        # Build segment adjacency graph
        # nodes = unique labels, edges = labels that are adjacent on the mesh
        unique_labels = np.unique(labels)
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        n_labels = len(unique_labels)

        # Find which segments are adjacent
        segment_adj = defaultdict(set)
        for f1, f2 in face_adjacency:
            l1, l2 = labels[f1], labels[f2]
            if l1 != l2:
                segment_adj[l1].add(l2)
                segment_adj[l2].add(l1)

        print(f"[ScrambleIntField] Built segment adjacency graph: {n_labels} nodes")

        # Greedy graph coloring to assign new values
        # Goal: adjacent segments get maximally different values
        rng = np.random.RandomState(seed)

        # Shuffle order for variety
        label_order = list(unique_labels)
        rng.shuffle(label_order)

        # Assign colors greedily
        color_assignment = {}
        for label in label_order:
            # Get colors used by neighbors
            neighbor_colors = {color_assignment[n] for n in segment_adj[label] if n in color_assignment}

            # Find the first available color that's not used by neighbors
            # Try to spread colors out by using modular spacing
            color = 0
            while color in neighbor_colors:
                color += 1
            color_assignment[label] = color

        # Now we have a valid graph coloring, but the values are 0, 1, 2, ...
        # Spread them out across the original value range for better visual contrast
        max_color = max(color_assignment.values()) + 1
        n_original = len(unique_labels)

        # Create a mapping that spreads colors across the range
        # Use golden ratio spacing for perceptually even distribution
        golden_ratio = (1 + np.sqrt(5)) / 2
        spread_map = {}
        for label, color in color_assignment.items():
            # Use golden ratio to spread values
            spread_value = int((color * golden_ratio * n_original) % n_original)
            spread_map[label] = spread_value

        # Apply the remapping to face labels
        new_labels = np.array([spread_map[l] for l in labels])

        # Create output mesh with scrambled field
        output_mesh = mesh.copy()

        # Store in face_attributes
        if not hasattr(output_mesh, 'face_attributes') or output_mesh.face_attributes is None:
            output_mesh.face_attributes = {}
        output_mesh.face_attributes[face_data_key] = new_labels

        # Also store in metadata for compatibility
        if not hasattr(output_mesh, 'metadata') or output_mesh.metadata is None:
            output_mesh.metadata = {}
        output_mesh.metadata[face_data_key] = new_labels

        print(f"[ScrambleIntField] Output: scrambled {n_labels} segments using {max_color} colors")

        return (output_mesh,)


NODE_CLASS_MAPPINGS = {
    "ScrambleIntField": ScrambleIntField,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScrambleIntField": "Scramble Int Field",
}
