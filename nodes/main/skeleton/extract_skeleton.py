# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Extract Skeleton Node - Extract skeleton from 3D mesh
"""

import numpy as np
import trimesh


def normalize_skeleton(vertices: np.ndarray) -> np.ndarray:
    """
    Normalize skeleton vertices to [-1, 1] range.

    Args:
        vertices: Array of shape [N, 3]

    Returns:
        Normalized vertices in [-1, 1] range
    """
    # Find bounding box
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)

    # Center at origin
    center = (min_coords + max_coords) / 2
    vertices = vertices - center

    # Scale to [-1, 1]
    scale = (max_coords - min_coords).max() / 2
    if scale > 0:
        vertices = vertices / scale

    return vertices


class ExtractSkeleton:
    """
    Extract skeleton from 3D mesh using Skeletor library.

    Outputs skeleton data (vertices + edges) with optional normalization to [-1, 1] range.
    By default, preserves the original mesh scale.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "algorithm": (["wavefront", "vertex_clusters", "edge_collapse", "teasar"],
                             {"default": "wavefront"}),
                "fix_mesh": ("BOOLEAN", {"default": True,
                                        "tooltip": "Fix mesh issues before skeletonization"}),
                "normalize": ("BOOLEAN", {"default": False,
                                         "tooltip": "Normalize skeleton to [-1, 1] range (False preserves original mesh scale)"}),
            },
            "optional": {
                # Wavefront parameters
                "waves": ("INT", {"default": 1, "min": 1, "max": 20,
                                 "tooltip": "Wavefront: number of waves"}),
                "step_size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0,
                                       "tooltip": "Wavefront: step size (higher = coarser)"}),

                # Vertex clusters parameters
                "sampling_dist": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 50.0,
                                           "tooltip": "Vertex clusters: max distance for clustering"}),
                "cluster_pos": (["median", "center"], {"default": "median",
                                                       "tooltip": "Vertex clusters: cluster position method"}),

                # Edge collapse parameters
                "shape_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0,
                                          "tooltip": "Edge collapse: shape preservation weight"}),
                "sample_weight": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0,
                                           "tooltip": "Edge collapse: sampling quality weight"}),

                # TEASAR parameters
                "inv_dist": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0,
                                      "tooltip": "TEASAR: invalidation distance (lower = more detail)"}),
                "min_length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0,
                                        "tooltip": "TEASAR: minimum branch length to keep"}),
            }
        }

    RETURN_TYPES = ("SKELETON",)
    RETURN_NAMES = ("skeleton",)
    FUNCTION = "extract"
    CATEGORY = "geompack/skeleton"

    def extract(self, trimesh, algorithm, fix_mesh, normalize,
                waves=1, step_size=1.0,
                sampling_dist=1.0, cluster_pos="median",
                shape_weight=1.0, sample_weight=0.1,
                inv_dist=10.0, min_length=0.0):
        """Extract skeleton from mesh.

        Args:
            trimesh: Input mesh
            algorithm: Skeletonization algorithm to use
            fix_mesh: Whether to fix mesh issues before extraction
            normalize: If True, normalize skeleton to [-1, 1] range. If False, preserve original scale.
            ... (algorithm-specific parameters)
        """
        try:
            import skeletor as sk
        except ImportError:
            raise ImportError(
                "Skeletor library not found. Please install: pip install skeletor"
            )

        print(f"[ExtractSkeleton] Extracting skeleton using {algorithm} algorithm...")

        # Print input mesh bounding box
        mesh_min = trimesh.bounds[0]
        mesh_max = trimesh.bounds[1]
        mesh_size = mesh_max - mesh_min
        mesh_center = (mesh_min + mesh_max) / 2
        print(f"[ExtractSkeleton] Input mesh bounding box:")
        print(f"  Min: [{mesh_min[0]:.3f}, {mesh_min[1]:.3f}, {mesh_min[2]:.3f}]")
        print(f"  Max: [{mesh_max[0]:.3f}, {mesh_max[1]:.3f}, {mesh_max[2]:.3f}]")
        print(f"  Size: [{mesh_size[0]:.3f}, {mesh_size[1]:.3f}, {mesh_size[2]:.3f}]")
        print(f"  Center: [{mesh_center[0]:.3f}, {mesh_center[1]:.3f}, {mesh_center[2]:.3f}]")

        # Fix mesh if requested
        if fix_mesh:
            print("[ExtractSkeleton] Fixing mesh...")
            mesh = sk.pre.fix_mesh(trimesh, remove_disconnected=5, inplace=False)
        else:
            mesh = trimesh

        # Extract skeleton based on algorithm
        try:
            if algorithm == "wavefront":
                print(f"  Parameters: waves={waves}, step_size={step_size}")
                skel = sk.skeletonize.by_wavefront(mesh, waves=waves, step_size=step_size)

            elif algorithm == "vertex_clusters":
                print(f"  Parameters: sampling_dist={sampling_dist}, cluster_pos={cluster_pos}")
                skel = sk.skeletonize.by_vertex_clusters(
                    mesh,
                    sampling_dist=sampling_dist,
                    cluster_pos=cluster_pos
                )

            elif algorithm == "edge_collapse":
                print(f"  Parameters: shape_weight={shape_weight}, sample_weight={sample_weight}")
                skel = sk.skeletonize.by_edge_collapse(
                    mesh,
                    shape_weight=shape_weight,
                    sample_weight=sample_weight
                )

            elif algorithm == "teasar":
                print(f"  Parameters: inv_dist={inv_dist}, min_length={min_length}")
                skel = sk.skeletonize.by_teasar(
                    mesh,
                    inv_dist=inv_dist,
                    min_length=min_length if min_length > 0 else None
                )

            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        except Exception as e:
            print(f"[ExtractSkeleton] Error during skeletonization: {e}")
            raise RuntimeError(f"Skeletonization failed: {e}")

        # Get vertices and edges
        vertices = np.array(skel.vertices)
        edges = np.array(skel.edges)

        print(f"[ExtractSkeleton] Extracted {len(vertices)} joints, {len(edges)} bones")

        # Print skeleton bounding box before any normalization
        skel_min = vertices.min(axis=0)
        skel_max = vertices.max(axis=0)
        skel_size = skel_max - skel_min
        skel_center = (skel_min + skel_max) / 2
        print(f"[ExtractSkeleton] Skeleton bounding box (original):")
        print(f"  Min: [{skel_min[0]:.3f}, {skel_min[1]:.3f}, {skel_min[2]:.3f}]")
        print(f"  Max: [{skel_max[0]:.3f}, {skel_max[1]:.3f}, {skel_max[2]:.3f}]")
        print(f"  Size: [{skel_size[0]:.3f}, {skel_size[1]:.3f}, {skel_size[2]:.3f}]")
        print(f"  Center: [{skel_center[0]:.3f}, {skel_center[1]:.3f}, {skel_center[2]:.3f}]")

        # Store original scale and center for metadata
        original_scale = float((skel_max - skel_min).max() / 2)
        original_center = skel_center.copy()

        # Conditionally normalize
        if normalize:
            vertices = normalize_skeleton(vertices)

            # Print skeleton bounding box after normalization
            norm_min = vertices.min(axis=0)
            norm_max = vertices.max(axis=0)
            norm_size = norm_max - norm_min
            print(f"[ExtractSkeleton] Skeleton bounding box AFTER normalization:")
            print(f"  Min: [{norm_min[0]:.3f}, {norm_min[1]:.3f}, {norm_min[2]:.3f}]")
            print(f"  Max: [{norm_max[0]:.3f}, {norm_max[1]:.3f}, {norm_max[2]:.3f}]")
            print(f"  Size: [{norm_size[0]:.3f}, {norm_size[1]:.3f}, {norm_size[2]:.3f}]")
            print(f"  Overall range: [{vertices.min():.3f}, {vertices.max():.3f}]")
        else:
            print(f"[ExtractSkeleton] Normalization skipped - preserving original scale")

        # Package as skeleton data
        skeleton = {
            "vertices": vertices,  # [N, 3] joint positions
            "edges": edges,        # [M, 2] bone connections (vertex indices)
            "scale": original_scale,  # Original scale factor (for denormalization if needed)
            "center": original_center.tolist(),  # Original center point
            "normalized": normalize,  # Whether this skeleton was normalized
        }

        return (skeleton,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackExtractSkeleton": ExtractSkeleton,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackExtractSkeleton": "Extract Skeleton",
}
