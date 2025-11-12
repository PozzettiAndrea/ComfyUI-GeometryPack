"""
Skeleton extraction and visualization nodes for geometry processing.

These nodes provide tools for:
- Extracting skeletons from 3D meshes (using Skeletor library)
- Converting skeletons to various mesh representations
- Visualizing skeletal structures
"""

import numpy as np
import trimesh
from typing import Dict


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


# =============================================================================
# Node 1: ExtractSkeleton
# =============================================================================

class ExtractSkeleton:
    """
    Extract skeleton from 3D mesh using Skeletor library.

    Outputs normalized skeleton data (vertices + edges) in [-1, 1] range.
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

    def extract(self, trimesh, algorithm, fix_mesh,
                waves=1, step_size=1.0,
                sampling_dist=1.0, cluster_pos="median",
                shape_weight=1.0, sample_weight=0.1,
                inv_dist=10.0, min_length=0.0):
        """Extract skeleton from mesh."""
        try:
            import skeletor as sk
        except ImportError:
            raise ImportError(
                "Skeletor library not found. Please install: pip install skeletor"
            )

        print(f"[ExtractSkeleton] Extracting skeleton using {algorithm} algorithm...")

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

        # NORMALIZE IMMEDIATELY to [-1, 1]
        vertices = normalize_skeleton(vertices)
        print(f"[ExtractSkeleton] Normalized to range [{vertices.min():.3f}, {vertices.max():.3f}]")

        # Package as skeleton data
        skeleton = {
            "vertices": vertices,  # [N, 3] joint positions
            "edges": edges,        # [M, 2] bone connections (vertex indices)
        }

        return (skeleton,)


# =============================================================================
# Node 2: SkeletonToTrimesh (Raw - just vertices and edges)
# =============================================================================

class SkeletonToTrimesh:
    """
    Convert skeleton to trimesh with just vertices and edges (line segments).

    Lightweight visualization without fancy geometry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON",),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "convert"
    CATEGORY = "geompack/skeleton"

    def convert(self, skeleton):
        """Convert skeleton to line mesh."""
        vertices = skeleton["vertices"]
        edges = skeleton["edges"]

        print(f"[SkeletonToTrimesh] Creating line mesh: {len(vertices)} vertices, {len(edges)} edges")

        # Create path3D (line segments)
        paths = []
        for edge in edges:
            paths.append([edge[0], edge[1]])

        # Create trimesh Path3D object
        skeleton_mesh = trimesh.path.Path3D(
            entities=[trimesh.path.entities.Line(path) for path in paths],
            vertices=vertices
        )

        return (skeleton_mesh,)


# =============================================================================
# Node 3: SkeletonToMesh (Fancy - with cylinders and spheres)
# =============================================================================

class SkeletonToMesh:
    """
    Convert skeleton to solid mesh with cylinders (bones) and spheres (joints).

    High-quality visualization with adjustable geometry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON",),
                "joint_radius": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 0.1, "step": 0.001}),
                "bone_radius": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.05, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "convert"
    CATEGORY = "geompack/skeleton"

    def convert(self, skeleton, joint_radius, bone_radius):
        """Convert skeleton to solid geometry."""
        vertices = skeleton["vertices"]
        edges = skeleton["edges"]

        print(f"[SkeletonToMesh] Creating solid mesh: {len(vertices)} joints, {len(edges)} bones")

        meshes = []

        # Create joint spheres
        for vertex in vertices:
            sphere = trimesh.creation.uv_sphere(radius=joint_radius, count=[8, 8])
            sphere.apply_translation(vertex)
            meshes.append(sphere)

        # Create bone cylinders
        for edge in edges:
            start = vertices[edge[0]]
            end = vertices[edge[1]]

            # Calculate cylinder parameters
            direction = end - start
            length = np.linalg.norm(direction)

            if length < 1e-6:
                continue  # Skip degenerate bones

            # Create cylinder along Z-axis
            cylinder = trimesh.creation.cylinder(
                radius=bone_radius,
                height=length,
                sections=8
            )

            # Calculate rotation to align with bone direction
            z_axis = np.array([0, 0, 1])
            bone_direction = direction / length

            # Rotation axis and angle
            rotation_axis = np.cross(z_axis, bone_direction)
            rotation_axis_norm = np.linalg.norm(rotation_axis)

            if rotation_axis_norm > 1e-6:
                rotation_axis = rotation_axis / rotation_axis_norm
                rotation_angle = np.arccos(np.clip(np.dot(z_axis, bone_direction), -1.0, 1.0))

                # Create rotation matrix
                from trimesh.transformations import rotation_matrix
                rotation = rotation_matrix(rotation_angle, rotation_axis)
                cylinder.apply_transform(rotation)

            # Translate to midpoint
            midpoint = (start + end) / 2
            cylinder.apply_translation(midpoint)

            meshes.append(cylinder)

        # Combine all meshes
        if not meshes:
            raise ValueError("No geometry created from skeleton")

        combined_mesh = trimesh.util.concatenate(meshes)

        print(f"[SkeletonToMesh] Created mesh: {len(combined_mesh.vertices)} vertices, "
              f"{len(combined_mesh.faces)} faces")

        return (combined_mesh,)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "ExtractSkeleton": ExtractSkeleton,
    "SkeletonToTrimesh": SkeletonToTrimesh,
    "SkeletonToMesh": SkeletonToMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractSkeleton": "Extract Skeleton from Mesh",
    "SkeletonToTrimesh": "Skeleton to Trimesh (Lines)",
    "SkeletonToMesh": "Skeleton to Mesh (Solid)",
}
