# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Fix inconsistent normal orientations.
"""

import trimesh
import numpy as np

try:
    import igl
    HAS_IGL = True
except ImportError:
    HAS_IGL = False


def _orient_outward_winding(V, F, face_normals):
    """
    Use fast winding number to orient faces outward.

    For each face, query winding number at a point offset slightly along
    the negative normal. If winding > 0.5, point is inside mesh = normal
    points outward (correct). If winding < 0.5, normal points inward (flip).

    Args:
        V: Vertices array (N, 3)
        F: Faces array (M, 3)
        face_normals: Face normals array (M, 3)

    Returns:
        tuple: (F_out, flip_mask, num_flipped)
    """
    # Compute face centroids
    face_centroids = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0

    # Adaptive epsilon based on mesh scale
    bbox_diag = np.linalg.norm(V.max(axis=0) - V.min(axis=0))
    eps = 1e-4 * bbox_diag

    # Query points offset slightly inside (along -normal direction)
    query_inside = face_centroids - face_normals * eps

    # Compute winding numbers
    W = igl.fast_winding_number(V, F, query_inside)

    # If winding < 0.5, the query point is outside, meaning normal points inward
    flip_mask = W < 0.5

    # Flip faces by reversing vertex order
    F_out = F.copy()
    F_out[flip_mask] = F_out[flip_mask][:, [0, 2, 1]]

    return F_out, flip_mask, np.sum(flip_mask)


def _orient_outward_raycast(V, F, face_normals):
    """
    Use ray-mesh intersection to orient faces outward (odd/even test).

    Cast ray from face centroid along normal direction. Odd number of
    intersections = pointing inward (flip needed).

    Args:
        V: Vertices array (N, 3)
        F: Faces array (M, 3)
        face_normals: Face normals array (M, 3)

    Returns:
        tuple: (F_out, flip_mask, num_flipped)
    """
    # Compute face centroids
    face_centroids = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0

    # Small offset to avoid self-intersection
    eps = 1e-6

    flip_mask = np.zeros(len(F), dtype=bool)

    for i in range(len(F)):
        origin = face_centroids[i] + face_normals[i] * eps
        direction = face_normals[i]

        # Cast ray and get hits
        hits = igl.ray_mesh_intersect(
            np.ascontiguousarray(origin, dtype=np.float64),
            np.ascontiguousarray(direction, dtype=np.float64),
            V, F
        )

        # Odd number of hits = pointing inward
        if hits is not None and len(hits) % 2 == 1:
            flip_mask[i] = True

    # Flip faces by reversing vertex order
    F_out = F.copy()
    F_out[flip_mask] = F_out[flip_mask][:, [0, 2, 1]]

    return F_out, flip_mask, np.sum(flip_mask)


def _orient_outward_signed_dist(V, F, face_normals):
    """
    Use signed distance to orient faces outward.

    Query signed distance at points offset along normal direction.
    Positive distance = inside surface = flip needed.

    Args:
        V: Vertices array (N, 3)
        F: Faces array (M, 3)
        face_normals: Face normals array (M, 3)

    Returns:
        tuple: (F_out, flip_mask, num_flipped)
    """
    # Compute face centroids
    face_centroids = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0

    # Adaptive epsilon based on mesh scale
    bbox_diag = np.linalg.norm(V.max(axis=0) - V.min(axis=0))
    eps = 1e-4 * bbox_diag

    # Query points offset along normal direction (outside if normal correct)
    query_points = face_centroids + face_normals * eps

    # Compute signed distance using pseudonormal method
    S, I, C, N = igl.signed_distance(
        np.ascontiguousarray(query_points, dtype=np.float64),
        V, F,
        igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    )

    # Positive signed distance = inside mesh = normal points inward
    flip_mask = S > 0

    # Flip faces by reversing vertex order
    F_out = F.copy()
    F_out[flip_mask] = F_out[flip_mask][:, [0, 2, 1]]

    return F_out, flip_mask, np.sum(flip_mask)


class FixNormalsNode:
    """
    Fix inconsistent normal orientations.

    Ensures all face normals point consistently (all outward or all inward).
    Uses graph traversal to propagate consistent orientation across the trimesh.
    Essential for proper rendering and boolean operations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Methods:
        # - trimesh: Basic trimesh fix_normals
        # - igl_bfs: BFS-based consistent orientation (best for thin/open surfaces)
        # - igl_winding: Fast winding number (best for closed volumes)
        # - igl_raycast: Ray-mesh intersection odd/even test (closed volumes)
        # - igl_signed_dist: Signed distance pseudonormal (closed volumes)
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "method": (["trimesh", "igl_bfs", "igl_winding", "igl_raycast", "igl_signed_dist"], {"default": "trimesh"}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("fixed_mesh", "info")
    FUNCTION = "fix_normals"
    CATEGORY = "geompack/repair"

    def fix_normals(self, trimesh, method="trimesh"):
        """
        Fix inconsistent face normal orientations.

        Args:
            trimesh: Input trimesh.Trimesh object
            method: Orientation method

        Methods:
            - trimesh: Basic trimesh fix_normals
            - igl_bfs: BFS-based consistent orientation (best for thin/open surfaces)
            - igl_winding: Fast winding number outward orientation (closed volumes)
            - igl_raycast: Ray-mesh intersection odd/even test (closed volumes)
            - igl_signed_dist: Signed distance pseudonormal (closed volumes)

        Returns:
            tuple: (fixed_trimesh, info_string)
        """
        print(f"[FixNormals] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Create a copy to avoid modifying the original
        fixed_mesh = trimesh.copy()

        # Check initial winding consistency
        was_consistent = fixed_mesh.is_winding_consistent

        # Fix normals using selected method
        method_used = method
        num_components = None
        num_flipped = None
        extra_info = ""

        # Check if igl is required but not available
        igl_methods = ["igl_bfs", "igl_winding", "igl_raycast", "igl_signed_dist"]
        if method in igl_methods and not HAS_IGL:
            print(f"[FixNormals] igl not available, falling back to trimesh method")
            fixed_mesh.fix_normals()
            method_used = "trimesh (fallback - igl not available)"

        elif method == "igl_bfs":
            # Use libigl's BFS-based orientation (best for thin/open surfaces)
            V = np.asarray(fixed_mesh.vertices, dtype=np.float64)
            F = np.asarray(fixed_mesh.faces, dtype=np.int64)

            # Orient faces using BFS
            FF, C = igl.bfs_orient(F)

            # Update mesh faces with oriented version
            fixed_mesh.faces = FF

            # Track number of orientation components
            num_components = len(np.unique(C))
            print(f"[FixNormals] igl.bfs_orient: {num_components} orientation components")
            extra_info = "\nNote: BFS makes faces consistent but doesn't determine inside/outside"

        elif method == "igl_winding":
            # Use fast winding number to orient outward (best for closed volumes)
            V = np.asarray(fixed_mesh.vertices, dtype=np.float64)
            F = np.asarray(fixed_mesh.faces, dtype=np.int64)

            # Compute face normals first
            face_normals = igl.per_face_normals(V, F, np.array([1., 1., 1.]))

            # Orient faces outward using winding number
            FF, flip_mask, num_flipped = _orient_outward_winding(V, F, face_normals)
            fixed_mesh.faces = FF

            print(f"[FixNormals] igl_winding: flipped {num_flipped}/{len(F)} faces")
            extra_info = "\nNote: Winding number works best on closed/watertight meshes"

        elif method == "igl_raycast":
            # Use ray-mesh intersection odd/even test
            V = np.asarray(fixed_mesh.vertices, dtype=np.float64)
            F = np.asarray(fixed_mesh.faces, dtype=np.int64)

            # Compute face normals first
            face_normals = igl.per_face_normals(V, F, np.array([1., 1., 1.]))

            # Orient faces outward using raycasting
            FF, flip_mask, num_flipped = _orient_outward_raycast(V, F, face_normals)
            fixed_mesh.faces = FF

            print(f"[FixNormals] igl_raycast: flipped {num_flipped}/{len(F)} faces")
            extra_info = "\nNote: Raycasting works best on closed meshes without self-intersections"

        elif method == "igl_signed_dist":
            # Use signed distance with pseudonormal
            V = np.asarray(fixed_mesh.vertices, dtype=np.float64)
            F = np.asarray(fixed_mesh.faces, dtype=np.int64)

            # Compute face normals first
            face_normals = igl.per_face_normals(V, F, np.array([1., 1., 1.]))

            # Orient faces outward using signed distance
            FF, flip_mask, num_flipped = _orient_outward_signed_dist(V, F, face_normals)
            fixed_mesh.faces = FF

            print(f"[FixNormals] igl_signed_dist: flipped {num_flipped}/{len(F)} faces")
            extra_info = "\nNote: Signed distance works best on watertight meshes"

        else:
            # Use trimesh's built-in method
            fixed_mesh.fix_normals()

        # Check if it's now consistent
        is_consistent = fixed_mesh.is_winding_consistent

        # Build info string
        components_info = f"\nOrientation Components: {num_components}" if num_components is not None else ""
        flipped_info = f"\nFaces Flipped: {num_flipped}" if num_flipped is not None else ""

        info = f"""Normal Orientation Fix:

Method: {method_used}
Before: {'Consistent' if was_consistent else 'Inconsistent'}
After:  {'Consistent' if is_consistent else 'Inconsistent'}{components_info}{flipped_info}

Vertices: {len(fixed_mesh.vertices):,}
Faces: {len(fixed_mesh.faces):,}
{extra_info}
{'✓ Normals are now consistently oriented!' if is_consistent else '⚠ Some inconsistencies may remain (check mesh topology)'}
"""

        print(f"[FixNormals] {'✓' if is_consistent else '⚠'} Normal orientation: {was_consistent} -> {is_consistent}")

        return (fixed_mesh, info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackFixNormals": FixNormalsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackFixNormals": "Fix Normals",
}
