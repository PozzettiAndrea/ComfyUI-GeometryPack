# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Mesh Quality Node - Analyze mesh quality metrics
"""

import numpy as np


class MeshQualityNode:
    """
    Mesh Quality - Compute quality metrics for mesh analysis.

    Analyzes various quality metrics of a mesh including:
    - Triangle quality (aspect ratio, angles)
    - Edge length statistics
    - Face area distribution
    - Degeneracy detection
    - Manifold status

    Useful for identifying problematic geometry before further processing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "include_face_quality": ("BOOLEAN", {"default": True}),
                "include_edge_stats": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("min_quality", "mean_quality", "report")
    OUTPUT_NODE = True
    FUNCTION = "analyze_quality"
    CATEGORY = "geompack/analysis"

    def analyze_quality(self, trimesh, include_face_quality=True, include_edge_stats=True):
        """
        Analyze mesh quality metrics.

        Args:
            trimesh: Input trimesh.Trimesh object
            include_face_quality: Compute face quality metrics
            include_edge_stats: Compute edge length statistics

        Returns:
            tuple: (min_quality, mean_quality, report_string)
        """
        print(f"[MeshQuality] Analyzing mesh: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        report_sections = []

        # Basic topology
        is_watertight = trimesh.is_watertight
        is_winding_consistent = trimesh.is_winding_consistent
        is_manifold = trimesh.is_volume

        report_sections.append(f"""Topology:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}
  Edges: {len(trimesh.edges):,}
  Watertight: {'Yes' if is_watertight else 'No'}
  Winding Consistent: {'Yes' if is_winding_consistent else 'No'}
  Volume Mesh: {'Yes' if is_manifold else 'No'}""")

        # Face quality metrics
        face_qualities = None
        if include_face_quality:
            face_qualities = self._compute_face_quality(trimesh)

            min_quality = float(np.min(face_qualities))
            max_quality = float(np.max(face_qualities))
            mean_quality = float(np.mean(face_qualities))
            median_quality = float(np.median(face_qualities))

            # Count poor quality faces (quality < 0.3)
            poor_faces = np.sum(face_qualities < 0.3)
            good_faces = np.sum(face_qualities >= 0.6)

            report_sections.append(f"""Face Quality (Aspect Ratio):
  Min: {min_quality:.4f}
  Max: {max_quality:.4f}
  Mean: {mean_quality:.4f}
  Median: {median_quality:.4f}
  Poor Quality Faces (< 0.3): {poor_faces:,} ({100.0 * poor_faces / len(trimesh.faces):.1f}%)
  Good Quality Faces (>= 0.6): {good_faces:,} ({100.0 * good_faces / len(trimesh.faces):.1f}%)""")
        else:
            min_quality = 1.0
            mean_quality = 1.0

        # Edge length statistics
        if include_edge_stats:
            edge_lengths = self._compute_edge_lengths(trimesh)

            min_edge = float(np.min(edge_lengths))
            max_edge = float(np.max(edge_lengths))
            mean_edge = float(np.mean(edge_lengths))
            median_edge = float(np.median(edge_lengths))
            std_edge = float(np.std(edge_lengths))

            # Edge length uniformity (std / mean)
            uniformity = std_edge / mean_edge if mean_edge > 0 else 0

            report_sections.append(f"""Edge Lengths:
  Min: {min_edge:.6f}
  Max: {max_edge:.6f}
  Mean: {mean_edge:.6f}
  Median: {median_edge:.6f}
  Std Dev: {std_edge:.6f}
  Uniformity (std/mean): {uniformity:.4f}""")

        # Face area statistics
        face_areas = trimesh.area_faces
        min_area = float(np.min(face_areas))
        max_area = float(np.max(face_areas))
        mean_area = float(np.mean(face_areas))
        total_area = float(trimesh.area)

        # Count degenerate faces (area near zero)
        degenerate_faces = np.sum(face_areas < 1e-10)

        report_sections.append(f"""Face Areas:
  Total Surface Area: {total_area:.6f}
  Min Face Area: {min_area:.6f}
  Max Face Area: {max_area:.6f}
  Mean Face Area: {mean_area:.6f}
  Degenerate Faces (area < 1e-10): {degenerate_faces:,}""")

        # Angle analysis
        if include_face_quality:
            angles = self._compute_face_angles(trimesh)
            min_angle = float(np.min(angles))
            max_angle = float(np.max(angles))
            mean_angle = float(np.mean(angles))

            # Count problematic angles
            acute_angles = np.sum(angles < 30.0)  # Very acute
            obtuse_angles = np.sum(angles > 120.0)  # Very obtuse

            report_sections.append(f"""Face Angles (degrees):
  Min: {min_angle:.2f}°
  Max: {max_angle:.2f}°
  Mean: {mean_angle:.2f}°
  Very Acute (< 30°): {acute_angles:,} ({100.0 * acute_angles / len(angles):.1f}%)
  Very Obtuse (> 120°): {obtuse_angles:,} ({100.0 * obtuse_angles / len(angles):.1f}%)""")

        # Combine report
        report = "Mesh Quality Analysis:\n\n" + "\n\n".join(report_sections)

        # Add recommendations
        recommendations = []
        if degenerate_faces > 0:
            recommendations.append("- Remove degenerate faces")
        if not is_watertight:
            recommendations.append("- Mesh has holes or boundaries")
        if not is_winding_consistent:
            recommendations.append("- Fix inconsistent face windings")
        if face_qualities is not None and mean_quality < 0.5:
            recommendations.append("- Consider remeshing to improve triangle quality")

        if recommendations:
            report += "\n\nRecommendations:\n" + "\n".join(recommendations)

        print(f"[MeshQuality] Analysis complete")

        return {
            "result": (min_quality, mean_quality, report),
            "ui": {"text": [report]}
        }

    def _compute_face_quality(self, mesh):
        """
        Compute face quality metric based on aspect ratio.

        Quality = (4 * sqrt(3) * area) / (sum of squared edge lengths)

        Returns values in [0, 1], where 1 = equilateral triangle, 0 = degenerate
        """
        vertices = mesh.vertices
        faces = mesh.faces

        # Get triangle vertices
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Compute edge vectors
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        # Edge lengths squared
        l0_sq = np.sum(e0 * e0, axis=1)
        l1_sq = np.sum(e1 * e1, axis=1)
        l2_sq = np.sum(e2 * e2, axis=1)

        # Face areas (using cross product)
        cross = np.cross(e0, -e2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        # Quality metric
        # q = (4 * sqrt(3) * area) / (l0^2 + l1^2 + l2^2)
        sum_lengths_sq = l0_sq + l1_sq + l2_sq
        quality = (4.0 * np.sqrt(3.0) * areas) / (sum_lengths_sq + 1e-10)

        # Clamp to [0, 1]
        quality = np.clip(quality, 0.0, 1.0)

        return quality

    def _compute_edge_lengths(self, mesh):
        """Compute all edge lengths"""
        edges = mesh.edges_unique
        vertices = mesh.vertices

        edge_vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)

        return edge_lengths

    def _compute_face_angles(self, mesh):
        """Compute all face angles in degrees"""
        vertices = mesh.vertices
        faces = mesh.faces

        # Get triangle vertices
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Compute edge vectors
        e0 = v1 - v0  # From v0 to v1
        e1 = v2 - v1  # From v1 to v2
        e2 = v0 - v2  # From v2 to v0

        # Normalize
        e0_norm = e0 / (np.linalg.norm(e0, axis=1, keepdims=True) + 1e-10)
        e1_norm = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-10)
        e2_norm = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-10)

        # Compute angles using dot product
        # Angle at v0
        angle_v0 = np.arccos(np.clip(np.sum(e0_norm * -e2_norm, axis=1), -1.0, 1.0))
        # Angle at v1
        angle_v1 = np.arccos(np.clip(np.sum(-e0_norm * e1_norm, axis=1), -1.0, 1.0))
        # Angle at v2
        angle_v2 = np.arccos(np.clip(np.sum(-e1_norm * e2_norm, axis=1), -1.0, 1.0))

        # Convert to degrees and concatenate all angles
        angles = np.concatenate([
            np.degrees(angle_v0),
            np.degrees(angle_v1),
            np.degrees(angle_v2)
        ])

        return angles


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackMeshQuality": MeshQualityNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackMeshQuality": "Mesh Quality",
}
