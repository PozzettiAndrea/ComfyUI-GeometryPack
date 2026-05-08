# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Transform - Apply various transformations to meshes.

Supports:
- translate: Move mesh by offset
- rotate: Rotate around axes (degrees)
- scale: Scale uniformly or per-axis
- mirror: Mirror/reflect across axis
- center: Center mesh at origin
- align_to_axes: Align principal axes to world axes
- apply_matrix: Apply custom 4x4 transformation matrix
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class TransformMeshNode(io.ComfyNode):
    """
    Unified Transform - Apply various transformations to meshes.

    Supports:
    - translate: Move mesh by offset
    - rotate: Rotate around axes (degrees)
    - scale: Scale uniformly or per-axis
    - mirror: Mirror/reflect across axis
    - center: Center mesh at origin
    - align_to_axes: Align principal axes to world axes
    - apply_matrix: Apply custom 4x4 transformation matrix
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackTransformMesh",
            display_name="Transform Mesh",
            category="geompack/transforms",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.DynamicCombo.Input("operation", options=[
                    io.DynamicCombo.Option("translate", [
                        io.Float.Input("translate_x", default=0.0, min=-1000.0, max=1000.0, step=0.1),
                        io.Float.Input("translate_y", default=0.0, min=-1000.0, max=1000.0, step=0.1),
                        io.Float.Input("translate_z", default=0.0, min=-1000.0, max=1000.0, step=0.1),
                    ]),
                    io.DynamicCombo.Option("rotate", [
                        io.Float.Input("rotate_x", default=0.0, min=-360.0, max=360.0, step=1.0),
                        io.Float.Input("rotate_y", default=0.0, min=-360.0, max=360.0, step=1.0),
                        io.Float.Input("rotate_z", default=0.0, min=-360.0, max=360.0, step=1.0),
                    ]),
                    io.DynamicCombo.Option("scale", [
                        io.Float.Input("scale_uniform", default=1.0, min=0.001, max=1000.0, step=0.1),
                        io.Float.Input("scale_x", default=1.0, min=0.001, max=1000.0, step=0.1),
                        io.Float.Input("scale_y", default=1.0, min=0.001, max=1000.0, step=0.1),
                        io.Float.Input("scale_z", default=1.0, min=0.001, max=1000.0, step=0.1),
                    ]),
                    io.DynamicCombo.Option("mirror", [
                        io.Combo.Input("mirror_axis", options=["x", "y", "z"], default="x"),
                    ]),
                    io.DynamicCombo.Option("center", [
                        io.Combo.Input("center_x", options=["true", "false"], default="true"),
                        io.Combo.Input("center_y", options=["true", "false"], default="true"),
                        io.Combo.Input("center_z", options=["true", "false"], default="true"),
                    ]),
                    io.DynamicCombo.Option("align_to_axes", []),
                    io.DynamicCombo.Option("apply_matrix", [
                        io.String.Input("matrix_string", default="1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1", multiline=False),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="transformed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, operation):
        """Apply transformation to mesh."""
        selected = operation["operation"]

        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Operation: %s", selected)

        # Create copy to avoid modifying original
        result = trimesh.copy()

        if selected == "translate":
            result, info = cls._translate(
                result,
                operation.get("translate_x", 0.0),
                operation.get("translate_y", 0.0),
                operation.get("translate_z", 0.0),
            )
        elif selected == "rotate":
            result, info = cls._rotate(
                result,
                operation.get("rotate_x", 0.0),
                operation.get("rotate_y", 0.0),
                operation.get("rotate_z", 0.0),
            )
        elif selected == "scale":
            result, info = cls._scale(
                result,
                operation.get("scale_uniform", 1.0),
                operation.get("scale_x", 1.0),
                operation.get("scale_y", 1.0),
                operation.get("scale_z", 1.0),
            )
        elif selected == "mirror":
            result, info = cls._mirror(result, operation.get("mirror_axis", "x"))
        elif selected == "center":
            result, info = cls._center(
                result,
                operation.get("center_x", "true"),
                operation.get("center_y", "true"),
                operation.get("center_z", "true"),
            )
        elif selected == "align_to_axes":
            result, info = cls._align_to_axes(result)
        elif selected == "apply_matrix":
            result, info = cls._apply_matrix(
                result,
                operation.get("matrix_string", "1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1"),
            )
        else:
            raise ValueError(f"Unknown operation: {selected}")

        # Preserve metadata
        result.metadata = trimesh.metadata.copy()
        result.metadata['transform'] = {
            'operation': selected,
            'original_bounds': trimesh.bounds.tolist(),
            'new_bounds': result.bounds.tolist()
        }

        log.info("Complete")
        return io.NodeOutput(result, info, ui={"text": [info]})

    @staticmethod
    def _translate(mesh, tx, ty, tz):
        """Translate mesh by offset."""
        translation = np.array([tx, ty, tz])
        mesh.apply_translation(translation)

        info = f"""Transform Results (Translate):

Translation Vector: [{tx:.3f}, {ty:.3f}, {tz:.3f}]

New Bounds:
  Min: [{mesh.bounds[0][0]:.3f}, {mesh.bounds[0][1]:.3f}, {mesh.bounds[0][2]:.3f}]
  Max: [{mesh.bounds[1][0]:.3f}, {mesh.bounds[1][1]:.3f}, {mesh.bounds[1][2]:.3f}]
"""
        return mesh, info

    @staticmethod
    def _rotate(mesh, rx, ry, rz):
        """Rotate mesh around axes (degrees)."""
        # Convert to radians
        rx_rad = np.radians(rx)
        ry_rad = np.radians(ry)
        rz_rad = np.radians(rz)

        # Create rotation matrices (X, Y, Z order)
        # Rotation around X
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(rx_rad), -np.sin(rx_rad), 0],
            [0, np.sin(rx_rad), np.cos(rx_rad), 0],
            [0, 0, 0, 1]
        ])

        # Rotation around Y
        Ry = np.array([
            [np.cos(ry_rad), 0, np.sin(ry_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(ry_rad), 0, np.cos(ry_rad), 0],
            [0, 0, 0, 1]
        ])

        # Rotation around Z
        Rz = np.array([
            [np.cos(rz_rad), -np.sin(rz_rad), 0, 0],
            [np.sin(rz_rad), np.cos(rz_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Combined rotation: Z * Y * X (applied in reverse order)
        rotation_matrix = Rz @ Ry @ Rx
        mesh.apply_transform(rotation_matrix)

        info = f"""Transform Results (Rotate):

Rotation (degrees):
  X: {rx:.1f}°
  Y: {ry:.1f}°
  Z: {rz:.1f}°

Order: X -> Y -> Z (Euler angles)

New Bounds:
  Min: [{mesh.bounds[0][0]:.3f}, {mesh.bounds[0][1]:.3f}, {mesh.bounds[0][2]:.3f}]
  Max: [{mesh.bounds[1][0]:.3f}, {mesh.bounds[1][1]:.3f}, {mesh.bounds[1][2]:.3f}]
"""
        return mesh, info

    @staticmethod
    def _scale(mesh, uniform, sx, sy, sz):
        """Scale mesh uniformly or per-axis."""
        # If uniform scale is not 1.0, use it; otherwise use per-axis
        if abs(uniform - 1.0) > 1e-6:
            scale_factors = [uniform, uniform, uniform]
        else:
            scale_factors = [sx, sy, sz]

        mesh.apply_scale(scale_factors)

        info = f"""Transform Results (Scale):

Scale Factors:
  X: {scale_factors[0]:.3f}
  Y: {scale_factors[1]:.3f}
  Z: {scale_factors[2]:.3f}

New Bounds:
  Min: [{mesh.bounds[0][0]:.3f}, {mesh.bounds[0][1]:.3f}, {mesh.bounds[0][2]:.3f}]
  Max: [{mesh.bounds[1][0]:.3f}, {mesh.bounds[1][1]:.3f}, {mesh.bounds[1][2]:.3f}]

New Extents: [{mesh.extents[0]:.3f}, {mesh.extents[1]:.3f}, {mesh.extents[2]:.3f}]
"""
        return mesh, info

    @staticmethod
    def _mirror(mesh, axis):
        """Mirror mesh across axis."""
        # Mirror by negating the appropriate coordinate
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]

        # Negate vertices along axis
        mesh.vertices[:, axis_idx] *= -1

        # Flip face winding to maintain correct normals
        mesh.faces = mesh.faces[:, ::-1]

        # Reset normals cache
        mesh._cache.clear()

        info = f"""Transform Results (Mirror):

Mirror Axis: {axis.upper()}

Vertices mirrored and face winding corrected.

New Bounds:
  Min: [{mesh.bounds[0][0]:.3f}, {mesh.bounds[0][1]:.3f}, {mesh.bounds[0][2]:.3f}]
  Max: [{mesh.bounds[1][0]:.3f}, {mesh.bounds[1][1]:.3f}, {mesh.bounds[1][2]:.3f}]
"""
        return mesh, info

    @staticmethod
    def _center(mesh, cx, cy, cz):
        """Center mesh at origin (selective axes)."""
        bounds_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        translation = np.array([0.0, 0.0, 0.0])

        if cx == "true":
            translation[0] = -bounds_center[0]
        if cy == "true":
            translation[1] = -bounds_center[1]
        if cz == "true":
            translation[2] = -bounds_center[2]

        mesh.apply_translation(translation)

        new_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0

        info = f"""Transform Results (Center):

Centered Axes: X={cx}, Y={cy}, Z={cz}

Original Center: [{bounds_center[0]:.3f}, {bounds_center[1]:.3f}, {bounds_center[2]:.3f}]
New Center: [{new_center[0]:.3f}, {new_center[1]:.3f}, {new_center[2]:.3f}]

Translation Applied: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]
"""
        return mesh, info

    @staticmethod
    def _align_to_axes(mesh):
        """Align mesh principal axes to world axes."""
        # Get principal axes using PCA
        centered_vertices = mesh.vertices - mesh.vertices.mean(axis=0)
        covariance = np.cov(centered_vertices.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Sort by eigenvalue (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1

        # Create rotation matrix to align with world axes
        rotation = np.eye(4)
        rotation[:3, :3] = eigenvectors.T

        # Center first, then rotate
        center = mesh.vertices.mean(axis=0)
        mesh.apply_translation(-center)
        mesh.apply_transform(rotation)

        info = f"""Transform Results (Align to Axes):

Principal axes aligned to world X, Y, Z axes.
Mesh centered and rotated using PCA.

New Bounds:
  Min: [{mesh.bounds[0][0]:.3f}, {mesh.bounds[0][1]:.3f}, {mesh.bounds[0][2]:.3f}]
  Max: [{mesh.bounds[1][0]:.3f}, {mesh.bounds[1][1]:.3f}, {mesh.bounds[1][2]:.3f}]

New Extents: [{mesh.extents[0]:.3f}, {mesh.extents[1]:.3f}, {mesh.extents[2]:.3f}]
"""
        return mesh, info

    @staticmethod
    def _apply_matrix(mesh, matrix_string):
        """Apply custom 4x4 transformation matrix."""
        try:
            # Parse comma-separated values
            values = [float(x.strip()) for x in matrix_string.split(',')]
            if len(values) != 16:
                raise ValueError(f"Expected 16 values, got {len(values)}")

            # Reshape to 4x4 (row-major)
            matrix = np.array(values).reshape(4, 4)

            mesh.apply_transform(matrix)

            info = f"""Transform Results (Apply Matrix):

Matrix (4x4):
  [{matrix[0, 0]:.3f}, {matrix[0, 1]:.3f}, {matrix[0, 2]:.3f}, {matrix[0, 3]:.3f}]
  [{matrix[1, 0]:.3f}, {matrix[1, 1]:.3f}, {matrix[1, 2]:.3f}, {matrix[1, 3]:.3f}]
  [{matrix[2, 0]:.3f}, {matrix[2, 1]:.3f}, {matrix[2, 2]:.3f}, {matrix[2, 3]:.3f}]
  [{matrix[3, 0]:.3f}, {matrix[3, 1]:.3f}, {matrix[3, 2]:.3f}, {matrix[3, 3]:.3f}]

New Bounds:
  Min: [{mesh.bounds[0][0]:.3f}, {mesh.bounds[0][1]:.3f}, {mesh.bounds[0][2]:.3f}]
  Max: [{mesh.bounds[1][0]:.3f}, {mesh.bounds[1][1]:.3f}, {mesh.bounds[1][2]:.3f}]
"""
        except Exception as e:
            raise ValueError(f"Invalid matrix string: {e}")

        return mesh, info


NODE_CLASS_MAPPINGS = {
    "GeomPackTransformMesh": TransformMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackTransformMesh": "Transform Mesh",
}
