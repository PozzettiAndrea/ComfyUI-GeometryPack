# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remesh Node - Universal topology-changing remeshing operations
"""

import numpy as np
import trimesh as trimesh_module

from .._utils import mesh_ops


class RemeshNode:
    """
    Universal Remesh - Unified topology-changing remeshing operations.

    Consolidates multiple remeshing backends into a single node:
    - cumesh: GPU-accelerated dual-contouring remeshing (same as TRELLIS2)
    - pymeshlab_isotropic: PyMeshLab isotropic remeshing
    - cgal_isotropic: CGAL high-quality isotropic remeshing
    - blender_voxel: Blender voxel-based remeshing (watertight output)
    - blender_quadriflow: Blender Quadriflow quad remeshing
    - instant_meshes: Field-aligned quad remeshing

    Parameters are backend-specific; unused parameters are ignored.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "backend": ([
                    "pymeshlab_isotropic",
                    "cgal_isotropic",
                    "blender_voxel",
                    "blender_quadriflow",
                    "instant_meshes",
                    "cumesh",
                ], {
                    "default": "pymeshlab_isotropic",
                    "tooltip": "Remeshing algorithm. pymeshlab=fast isotropic, cgal=high-quality isotropic, blender_voxel=watertight output, blender_quadriflow=quad remesh, cumesh=GPU dual-contouring, instant_meshes=field-aligned quads"
                }),
            },
            "optional": {
                # Isotropic params (pymeshlab, cgal)
                "target_edge_length": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Target edge length for output triangles. Value is relative to mesh scale.",
                    "backends": ["pymeshlab_isotropic", "cgal_isotropic"],
                }),
                "iterations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of remeshing passes. More iterations = smoother result, slower processing.",
                    "backends": ["pymeshlab_isotropic", "cgal_isotropic"],
                }),
                # PyMeshLab-specific
                "feature_angle": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Angle threshold (degrees) for feature edge detection. Edges with dihedral angle greater than this are preserved as sharp creases.",
                    "backends": ["pymeshlab_isotropic"],
                }),
                "adaptive": (["true", "false"], {
                    "default": "false",
                    "tooltip": "Use curvature-adaptive edge lengths. Creates smaller triangles in high-curvature areas, larger triangles in flat areas.",
                    "backends": ["pymeshlab_isotropic"],
                }),
                # CGAL-specific
                "protect_boundaries": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Lock boundary/open edges in place during remeshing. Prevents modification of mesh borders and holes.",
                    "backends": ["cgal_isotropic"],
                }),
                # Blender voxel
                "voxel_size": ("FLOAT", {
                    "default": 1,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Voxel size for Blender voxel remesh. Smaller = more detail, more faces. Output is always watertight.",
                    "backends": ["blender_voxel"],
                }),
                # CuMesh / Quadriflow
                "target_face_count": ("INT", {
                    "default": 500000,
                    "min": 1000,
                    "max": 5000000,
                    "step": 1000,
                    "tooltip": "Target number of output faces for cumesh/quadriflow backends.",
                    "backends": ["cumesh", "blender_quadriflow"],
                }),
                # CuMesh specific (matches TRELLIS2)
                "remesh_band": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Band width for CuMesh dual-contouring. Affects surface detail capture. Higher = smoother but may lose fine details.",
                    "backends": ["cumesh"],
                }),
                # Instant Meshes specific
                "target_vertex_count": ("INT", {
                    "default": 5000,
                    "min": 100,
                    "max": 1000000,
                    "step": 100,
                    "tooltip": "Target vertex count for Instant Meshes output. Creates field-aligned quad-dominant mesh.",
                    "backends": ["instant_meshes"],
                }),
                "deterministic": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Use deterministic algorithm for reproducible results. Disable for potentially better quality but non-reproducible output.",
                    "backends": ["instant_meshes"],
                }),
                "crease_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Angle threshold (degrees) for preserving sharp/crease edges in Instant Meshes. 0 = no crease preservation.",
                    "backends": ["instant_meshes"],
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("remeshed_mesh", "info")
    FUNCTION = "remesh"
    CATEGORY = "geompack/remeshing"
    OUTPUT_NODE = True

    def remesh(self, trimesh, backend, target_edge_length=0.05, iterations=3,
               feature_angle=30.0, adaptive="false",
               protect_boundaries="true", voxel_size=0.02, target_face_count=500000,
               target_vertex_count=5000, deterministic="true", crease_angle=0.0,
               remesh_band=1.0):
        """
        Apply remeshing based on selected backend.

        Args:
            trimesh: Input trimesh.Trimesh object
            backend: Remeshing backend to use
            [other params]: Backend-specific parameters

        Returns:
            tuple: (remeshed_mesh, info_string)
        """
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        # Log backend and parameters
        print(f"\n{'='*60}")
        print(f"[Remesh] Backend: {backend}")
        print(f"[Remesh] Input: {initial_vertices:,} vertices, {initial_faces:,} faces")
        if backend == "pymeshlab_isotropic":
            print(f"[Remesh] Parameters: target_edge_length={target_edge_length}, iterations={iterations}, feature_angle={feature_angle}, adaptive={adaptive}")
        elif backend == "cgal_isotropic":
            print(f"[Remesh] Parameters: target_edge_length={target_edge_length}, iterations={iterations}, protect_boundaries={protect_boundaries}")
        elif backend == "blender_voxel":
            print(f"[Remesh] Parameters: voxel_size={voxel_size}")
        elif backend == "blender_quadriflow":
            print(f"[Remesh] Parameters: target_face_count={target_face_count:,}")
        elif backend == "instant_meshes":
            print(f"[Remesh] Parameters: target_vertex_count={target_vertex_count:,}, deterministic={deterministic}, crease_angle={crease_angle}")
        elif backend == "cumesh":
            print(f"[Remesh] Parameters: target_face_count={target_face_count:,}, remesh_band={remesh_band}")
        print(f"{'='*60}\n")

        if backend == "pymeshlab_isotropic":
            remeshed_mesh, info = self._pymeshlab_isotropic(
                trimesh, target_edge_length, iterations, feature_angle, adaptive
            )
        elif backend == "cgal_isotropic":
            remeshed_mesh, info = self._cgal_isotropic(
                trimesh, target_edge_length, iterations, protect_boundaries
            )
        elif backend == "blender_voxel":
            remeshed_mesh, info = self._blender_voxel(trimesh, voxel_size)
        elif backend == "blender_quadriflow":
            remeshed_mesh, info = self._blender_quadriflow(trimesh, target_face_count)
        elif backend == "instant_meshes":
            remeshed_mesh, info = self._instant_meshes(
                trimesh, target_vertex_count, deterministic, crease_angle
            )
        elif backend == "cumesh":
            remeshed_mesh, info = self._cumesh(
                trimesh, remesh_band, target_face_count
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        vertex_change = len(remeshed_mesh.vertices) - initial_vertices
        face_change = len(remeshed_mesh.faces) - initial_faces

        print(f"[Remesh] Output: {len(remeshed_mesh.vertices)} vertices ({vertex_change:+d}), "
              f"{len(remeshed_mesh.faces)} faces ({face_change:+d})")

        return {"ui": {"text": [info]}, "result": (remeshed_mesh, info)}

    def _pymeshlab_isotropic(self, trimesh, target_edge_length, iterations, feature_angle, adaptive):
        """PyMeshLab isotropic remeshing."""
        adaptive_bool = (adaptive == "true")
        remeshed_mesh, error = mesh_ops.pymeshlab_isotropic_remesh(
            trimesh, target_edge_length, iterations,
            adaptive=adaptive_bool, feature_angle=feature_angle
        )
        if remeshed_mesh is None:
            raise ValueError(f"PyMeshLab remeshing failed: {error}")

        info = f"""Remesh Results (PyMeshLab Isotropic):

Target Edge Length: {target_edge_length}
Iterations: {iterations}
Feature Angle: {feature_angle}°
Adaptive: {adaptive}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}
"""
        return remeshed_mesh, info

    def _cgal_isotropic(self, trimesh, target_edge_length, iterations, protect_boundaries):
        """CGAL isotropic remeshing."""
        protect = (protect_boundaries == "true")
        remeshed_mesh, error = mesh_ops.cgal_isotropic_remesh(
            trimesh, target_edge_length, iterations, protect
        )
        if remeshed_mesh is None:
            raise ValueError(f"CGAL remeshing failed: {error}")

        info = f"""Remesh Results (CGAL Isotropic):

Target Edge Length: {target_edge_length}
Iterations: {iterations}
Protect Boundaries: {protect_boundaries}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}
"""
        return remeshed_mesh, info

    def _blender_voxel(self, trimesh, voxel_size):
        """Blender voxel remeshing using direct bpy via comfy-env isolation."""
        from .._utils.bpy_worker import call_bpy

        print(f"[Remesh] Running Blender voxel remesh (bpy isolated, voxel_size={voxel_size})...")
        result = call_bpy('bpy_voxel_remesh',
            vertices=np.asarray(trimesh.vertices, dtype=np.float32),
            faces=np.asarray(trimesh.faces, dtype=np.int32),
            voxel_size=voxel_size
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=np.array(result['vertices'], dtype=np.float32),
            faces=np.array(result['faces'], dtype=np.int32),
            process=False
        )

        # Preserve metadata
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'blender_voxel',
            'voxel_size': voxel_size,
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces)
        }

        info = f"""Remesh Results (Blender Voxel):

Voxel Size: {voxel_size}
Method: bpy isolated

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}
"""
        return remeshed_mesh, info

    def _blender_quadriflow(self, trimesh, target_face_count):
        """Blender Quadriflow remeshing using direct bpy via comfy-env isolation."""
        from .._utils.bpy_worker import call_bpy

        print(f"[Remesh] Running Blender Quadriflow (bpy isolated, target_faces={target_face_count})...")
        result = call_bpy('bpy_quadriflow_remesh',
            vertices=np.asarray(trimesh.vertices, dtype=np.float32),
            faces=np.asarray(trimesh.faces, dtype=np.int32),
            target_face_count=target_face_count
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=np.array(result['vertices'], dtype=np.float32),
            faces=np.array(result['faces'], dtype=np.int32),
            process=False
        )

        # Preserve metadata
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'blender_quadriflow',
            'target_face_count': target_face_count,
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces)
        }

        info = f"""Remesh Results (Blender Quadriflow):

Target Face Count: {target_face_count:,}
Method: bpy isolated

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}

Quadriflow creates quad-dominant meshes with good topology.
"""
        return remeshed_mesh, info

    def _instant_meshes(self, trimesh, target_vertex_count, deterministic, crease_angle):
        """Instant Meshes field-aligned remeshing."""
        try:
            import pynanoinstantmeshes as pynano
        except ImportError:
            raise ImportError(
                "PyNanoInstantMeshes not installed. Install with: pip install PyNanoInstantMeshes"
            )

        V = trimesh.vertices.astype(np.float32)
        F = trimesh.faces.astype(np.uint32)

        V_out, F_out = pynano.remesh(
            V, F,
            vertex_count=target_vertex_count,
            deterministic=(deterministic == "true"),
            creaseAngle=crease_angle
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=V_out,
            faces=F_out,
            process=False
        )

        # Preserve metadata
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'instant_meshes',
            'target_vertex_count': target_vertex_count,
            'deterministic': deterministic == "true",
            'crease_angle': crease_angle,
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces)
        }

        info = f"""Remesh Results (Instant Meshes):

Target Vertex Count: {target_vertex_count:,}
Deterministic: {deterministic}
Crease Angle: {crease_angle}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}

Instant Meshes creates flow-aligned quad meshes.
"""
        return remeshed_mesh, info

    def _cumesh(self, trimesh, remesh_band, target_face_count):
        """CuMesh GPU dual-contouring remeshing (same algorithm as TRELLIS2)."""
        import torch
        import cumesh as CuMesh

        # Hardcoded resolution = 512 (same as TRELLIS2)
        grid_resolution = 512

        remeshed_mesh, error = mesh_ops.cumesh_dc_remesh(
            trimesh, grid_resolution, fill_holes_first=False, band=remesh_band
        )
        if remeshed_mesh is None:
            raise ValueError(f"CuMesh remeshing failed: {error}")

        # Simplify to target face count
        pre_simplify_faces = len(remeshed_mesh.faces)
        vertices = torch.tensor(remeshed_mesh.vertices, dtype=torch.float32).cuda()
        faces = torch.tensor(remeshed_mesh.faces, dtype=torch.int32).cuda()

        cumesh_obj = CuMesh.CuMesh()
        cumesh_obj.init(vertices, faces)

        # Skip pre-simplify unify on large meshes - CuMesh crashes on >2M faces
        # TRELLIS2 does unify here, but their mesh comes from a different path
        if len(faces) < 2_000_000:
            cumesh_obj.unify_face_orientations()
            print(f"[Remesh] Unified face orientations (pre-simplify)")
        else:
            print(f"[Remesh] Skipping pre-simplify unify (mesh too large: {len(faces):,} faces)")

        # Simplify to target
        cumesh_obj.simplify(target_face_count, verbose=True)
        print(f"[Remesh] After simplify: {cumesh_obj.num_faces:,} faces")

        # Unify after simplify (on smaller mesh, should work)
        cumesh_obj.unify_face_orientations()
        print(f"[Remesh] Unified face orientations (post-simplify)")

        final_verts, final_faces = cumesh_obj.read()
        remeshed_mesh = trimesh_module.Trimesh(
            vertices=final_verts.cpu().numpy(),
            faces=final_faces.cpu().numpy(),
            process=False
        )

        # Preserve metadata
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'cumesh',
            'remesh_band': remesh_band,
            'target_face_count': target_face_count,
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces)
        }

        info = f"""Remesh Results (CuMesh):

Band Width: {remesh_band}
Target Face Count: {target_face_count:,}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After Remesh: {pre_simplify_faces:,} faces
After Simplify: {len(remeshed_mesh.faces):,} faces

GPU-accelerated dual contouring (same algorithm as TRELLIS2).
"""
        return remeshed_mesh, info
