# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Fill holes in mesh by adding new faces.
"""

import trimesh
import numpy as np

try:
    import cumesh as CuMesh
    import torch
    HAS_CUMESH = True
except ImportError:
    HAS_CUMESH = False

try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False

try:
    import igl
    HAS_IGL = True
except ImportError:
    HAS_IGL = False


class FillHolesNode:
    """
    Fill holes in mesh by adding new faces.

    Identifies boundary loops (holes) and fills them with new triangular faces.
    Useful for mesh repair, creating watertight meshes for 3D printing, or
    closing gaps in scanned geometry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "method": (["cumesh", "trimesh", "pymeshlab", "igl_fan"], {"default": "cumesh"}),
            },
            "optional": {
                # CuMesh parameter (same as TRELLIS2 fill_holes_perimeter)
                "perimeter": ("FLOAT", {
                    "default": 0.03,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.001,
                    "visible_when": {"method": ["cumesh"]},
                }),
                # PyMeshLab parameter
                "maxholesize": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 100000,
                    "step": 100,
                    "visible_when": {"method": ["pymeshlab"]},
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("filled_mesh", "info")
    OUTPUT_NODE = True
    FUNCTION = "fill_holes"
    CATEGORY = "geompack/repair"

    def fill_holes(self, mesh, method="cumesh", perimeter=0.03, maxholesize=1000):
        """
        Fill holes in the mesh.

        Args:
            mesh: Input trimesh.Trimesh object
            method: Hole filling method ("cumesh", "trimesh", "pymeshlab", or "igl_fan")
            perimeter: Maximum hole perimeter to fill (used by cumesh, default matches TRELLIS2)

        Returns:
            tuple: (filled_trimesh, info_string)
        """
        # Log method and parameters
        print(f"\n{'='*60}")
        print(f"[FillHoles] Method: {method}")
        print(f"[FillHoles] Input: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
        if method == "cumesh":
            print(f"[FillHoles] Parameters: perimeter={perimeter}")
        elif method == "pymeshlab":
            print(f"[FillHoles] Parameters: maxholesize={maxholesize}")
        elif method in ["trimesh", "igl_fan"]:
            print(f"[FillHoles] Parameters: (none)")
        print(f"{'='*60}\n")

        # Check initial state
        was_watertight = mesh.is_watertight
        initial_vertices = len(mesh.vertices)
        initial_faces = len(mesh.faces)

        # Create a copy
        filled_mesh = mesh.copy()

        # Track method actually used (for fallback cases)
        method_used = method
        num_holes_filled = None

        # Fill holes using selected method
        if method == "cumesh" and HAS_CUMESH:
            # GPU-accelerated hole filling (same as TRELLIS2)
            vertices = torch.tensor(filled_mesh.vertices, dtype=torch.float32).cuda()
            faces = torch.tensor(filled_mesh.faces, dtype=torch.int32).cuda()

            # Initialize CuMesh
            cumesh_obj = CuMesh.CuMesh()
            cumesh_obj.init(vertices, faces)

            # Fill holes with perimeter limit
            cumesh_obj.fill_holes(max_hole_perimeter=perimeter)

            # Read back result
            final_verts, final_faces = cumesh_obj.read()
            filled_mesh = trimesh.Trimesh(
                vertices=final_verts.cpu().numpy(),
                faces=final_faces.cpu().numpy(),
                process=False
            )

            print(f"[FillHoles] CuMesh method completed (perimeter={perimeter})")

        elif method == "cumesh" and not HAS_CUMESH:
            # Fallback to trimesh if cumesh not available
            print(f"[FillHoles] CuMesh not available, falling back to trimesh method")
            filled_mesh.fill_holes()
            method_used = "trimesh (fallback)"

        elif method == "pymeshlab" and HAS_PYMESHLAB:
            # Use PyMeshLab's hole closing
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(
                vertex_matrix=filled_mesh.vertices,
                face_matrix=filled_mesh.faces
            ))

            # Close holes (pymeshlab uses edge count, use large default)
            ms.meshing_close_holes(maxholesize=maxholesize)

            # Extract result
            m = ms.current_mesh()
            filled_mesh = trimesh.Trimesh(
                vertices=m.vertex_matrix(),
                faces=m.face_matrix(),
                process=False
            )

            print(f"[FillHoles] PyMeshLab method completed")

        elif method == "pymeshlab" and not HAS_PYMESHLAB:
            # Fallback to trimesh if pymeshlab not available
            print(f"[FillHoles] PyMeshLab not available, falling back to trimesh method")
            filled_mesh.fill_holes()
            method_used = "trimesh (fallback)"

        elif method == "igl_fan" and HAS_IGL:
            # Use libigl to find boundary and fill with fan triangulation
            V = np.asarray(filled_mesh.vertices, dtype=np.float64)
            F = np.asarray(filled_mesh.faces, dtype=np.int32)

            # Get boundary loop (returns single loop as 1D array)
            try:
                loop = igl.boundary_loop(F)

                # Check if we got a valid loop
                if isinstance(loop, np.ndarray) and loop.size > 0 and loop.ndim == 1:
                    if len(loop) >= 3:
                        # Create fan triangulation from first vertex
                        new_faces = []
                        center_idx = loop[0]
                        for i in range(1, len(loop) - 1):
                            new_faces.append([center_idx, loop[i], loop[i + 1]])

                        # Add new faces to mesh
                        if new_faces:
                            all_faces = np.vstack([F, np.array(new_faces, dtype=np.int32)])
                            filled_mesh = trimesh.Trimesh(
                                vertices=V,
                                faces=all_faces,
                                process=False
                            )
                            num_holes_filled = 1
                            print(f"[FillHoles] igl_fan filled boundary loop with {len(new_faces)} faces")
                    else:
                        print(f"[FillHoles] Boundary loop too small ({len(loop)} vertices)")
                else:
                    print(f"[FillHoles] No boundary loop found or invalid format")
            except Exception as e:
                print(f"[FillHoles] igl boundary_loop failed: {e}, using trimesh fallback")
                filled_mesh.fill_holes()
                method_used = "trimesh (igl error fallback)"

        elif method == "igl_fan" and not HAS_IGL:
            # Fallback to trimesh if igl not available
            print(f"[FillHoles] libigl not available, falling back to trimesh method")
            filled_mesh.fill_holes()
            method_used = "trimesh (fallback)"

        else:
            # Use trimesh's built-in method
            filled_mesh.fill_holes()
            print(f"[FillHoles] Trimesh method completed")

        # Check result
        is_watertight = filled_mesh.is_watertight
        final_vertices = len(filled_mesh.vertices)
        final_faces = len(filled_mesh.faces)

        added_vertices = final_vertices - initial_vertices
        added_faces = final_faces - initial_faces

        # Build holes info
        holes_info = ""
        if num_holes_filled is not None:
            holes_info = f"\n  Holes Filled: {num_holes_filled}"

        # Build param info based on method
        if "cumesh" in method_used.lower():
            param_info = f"Max Hole Perimeter: {perimeter}"
        elif "pymeshlab" in method_used.lower():
            param_info = f"Max Hole Size: {maxholesize}"
        else:
            param_info = "(no params)"

        info = f"""Hole Filling Results:

Method: {method_used}
{param_info}

Initial State:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}
  Watertight: {'Yes' if was_watertight else 'No'}

After Filling:
  Vertices: {final_vertices:,} (+{added_vertices})
  Faces: {final_faces:,} (+{added_faces})
  Watertight: {'✓ Yes' if is_watertight else '⚠ No'}{holes_info}

{'✓ All holes successfully filled!' if is_watertight and added_faces > 0 else ''}
{'ℹ No holes detected - mesh was already watertight.' if was_watertight else ''}
{'⚠ Some holes may remain (check mesh topology).' if not is_watertight and added_faces > 0 else ''}
"""

        print(f"[FillHoles] Added {added_faces} faces, Watertight: {was_watertight} -> {is_watertight}")

        return {
            "result": (filled_mesh, info),
            "ui": {"text": [info]}
        }


NODE_CLASS_MAPPINGS = {
    "GeomPackFillHoles": FillHolesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackFillHoles": "Fill Holes",
}
