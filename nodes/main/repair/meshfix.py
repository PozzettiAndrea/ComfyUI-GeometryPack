# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
MeshFix Node - Automatic mesh repair using pymeshfix.

Closes holes, removes self-intersections, and creates watertight meshes
with light touch-ups.
"""

import numpy as np
import trimesh
import pymeshfix


class MeshFixNode:
    """
    Automatic mesh repair using MeshFix algorithm.

    Performs light touch-up repairs:
    - Remove small isolated components
    - Join nearby disconnected parts
    - Fill boundary holes
    - Remove self-intersections and degenerate faces

    Based on the MeshFix library by Marco Attene.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mesh": ("TRIMESH",),
            },
            "optional": {
                "remove_small_components": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Remove small isolated mesh fragments before repair"
                }),
                "join_components": (["true", "false"], {
                    "default": "false",
                    "tooltip": "Attempt to join nearby disconnected components"
                }),
                "fill_holes": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Fill boundary holes in the mesh"
                }),
                "max_hole_edges": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 10,
                    "tooltip": "Max edges for holes to fill. 0 = fill all holes regardless of size"
                }),
                "refine_holes": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Refine triangulation when filling holes for better quality"
                }),
                "clean_mesh": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Remove self-intersections and degenerate faces"
                }),
                "clean_iterations": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Max iterations for self-intersection removal"
                }),
                "inner_loops": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Inner loops per clean iteration"
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("repaired_mesh", "report")
    FUNCTION = "repair_mesh"
    CATEGORY = "geompack/repair"
    OUTPUT_NODE = True

    def repair_mesh(
        self,
        input_mesh,
        remove_small_components="true",
        join_components="false",
        fill_holes="true",
        max_hole_edges=0,
        refine_holes="true",
        clean_mesh="true",
        clean_iterations=10,
        inner_loops=3
    ):
        """
        Repair mesh using MeshFix algorithm.

        Args:
            input_mesh: Input trimesh.Trimesh object
            remove_small_components: Remove isolated fragments
            join_components: Join nearby components
            fill_holes: Fill boundary holes
            max_hole_edges: Max hole size (0 = all)
            refine_holes: Refine hole triangulation
            clean_mesh: Remove self-intersections
            clean_iterations: Clean iteration count
            inner_loops: Inner loops per iteration

        Returns:
            tuple: (repaired_trimesh, report_string)
        """
        # Convert string bools
        remove_small_components = remove_small_components == "true"
        join_components = join_components == "true"
        fill_holes = fill_holes == "true"
        refine_holes = refine_holes == "true"
        clean_mesh = clean_mesh == "true"

        # Log input
        print(f"\n{'='*60}")
        print(f"[MeshFix] Input: {len(input_mesh.vertices):,} vertices, {len(input_mesh.faces):,} faces")
        print(f"[MeshFix] Options: remove_small={remove_small_components}, join={join_components}, fill_holes={fill_holes}, clean={clean_mesh}")
        print(f"{'='*60}\n")

        # Track initial state
        initial_vertices = len(input_mesh.vertices)
        initial_faces = len(input_mesh.faces)
        was_watertight = input_mesh.is_watertight

        # Convert to numpy arrays
        v = np.asarray(input_mesh.vertices, dtype=np.float64)
        f = np.asarray(input_mesh.faces, dtype=np.int32)

        # Create PyTMesh instance
        tin = pymeshfix.PyTMesh()
        tin.load_array(v, f)

        # Track operations
        operations = []

        # Get initial boundary count
        try:
            initial_boundaries = tin.boundaries()
        except:
            initial_boundaries = -1

        # Apply repairs in order
        if remove_small_components:
            print("[MeshFix] Removing small components...")
            tin.remove_smallest_components()
            operations.append("Removed small components")

        if join_components:
            print("[MeshFix] Joining nearby components...")
            tin.join_closest_components()
            operations.append("Joined nearby components")

        if fill_holes:
            # 0 means fill all holes - use large number since pymeshfix requires int
            nbe = max_hole_edges if max_hole_edges > 0 else 100000
            print(f"[MeshFix] Filling holes (max_edges={nbe}, refine={refine_holes})...")
            tin.fill_small_boundaries(nbe=nbe, refine=refine_holes)
            operations.append(f"Filled holes (max_edges={'all' if max_hole_edges == 0 else nbe})")

        if clean_mesh:
            print(f"[MeshFix] Cleaning mesh (iterations={clean_iterations}, inner_loops={inner_loops})...")
            tin.clean(max_iters=clean_iterations, inner_loops=inner_loops)
            operations.append(f"Cleaned (iters={clean_iterations})")

        # Get final boundary count
        try:
            final_boundaries = tin.boundaries()
        except:
            final_boundaries = -1

        # Extract result
        vclean, fclean = tin.return_arrays()

        # Create result mesh
        result_mesh = trimesh.Trimesh(
            vertices=vclean,
            faces=fclean,
            process=False
        )

        # Copy metadata if present
        if hasattr(input_mesh, 'metadata') and input_mesh.metadata:
            result_mesh.metadata = input_mesh.metadata.copy()

        # Final stats
        final_vertices = len(result_mesh.vertices)
        final_faces = len(result_mesh.faces)
        is_watertight = result_mesh.is_watertight

        vertex_diff = final_vertices - initial_vertices
        face_diff = final_faces - initial_faces

        # Build report
        report = f"""MeshFix Repair Report
{'='*40}

Operations Performed:
{chr(10).join(f'  - {op}' for op in operations) if operations else '  (none)'}

Before:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}
  Watertight: {'Yes' if was_watertight else 'No'}
  Boundaries: {initial_boundaries if initial_boundaries >= 0 else 'unknown'}

After:
  Vertices: {final_vertices:,} ({'+' if vertex_diff >= 0 else ''}{vertex_diff})
  Faces: {final_faces:,} ({'+' if face_diff >= 0 else ''}{face_diff})
  Watertight: {'Yes' if is_watertight else 'No'}
  Boundaries: {final_boundaries if final_boundaries >= 0 else 'unknown'}

Status: {'Mesh is now watertight!' if is_watertight and not was_watertight else 'Mesh was already watertight.' if was_watertight else 'Mesh still has open boundaries.'}
"""

        print(f"[MeshFix] Result: {final_vertices:,} vertices, {final_faces:,} faces")
        print(f"[MeshFix] Watertight: {was_watertight} -> {is_watertight}")

        return {"ui": {"text": [report]}, "result": (result_mesh, report)}


NODE_CLASS_MAPPINGS = {
    "GeomPackMeshFix": MeshFixNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackMeshFix": "MeshFix",
}
