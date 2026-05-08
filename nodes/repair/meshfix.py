# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
MeshFix Node - Automatic mesh repair using pymeshfix.

Closes holes, removes self-intersections, and creates watertight meshes
with light touch-ups.
"""

import logging

import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")

class MeshFixNode(io.ComfyNode):
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
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackMeshFix",
            display_name="MeshFix",
            category="geompack/repair",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("input_mesh"),
                io.Combo.Input("remove_small_components", options=["true", "false"], default="true", tooltip="Remove small isolated mesh fragments before repair", optional=True),
                io.Combo.Input("join_components", options=["true", "false"], default="false", tooltip="Attempt to join nearby disconnected components", optional=True),
                io.Combo.Input("fill_holes", options=["true", "false"], default="true", tooltip="Fill boundary holes in the mesh", optional=True),
                io.Int.Input("max_hole_edges", default=0, min=0, max=10000, step=10, tooltip="Max edges for holes to fill. 0 = fill all holes regardless of size", optional=True),
                io.Combo.Input("refine_holes", options=["true", "false"], default="true", tooltip="Refine triangulation when filling holes for better quality", optional=True),
                io.Combo.Input("clean_mesh", options=["true", "false"], default="true", tooltip="Remove self-intersections and degenerate faces", optional=True),
                io.Int.Input("clean_iterations", default=10, min=1, max=100, step=1, tooltip="Max iterations for self-intersection removal", optional=True),
                io.Int.Input("inner_loops", default=3, min=1, max=10, step=1, tooltip="Inner loops per clean iteration", optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="repaired_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(
        cls,
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
        log.info("Input: %d vertices, %d faces", len(input_mesh.vertices), len(input_mesh.faces))
        log.info("Options: remove_small=%s, join=%s, fill_holes=%s, clean=%s",
                 remove_small_components, join_components, fill_holes, clean_mesh)

        # Track initial state
        initial_vertices = len(input_mesh.vertices)
        initial_faces = len(input_mesh.faces)
        was_watertight = input_mesh.is_watertight

        # Convert to numpy arrays
        v = np.asarray(input_mesh.vertices, dtype=np.float64)
        f = np.asarray(input_mesh.faces, dtype=np.int32)

        try:
            import pymeshfix
        except (ImportError, OSError):
            raise ImportError("pymeshfix is required. Install with: pip install pymeshfix")

        # Create PyTMesh instance
        tin = pymeshfix.PyTMesh()
        tin.load_array(v, f)

        # Track operations
        operations = []

        # Get initial boundary count
        try:
            initial_boundaries = tin.boundaries()
        except Exception as e:
            log.debug("Failed to get initial boundary count: %s", e)
            initial_boundaries = -1

        # Apply repairs in order
        if remove_small_components:
            log.info("Removing small components...")
            tin.remove_smallest_components()
            operations.append("Removed small components")

        if join_components:
            log.info("Joining nearby components...")
            tin.join_closest_components()
            operations.append("Joined nearby components")

        if fill_holes:
            # 0 means fill all holes - use large number since pymeshfix requires int
            nbe = max_hole_edges if max_hole_edges > 0 else 100000
            log.info("Filling holes (max_edges=%d, refine=%s)...", nbe, refine_holes)
            tin.fill_small_boundaries(nbe=nbe, refine=refine_holes)
            operations.append(f"Filled holes (max_edges={'all' if max_hole_edges == 0 else nbe})")

        if clean_mesh:
            log.info("Cleaning mesh (iterations=%d, inner_loops=%d)...", clean_iterations, inner_loops)
            tin.clean(max_iters=clean_iterations, inner_loops=inner_loops)
            operations.append(f"Cleaned (iters={clean_iterations})")

        # Get final boundary count
        try:
            final_boundaries = tin.boundaries()
        except Exception as e:
            log.debug("Failed to get final boundary count: %s", e)
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

        log.info("Result: %d vertices, %d faces", final_vertices, final_faces)
        log.info("Watertight: %s -> %s", was_watertight, is_watertight)

        return io.NodeOutput(result_mesh, report, ui={"text": [report]})

NODE_CLASS_MAPPINGS = {
    "GeomPackMeshFix": MeshFixNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackMeshFix": "MeshFix",
}
