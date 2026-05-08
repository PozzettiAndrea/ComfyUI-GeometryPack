# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remove self-intersections by remeshing.
"""

import logging

import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class RemeshSelfIntersectionsNode(io.ComfyNode):
    """
    Remove self-intersections by remeshing.

    Uses libigl CGAL to subdivide self-intersecting triangles so that
    intersections lie exactly on edges. Can optionally extract outer hull
    for a clean manifold result. Essential for preparing meshes for
    boolean operations or 3D printing.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemeshSelfIntersections",
            display_name="Remesh Self Intersections",
            category="geompack/repair",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh"),
                io.Boolean.Input("detect_only", default=False),
                io.Boolean.Input("remove_unreferenced", default=True),
                io.Boolean.Input("extract_outer_hull", default=False),
                io.Boolean.Input("stitch_all", default=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="fixed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh, detect_only=False, remove_unreferenced=True,
                           extract_outer_hull=False, stitch_all=True):
        """
        Remesh self-intersections using libigl CGAL.

        Args:
            mesh: Input trimesh.Trimesh object
            detect_only: Only detect intersections, don't remesh
            remove_unreferenced: Remove unreferenced vertices after remeshing
            extract_outer_hull: Extract outer hull for manifold result (slow)
            stitch_all: Attempt to stitch all boundaries

        Returns:
            tuple: (remeshed_mesh, report_string)
        """
        log.info("Processing mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
        log.info("Options: detect_only=%s, remove_unreferenced=%s, extract_outer_hull=%s, stitch_all=%s",
                 detect_only, remove_unreferenced, extract_outer_hull, stitch_all)

        try:
            # Try to use libigl with CGAL
            import igl

            # Check if CGAL functions are available
            try:
                import igl.copyleft.cgal as cgal
                has_cgal = hasattr(cgal, 'remesh_self_intersections')
            except (ImportError, AttributeError):
                has_cgal = False

            if not has_cgal:
                error_msg = """Error: libigl CGAL not available

Self-intersection remeshing requires libigl with CGAL support.
Install with: pip install cgal

Returning mesh unchanged.
"""
                log.info("CGAL not available")
                return io.NodeOutput(mesh, error_msg, ui={"text": [error_msg]})

            log.info("Using libigl CGAL method")

            # Convert mesh to numpy arrays with proper dtypes
            V = np.asarray(mesh.vertices, dtype=np.float64)
            F = np.asarray(mesh.faces, dtype=np.int64)

            initial_vertices = len(V)
            initial_faces = len(F)

            # Perform remeshing with keyword arguments
            try:
                VV, FF, IF, J, IM = cgal.remesh_self_intersections(
                    V, F,
                    detect_only=detect_only,
                    first_only=False,
                    stitch_all=stitch_all
                )

                num_intersection_pairs = IF.shape[0] if IF is not None and hasattr(IF, 'shape') else 0

                if detect_only:
                    log.info("Detected %d intersection pairs", num_intersection_pairs)
                    result_mesh = mesh.copy()

                    if num_intersection_pairs > 0:
                        # Mark intersecting faces
                        intersecting_faces = np.unique(IF.flatten())
                        face_field = np.zeros(len(F), dtype=np.float32)
                        face_field[intersecting_faces] = 1.0
                        result_mesh.face_attributes['self_intersecting'] = face_field

                else:
                    log.info("Remeshing complete: %d vertices, %d faces", len(VV), len(FF))

                    # Post-processing
                    if remove_unreferenced and not detect_only:
                        log.info("Removing unreferenced vertices...")
                        VV_clean, FF_clean, _, _ = igl.remove_unreferenced(VV, FF)
                        log.info("After cleanup: %d vertices, %d faces", len(VV_clean), len(FF_clean))
                        VV, FF = VV_clean, FF_clean

                    if extract_outer_hull and not detect_only:
                        log.info("Extracting outer hull (this may take a while)...")
                        try:
                            # Try to extract outer hull for manifold result
                            if hasattr(igl, 'outer_hull_legacy'):
                                VV_hull, FF_hull, _, _ = igl.outer_hull_legacy(VV, FF)
                                log.info("Outer hull: %d vertices, %d faces", len(VV_hull), len(FF_hull))
                                VV, FF = VV_hull, FF_hull
                            else:
                                log.warning("outer_hull_legacy not available, skipping")
                        except Exception as e:
                            log.error("Outer hull extraction failed: %s", e)

                    # Create new trimesh from remeshed data
                    result_mesh = trimesh.Trimesh(vertices=VV, faces=FF, process=False)

                    # Store operation metadata
                    result_mesh.metadata['remeshed_self_intersections'] = True
                    result_mesh.metadata['original_vertices'] = initial_vertices
                    result_mesh.metadata['original_faces'] = initial_faces
                    result_mesh.metadata['intersections_found'] = num_intersection_pairs

                # Generate report
                final_vertices = len(result_mesh.vertices)
                final_faces = len(result_mesh.faces)
                added_vertices = final_vertices - initial_vertices
                added_faces = final_faces - initial_faces

                if detect_only:
                    report = f"""Self-Intersection Detection (detect_only mode):

Mesh Statistics:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

Detection Results:
  Intersection Pairs: {num_intersection_pairs:,}

Status:
  {'[OK] No self-intersections detected!' if num_intersection_pairs == 0 else '[WARN] Self-intersections found!'}

Note: Mesh was not modified (detect_only=True)
To fix intersections, set detect_only=False
"""
                else:
                    report = f"""Self-Intersection Remeshing:

Initial Mesh:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After Remeshing:
  Vertices: {final_vertices:,} ({'+' if added_vertices >= 0 else ''}{added_vertices:,})
  Faces: {final_faces:,} ({'+' if added_faces >= 0 else ''}{added_faces:,})

Processing:
  Intersection Pairs Found: {num_intersection_pairs:,}
  Removed Unreferenced: {'Yes' if remove_unreferenced else 'No'}
  Extracted Outer Hull: {'Yes' if extract_outer_hull else 'No'}
  Stitch All: {'Yes' if stitch_all else 'No'}

Status:
  {'[OK] Remeshing complete!' if not detect_only else '[OK] Detection complete!'}
  {'[OK] Mesh is now manifold' if extract_outer_hull and result_mesh.is_watertight else ''}

{'[WARN] Note: Remeshing subdivides intersections but may create non-manifold edges.' if not extract_outer_hull else ''}
{'  Consider enabling extract_outer_hull for a clean manifold result.' if not extract_outer_hull else ''}
"""

                return io.NodeOutput(result_mesh, report, ui={"text": [report]})

            except Exception as e:
                log.error("Remeshing error", exc_info=True)
                error_msg = f"""Error during remeshing:

{str(e)}

Returning mesh unchanged. Check console for details.
"""
                log.error("Remeshing error: %s", e)
                return io.NodeOutput(mesh, error_msg, ui={"text": [error_msg]})

        except ImportError as e:
            error_msg = f"""Error: libigl not available

{str(e)}

Self-intersection remeshing requires libigl with CGAL support.
Install with: pip install libigl cgal

Returning mesh unchanged.
"""
            log.error("libigl import error: %s", e)
            return io.NodeOutput(mesh, error_msg, ui={"text": [error_msg]})

        except Exception as e:
            log.error("Unexpected error", exc_info=True)
            error_msg = f"""Unexpected error:

{str(e)}

Returning mesh unchanged. Check console for details.
"""
            log.error("Unexpected error: %s", e)
            return io.NodeOutput(mesh, error_msg, ui={"text": [error_msg]})


NODE_CLASS_MAPPINGS = {
    "GeomPackRemeshSelfIntersections": RemeshSelfIntersectionsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemeshSelfIntersections": "Remesh Self Intersections",
}
