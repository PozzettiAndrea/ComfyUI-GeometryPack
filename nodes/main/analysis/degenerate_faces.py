# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Degenerate Faces Node - Detect truly degenerate faces and show smallest faces.

Degenerate faces have:
- Duplicate vertices (edge between same point, e.g., v0==v1)
- Zero area (collinear vertices)

Also shows the 30 smallest faces by area (for informational purposes).
"""

import os
import numpy as np


class DegenerateFacesNode:
    """
    Detect degenerate faces and show smallest faces by area.

    Truly degenerate faces have:
    - Duplicate vertices (collapsed edges)
    - Zero area (collinear points)

    Also displays the 30 smallest faces by area for inspection.
    """

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("trimesh", "degenerate_summary")
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_NODE = True
    FUNCTION = "find_degenerate_faces"
    CATEGORY = "geompack/analysis"

    def find_degenerate_faces(self, trimesh):
        """
        Find truly degenerate faces and smallest faces by area.

        Args:
            trimesh: Input trimesh object(s)

        Returns:
            dict with "result" tuple and "ui" data for display
        """
        # Handle batch input
        meshes = trimesh if isinstance(trimesh, list) else [trimesh]

        result_meshes = []
        summary_lines = []
        ui_data = []

        for mesh in meshes:
            faces = mesh.faces
            vertices = mesh.vertices
            num_faces = len(faces)

            degenerate_faces = []  # Truly degenerate (duplicate verts or zero area)
            all_face_areas = []    # All faces with their areas

            for face_idx, face in enumerate(faces):
                v0, v1, v2 = face

                # Check for duplicate vertices (collapsed edges)
                has_duplicate = (v0 == v1) or (v1 == v2) or (v2 == v0)

                # Get vertex positions
                p0 = vertices[v0]
                p1 = vertices[v1]
                p2 = vertices[v2]

                # Calculate area using cross product
                edge1 = p1 - p0
                edge2 = p2 - p0
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)

                all_face_areas.append({
                    "id": face_idx,
                    "area": float(area),
                    "vertices": face.tolist()
                })

                # Only flag truly degenerate faces
                if has_duplicate:
                    degenerate_faces.append({
                        "id": face_idx,
                        "area": float(area),
                        "vertices": face.tolist(),
                        "reason": "duplicate_vertex"
                    })
                elif area == 0:
                    degenerate_faces.append({
                        "id": face_idx,
                        "area": float(area),
                        "vertices": face.tolist(),
                        "reason": "zero_area"
                    })

            # Sort all faces by area to get smallest
            all_face_areas.sort(key=lambda x: x["area"])
            smallest_faces = all_face_areas[:30]

            num_degenerate = len(degenerate_faces)

            # Count by type
            duplicate_count = sum(1 for d in degenerate_faces if d["reason"] == "duplicate_vertex")
            zero_count = sum(1 for d in degenerate_faces if d["reason"] == "zero_area")

            # Create face attribute (only truly degenerate)
            degenerate_field = np.zeros(num_faces, dtype=np.float32)
            for info in degenerate_faces:
                degenerate_field[info["id"]] = 1.0

            # Get mesh name
            mesh_name = mesh.metadata.get('file_name', 'mesh') if hasattr(mesh, 'metadata') else 'mesh'
            mesh_name_short = os.path.splitext(mesh_name)[0]

            # Build summary
            detail_lines = [f"{mesh_name_short}: {num_degenerate} degenerate face(s)"]

            if duplicate_count > 0:
                detail_lines.append(f"  {duplicate_count} with duplicate vertices")
            if zero_count > 0:
                detail_lines.append(f"  {zero_count} with zero area")

            if num_degenerate > 0:
                detail_lines.append("")
                detail_lines.append("Degenerate faces:")
                for info in degenerate_faces[:30]:
                    detail_lines.append(f"  Face {info['id']}: area={info['area']:.2e} ({info['reason']})")

            detail_lines.append("")
            detail_lines.append("Smallest 30 faces by area:")
            for info in smallest_faces:
                detail_lines.append(f"  Face {info['id']}: area={info['area']:.2e}")

            summary_lines.append("\n".join(detail_lines))

            # UI data
            ui_data.append({
                "mesh_name": mesh_name_short,
                "num_degenerate": num_degenerate,
                "duplicate_count": duplicate_count,
                "zero_count": zero_count,
                "total_faces": num_faces,
                "total_vertices": len(vertices),
                "degenerate_faces": degenerate_faces[:30],
                "smallest_faces": smallest_faces
            })

            # Console output
            print(f"[DegenerateFaces] {mesh_name_short}: {num_degenerate} degenerate faces ({duplicate_count} duplicate verts, {zero_count} zero area)")
            if num_degenerate > 0:
                for info in degenerate_faces[:5]:
                    print(f"[DegenerateFaces]   Face {info['id']}: {info['reason']}")
                if num_degenerate > 5:
                    print(f"[DegenerateFaces]   ... and {num_degenerate - 5} more")
            print(f"[DegenerateFaces] Smallest face area: {smallest_faces[0]['area']:.2e}")

            # Store result
            result_mesh = mesh.copy()
            result_mesh.face_attributes['degenerate'] = degenerate_field

            if not hasattr(result_mesh, 'metadata'):
                result_mesh.metadata = {}
            result_mesh.metadata['num_degenerate_faces'] = num_degenerate
            result_mesh.metadata['degenerate_face_ids'] = [d["id"] for d in degenerate_faces]

            result_meshes.append(result_mesh)

        summary = "\n\n".join(summary_lines)

        print(f"[DegenerateFaces] Processed {len(meshes)} mesh(es)")

        return {
            "result": (result_meshes, summary),
            "ui": {
                "degenerate_data": ui_data
            }
        }


NODE_CLASS_MAPPINGS = {
    "GeomPackDegenerateFaces": DegenerateFacesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackDegenerateFaces": "Degenerate Faces",
}
