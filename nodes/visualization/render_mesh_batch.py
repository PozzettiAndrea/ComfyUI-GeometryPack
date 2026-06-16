# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Render Mesh Batch - render each mesh in a batch with PyVista (offscreen) and
stamp the mesh name as a title. Returns an IMAGE batch (one render per mesh).

The mesh name is taken from trimesh metadata['file_name'] (set by Load Mesh);
falls back to "mesh <i>" if absent.
"""

import logging

from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _ensure_offscreen_backend():
    """Best-effort headless GL setup for PyVista offscreen rendering."""
    import os
    import pyvista as pv
    pv.OFF_SCREEN = True
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        try:
            pv.start_xvfb()
        except Exception as e:
            log.debug("start_xvfb failed (may already have EGL/OSMesa): %s", e)


def _mesh_title(mesh, idx):
    name = None
    try:
        name = (mesh.metadata or {}).get("file_name")
    except Exception:
        pass
    return str(name) if name else f"mesh {idx}"


class RenderMeshBatch(io.ComfyNode):
    """Render each mesh in a batch (PyVista, offscreen), titled by mesh name."""

    INPUT_IS_LIST = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRenderMeshBatch",
            display_name="Render Mesh Batch",
            category="geompack/visualization",
            inputs=[
                io.Custom("TRIMESH").Input("meshes", tooltip="Batch/list of meshes to render."),
                io.Int.Input("resolution", default=512, min=64, max=4096, step=16,
                    tooltip="Square render size (px)."),
                io.Combo.Input("background", options=["white", "black", "gray"], default="white"),
                io.Boolean.Input("show_edges", default=False,
                    tooltip="Overlay wireframe edges."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.String.Output(display_name="titles"),
            ],
        )

    @classmethod
    def execute(cls, meshes, resolution=512, background="white", show_edges=False):
        import numpy as np
        import torch
        import pyvista as pv

        # INPUT_IS_LIST wraps every arg in a list; unwrap scalars and the mesh list.
        def _scalar(x, default):
            if isinstance(x, list):
                return x[0] if x else default
            return x
        res = int(_scalar(resolution, 512))
        bg = _scalar(background, "white")
        edges = bool(_scalar(show_edges, False))

        # ComfyUI may hand us [[m1, m2, ...]] or [m1, m2, ...]
        if len(meshes) == 1 and isinstance(meshes[0], list):
            meshes = meshes[0]
        meshes = [m for m in meshes if m is not None]
        if not meshes:
            raise ValueError("Render Mesh Batch: no meshes provided.")

        _ensure_offscreen_backend()
        bg_rgb = {"white": "white", "black": "black", "gray": (0.5, 0.5, 0.5)}.get(bg, "white")
        text_color = "black" if bg == "white" else "white"

        imgs, titles = [], []
        for i, mesh in enumerate(meshes):
            title = _mesh_title(mesh, i)
            titles.append(title)
            try:
                poly = pv.wrap(mesh)  # trimesh.Trimesh -> PolyData
                p = pv.Plotter(off_screen=True, window_size=[res, res])
                p.background_color = bg_rgb
                p.add_mesh(poly, color="lightgray", show_edges=edges,
                           edge_color="dimgray", smooth_shading=True)
                p.add_text(title, position="upper_edge", font_size=int(max(8, res / 40)),
                           color=text_color)
                p.view_isometric()
                shot = p.screenshot(return_img=True)
                p.close()
                arr = np.asarray(shot, dtype=np.float32) / 255.0
                if arr.ndim == 3 and arr.shape[-1] == 4:
                    arr = arr[..., :3]
                imgs.append(arr)
            except Exception as e:
                log.error("Render failed for %s: %s", title, e)
                imgs.append(np.ones((res, res, 3), np.float32))  # blank placeholder

        # pad to common size (defensive; screenshots should already match)
        h = max(a.shape[0] for a in imgs)
        w = max(a.shape[1] for a in imgs)
        batch = np.stack([
            np.pad(a, ((0, h - a.shape[0]), (0, w - a.shape[1]), (0, 0)), constant_values=1.0)
            for a in imgs
        ], axis=0)
        images = torch.from_numpy(batch)
        summary = " | ".join(titles)
        log.info("Render Mesh Batch: %d mesh(es) @ %dx%d", len(imgs), w, h)
        return io.NodeOutput(images, summary)


NODE_CLASS_MAPPINGS = {"GeomPackRenderMeshBatch": RenderMeshBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackRenderMeshBatch": "Render Mesh Batch"}
