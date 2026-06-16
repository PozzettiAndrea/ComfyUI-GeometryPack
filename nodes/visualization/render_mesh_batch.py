# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Preview Mesh Batch Render - render each mesh in a TRIMESH batch with PyVista
(offscreen) and return an IMAGE batch. Choose resolution; the mesh filename is
drawn on each render as the plotter title. Optional GPU (native/EGL) offscreen
rendering for speed (falls back to software), with a tqdm progress bar.

Mesh name comes from trimesh metadata['file_name'] (set by Load Mesh); falls
back to "mesh <i>".
"""

import logging

from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _setup_offscreen(use_gpu):
    """Prepare a PyVista offscreen backend.

    use_gpu=True  -> try native/EGL offscreen (GPU) with no virtual X display.
    use_gpu=False -> force software (xvfb / OSMesa).
    Falls back GPU->software on failure. Returns a short backend label for logging.
    """
    import os
    import pyvista as pv
    pv.OFF_SCREEN = True

    def _probe():
        try:
            p = pv.Plotter(off_screen=True, window_size=[64, 64])
            p.add_mesh(pv.Sphere())
            p.show(auto_close=False)
            cls = ""
            try:
                cls = p.render_window.GetClassName()
            except Exception:
                pass
            p.close()
            return cls or "ok"
        except Exception as e:
            log.warning("[PreviewMeshBatch] offscreen probe failed: %s", e)
            return None

    if use_gpu:
        cls = _probe()           # no xvfb -> VTK uses EGL/native if built for it
        if cls:
            return f"gpu/native ({cls})"
        log.info("[PreviewMeshBatch] GPU offscreen unavailable; using software")

    if os.name != "nt" and not os.environ.get("DISPLAY"):
        try:
            pv.start_xvfb()
        except Exception as e:
            log.debug("[PreviewMeshBatch] start_xvfb: %s", e)
    cls = _probe()
    return f"software ({cls})" if cls else "none"


def _mesh_title(mesh, idx):
    name = None
    try:
        name = (mesh.metadata or {}).get("file_name")
    except Exception:
        pass
    return str(name) if name else f"mesh {idx}"


class RenderMeshBatch(io.ComfyNode):
    """Render each mesh in a batch (PyVista offscreen), titled by filename."""

    INPUT_IS_LIST = True

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackPreviewMeshBatchRender",
            display_name="Preview Mesh Batch Render",
            category="geompack/visualization",
            inputs=[
                io.Custom("TRIMESH").Input("meshes", tooltip="Batch/list of meshes to render."),
                io.Int.Input("resolution", default=512, min=64, max=4096, step=16,
                    tooltip="Square render size (px)."),
                io.Combo.Input("background", options=["white", "black", "gray"], default="white"),
                io.Boolean.Input("show_edges", default=False, tooltip="Overlay wireframe edges."),
                io.Boolean.Input("show_title", default=True,
                    tooltip="Draw the mesh filename as the plotter title on each render."),
                io.Boolean.Input("use_gpu", default=True,
                    tooltip="Try GPU/native (EGL) offscreen rendering for speed; falls back to "
                            "software (xvfb/OSMesa) if unavailable."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.String.Output(display_name="titles"),
            ],
        )

    @classmethod
    def execute(cls, meshes, resolution=512, background="white", show_edges=False,
                show_title=True, use_gpu=True):
        import numpy as np
        import torch
        import pyvista as pv

        def _scalar(x, d):
            return (x[0] if x else d) if isinstance(x, list) else x
        res = int(_scalar(resolution, 512))
        bg = _scalar(background, "white")
        edges = bool(_scalar(show_edges, False))
        titled = bool(_scalar(show_title, True))
        gpu = bool(_scalar(use_gpu, True))

        if len(meshes) == 1 and isinstance(meshes[0], list):
            meshes = meshes[0]
        meshes = [m for m in meshes if m is not None]
        if not meshes:
            raise ValueError("Preview Mesh Batch Render: no meshes provided.")

        backend = _setup_offscreen(gpu)
        log.info("[PreviewMeshBatch] backend: %s | %d mesh(es) @ %dpx", backend, len(meshes), res)

        bg_rgb = {"white": "white", "black": "black", "gray": (0.5, 0.5, 0.5)}.get(bg, "white")
        text_color = "black" if bg == "white" else "white"

        # progress: tqdm (console) + ComfyUI bar (UI), both best-effort.
        try:
            from tqdm import tqdm
            iterator = tqdm(list(enumerate(meshes)), desc="Preview Mesh Batch", unit="mesh")
        except Exception:
            iterator = enumerate(meshes)
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(len(meshes))
        except Exception:
            pbar = None

        imgs, titles = [], []
        for i, mesh in iterator:
            title = _mesh_title(mesh, i)
            titles.append(title)
            try:
                poly = pv.wrap(mesh)
                p = pv.Plotter(off_screen=True, window_size=[res, res])
                p.background_color = bg_rgb
                p.add_mesh(poly, color="lightgray", show_edges=edges,
                           edge_color="dimgray", smooth_shading=True)
                if titled:
                    p.add_title(title, font_size=max(8, int(res / 45)), color=text_color)
                p.view_isometric()
                shot = p.screenshot(return_img=True)
                p.close()
                arr = np.asarray(shot, dtype=np.float32) / 255.0
                if arr.ndim == 3 and arr.shape[-1] == 4:
                    arr = arr[..., :3]
                imgs.append(arr)
            except Exception as e:
                log.error("[PreviewMeshBatch] render failed for %s: %s", title, e)
                imgs.append(np.ones((res, res, 3), np.float32))
            if pbar is not None:
                pbar.update(1)

        h = max(a.shape[0] for a in imgs)
        w = max(a.shape[1] for a in imgs)
        batch = np.stack([
            np.pad(a, ((0, h - a.shape[0]), (0, w - a.shape[1]), (0, 0)), constant_values=1.0)
            for a in imgs
        ], axis=0)
        images = torch.from_numpy(batch)
        return io.NodeOutput(images, " | ".join(titles))


NODE_CLASS_MAPPINGS = {"GeomPackPreviewMeshBatchRender": RenderMeshBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackPreviewMeshBatchRender": "Preview Mesh Batch Render"}
