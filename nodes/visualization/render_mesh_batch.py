# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Preview Mesh Batch Render - render each mesh in a TRIMESH batch with PyVista
(offscreen) and return an IMAGE batch. The mesh filename is drawn as the title.

Options:
  * resolution, background, mesh_opacity
  * show_edges (+ edge_opacity, shown only when edges are on)
  * show_connected_components -> color by pyvista connectivity() RegionId
  * revert_yz_flip -> swap Y/Z axes (undo a y/z flip)
  * three_plane    -> 2x2 grid: iso(random) view, then XY, YZ, XZ
  * use_gpu        -> try native/EGL offscreen (fast), software fallback
Progress shown via tqdm + the ComfyUI bar.
"""

import logging

from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _setup_offscreen(use_gpu):
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
        cls = _probe()
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


def _add_geometry(p, poly, mesh_opacity, edges, edge_opacity, show_cc):
    """Add the surface (optionally colored by connected component) + optional edges."""
    if show_cc:
        try:
            cc = poly.connectivity()  # adds 'RegionId'
            p.add_mesh(cc, scalars="RegionId", cmap="tab20", show_scalar_bar=False,
                       opacity=mesh_opacity, smooth_shading=True)
            base = cc
        except Exception as e:
            log.warning("[PreviewMeshBatch] connectivity() failed: %s", e)
            p.add_mesh(poly, color="lightgray", opacity=mesh_opacity, smooth_shading=True)
            base = poly
    else:
        p.add_mesh(poly, color="lightgray", opacity=mesh_opacity, smooth_shading=True)
        base = poly
    if edges:
        try:
            p.add_mesh(base.extract_all_edges(), color="dimgray",
                       opacity=edge_opacity, line_width=1)
        except Exception:
            pass


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
                io.Float.Input("mesh_opacity", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Opacity of the mesh surface."),
                io.Boolean.Input("show_edges", default=False, tooltip="Overlay wireframe edges."),
                io.Float.Input("edge_opacity", default=1.0, min=0.0, max=1.0, step=0.05, optional=True,
                    tooltip="Opacity of the edge wireframe (only used when show_edges is on)."),
                io.Boolean.Input("show_connected_components", default=False,
                    tooltip="Color the mesh by connected component (pyvista connectivity / RegionId)."),
                io.Boolean.Input("revert_yz_flip", default=False,
                    tooltip="Swap the Y and Z axes (undo a y/z flip)."),
                io.Boolean.Input("three_plane", default=False,
                    tooltip="Render a 2x2 grid: iso view, then XY, YZ, XZ."),
                io.Boolean.Input("show_title", default=True,
                    tooltip="Draw the mesh filename as the title."),
                io.Boolean.Input("use_gpu", default=True,
                    tooltip="Try GPU/native (EGL) offscreen for speed; software fallback."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.String.Output(display_name="titles"),
            ],
        )

    @classmethod
    def execute(cls, meshes, resolution=512, background="white", mesh_opacity=1.0,
                show_edges=False, edge_opacity=1.0, show_connected_components=False,
                revert_yz_flip=False, three_plane=False, show_title=True, use_gpu=True):
        import random
        import numpy as np
        import torch
        import pyvista as pv

        def _s(x, d):
            return (x[0] if x else d) if isinstance(x, list) else x
        res = int(_s(resolution, 512))
        bg = _s(background, "white")
        m_op = float(_s(mesh_opacity, 1.0))
        edges = bool(_s(show_edges, False))
        e_op = float(_s(edge_opacity, 1.0))
        show_cc = bool(_s(show_connected_components, False))
        revert_yz = bool(_s(revert_yz_flip, False))
        three = bool(_s(three_plane, False))
        titled = bool(_s(show_title, True))
        gpu = bool(_s(use_gpu, True))

        if len(meshes) == 1 and isinstance(meshes[0], list):
            meshes = meshes[0]
        meshes = [m for m in meshes if m is not None]
        if not meshes:
            raise ValueError("Preview Mesh Batch Render: no meshes provided.")

        backend = _setup_offscreen(gpu)
        log.info("[PreviewMeshBatch] backend: %s | %d mesh(es) @ %dpx%s%s",
                 backend, len(meshes), res, " 3-plane" if three else "",
                 " CC" if show_cc else "")

        bg_rgb = {"white": "white", "black": "black", "gray": (0.5, 0.5, 0.5)}.get(bg, "white")
        text_color = "black" if bg == "white" else "white"
        fs = max(8, int(res / 45))

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

        def _render_one(poly, title):
            if three:
                p = pv.Plotter(shape=(2, 2), off_screen=True, window_size=[res, res])
                cells = [
                    (0, 0, "iso"), (0, 1, "XY"), (1, 0, "YZ"), (1, 1, "XZ"),
                ]
                for (r, c, label) in cells:
                    p.subplot(r, c)
                    p.background_color = bg_rgb
                    _add_geometry(p, poly, m_op, edges, e_op, show_cc)
                    if label == "iso":
                        p.view_isometric()
                        try:  # "random" initial view
                            p.camera.azimuth = random.uniform(0, 360)
                            p.camera.elevation = random.uniform(-25, 25)
                        except Exception:
                            pass
                        cap = title if titled else ""
                    else:
                        getattr(p, f"view_{label.lower()}")()
                        cap = f"{label}" + (f"  {title}" if titled else "")
                    if cap:
                        p.add_text(cap, font_size=max(7, fs - 3), color=text_color)
                shot = p.screenshot(return_img=True)
                p.close()
                return shot
            p = pv.Plotter(off_screen=True, window_size=[res, res])
            p.background_color = bg_rgb
            _add_geometry(p, poly, m_op, edges, e_op, show_cc)
            if titled:
                p.add_title(title, font_size=fs, color=text_color)
            p.view_isometric()
            shot = p.screenshot(return_img=True)
            p.close()
            return shot

        imgs, titles = [], []
        for i, mesh in iterator:
            title = _mesh_title(mesh, i)
            titles.append(title)
            try:
                poly = pv.wrap(mesh)
                if revert_yz:
                    pts = np.asarray(poly.points).copy()
                    pts[:, [1, 2]] = pts[:, [2, 1]]
                    poly.points = pts
                shot = _render_one(poly, title)
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
        return io.NodeOutput(torch.from_numpy(batch), " | ".join(titles))


NODE_CLASS_MAPPINGS = {"GeomPackPreviewMeshBatchRender": RenderMeshBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackPreviewMeshBatchRender": "Preview Mesh Batch Render"}
