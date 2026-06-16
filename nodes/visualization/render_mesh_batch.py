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
                      When OFF, rendering is parallelised across CPU cores with a
                      process pool (software rasterisation is embarrassingly parallel).
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


def _make_plotter(o):
    """Create an offscreen plotter (2x2 for three-plane, else single)."""
    import pyvista as pv
    if o["three"]:
        return pv.Plotter(shape=(2, 2), off_screen=True, window_size=[o["res"], o["res"]])
    return pv.Plotter(off_screen=True, window_size=[o["res"], o["res"]])


def _prep_poly(mesh, o):
    """Wrap a trimesh as PolyData, applying the optional Y/Z revert."""
    import numpy as np
    import pyvista as pv
    poly = pv.wrap(mesh)
    if o["revert_yz"]:
        # Deep-copy so we never mutate the upstream mesh. Swapping Y/Z is a
        # reflection: it flips face winding/normals (which can make a lit surface
        # render dark/inside-out), so rebuild normals afterwards.
        poly = poly.copy(deep=True)
        pts = np.asarray(poly.points, dtype=np.float64).copy()
        pts[:, [1, 2]] = pts[:, [2, 1]]
        poly.points = pts
        try:
            poly.flip_normals()
        except Exception:
            pass
    return poly


def _draw_into(p, poly, title, o):
    """Populate an already-created, freshly p.clear()'d plotter with the mesh and
    requested view(s). Titles/captions use named add_text so they replace cleanly
    when the plotter is reused across meshes."""
    import random
    bg_rgb, tcol, fs, titled = o["bg_rgb"], o["text_color"], o["fs"], o["titled"]

    def _surf():
        _add_geometry(p, poly, o["m_op"], o["edges"], o["e_op"], o["show_cc"])

    if o["three"]:
        for (r, c, label) in [(0, 0, "iso"), (0, 1, "XY"), (1, 0, "YZ"), (1, 1, "XZ")]:
            p.subplot(r, c)
            p.background_color = bg_rgb
            _surf()
            if label == "iso":
                p.view_isometric()
                try:
                    p.camera.azimuth = random.uniform(0, 360)
                    p.camera.elevation = random.uniform(-25, 25)
                except Exception:
                    pass
                p.reset_camera()
                cap = title if titled else ""
            else:
                getattr(p, f"view_{label.lower()}")()
                p.reset_camera()
                cap = label + (f"  {title}" if titled else "")
            if cap:
                p.add_text(cap, font_size=max(7, fs - 3), color=tcol, name=f"cap_{r}{c}")
    else:
        p.background_color = bg_rgb
        _surf()
        if titled:
            p.add_text(title, position="upper_edge", font_size=fs, color=tcol, name="gp_title")
        p.view_isometric()
        p.reset_camera()


def _screenshot_arr(p):
    import numpy as np
    shot = p.screenshot(return_img=True)
    arr = np.asarray(shot, dtype=np.float32) / 255.0
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _render_mesh_array(mesh, title, o):
    """One-shot render with its own plotter -- used by the process-pool workers."""
    p = _make_plotter(o)
    try:
        _draw_into(p, _prep_poly(mesh, o), title, o)
        return _screenshot_arr(p)
    finally:
        try:
            p.close()
        except Exception:
            pass


def _render_worker(payload):
    """ProcessPool worker (SOFTWARE path only). payload = (spec, title, opts) where
    spec is ("path", filepath) or ("mesh", trimesh). The parent started Xvfb with no
    GL context (fork-safe); each child makes its own software context.
    Returns the array, or ("ERR", message, res) on failure."""
    try:
        spec, title, o = payload
        import pyvista as pv
        pv.OFF_SCREEN = True
        if spec[0] == "path":
            from ..io import mesh_io
            mesh, err = mesh_io.load_mesh_file(spec[1])
            if mesh is None:
                return ("ERR", f"load failed: {err}", o["res"])
        else:
            mesh = spec[1]
        return _render_mesh_array(mesh, title, o)
    except Exception as e:
        try:
            res = payload[2]["res"]
        except Exception:
            res = 512
        return ("ERR", str(e), res)


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
                    tooltip="Try GPU/native (EGL) offscreen for speed. When OFF, render is "
                            "parallelised across CPU cores with a process pool."),
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
        import os
        import numpy as np
        import torch
        import pyvista as pv

        def _s(x, d):
            return (x[0] if x else d) if isinstance(x, list) else x
        res = int(_s(resolution, 512))
        bg = _s(background, "white")
        gpu = bool(_s(use_gpu, True))
        three = bool(_s(three_plane, False))
        show_cc = bool(_s(show_connected_components, False))

        if len(meshes) == 1 and isinstance(meshes[0], list):
            meshes = meshes[0]
        meshes = [m for m in meshes if m is not None]
        if not meshes:
            raise ValueError("Preview Mesh Batch Render: no meshes provided.")

        bg_rgb = {"white": "white", "black": "black", "gray": (0.5, 0.5, 0.5)}.get(bg, "white")
        opts = {
            "res": res, "bg_rgb": bg_rgb,
            "text_color": "black" if bg == "white" else "white",
            "fs": max(8, int(res / 45)),
            "m_op": float(_s(mesh_opacity, 1.0)),
            "edges": bool(_s(show_edges, False)),
            "e_op": float(_s(edge_opacity, 1.0)),
            "show_cc": show_cc,
            "revert_yz": bool(_s(revert_yz_flip, False)),
            "three": three,
            "titled": bool(_s(show_title, True)),
        }

        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(len(meshes))
        except Exception:
            pbar = None
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None

        titles = [_mesh_title(m, i) for i, m in enumerate(meshes)]
        imgs = [None] * len(meshes)

        def _run_pool():
            """SOFTWARE-only: render across CPU cores via a process pool (max workers).
            Each worker renders in its own software GL context and returns the small
            image (not the mesh). Returns True on success, False -> serial fallback.
            (GPU does NOT use this -- multiple EGL contexts gave 'bad X server' fallbacks
            and were slower than serial; GPU goes through the per-mesh serial path.)"""
            # Children share a parent-started Xvfb (started WITH NO GL context so the
            # fork stays safe).
            try:
                if os.name != "nt" and not os.environ.get("DISPLAY"):
                    pv.start_xvfb()
            except Exception as e:
                log.debug("[PreviewMeshBatch] start_xvfb: %s", e)

            payloads = []
            for i, mesh in enumerate(meshes):
                fp = None
                try:
                    fp = (mesh.metadata or {}).get("file_path")
                except Exception:
                    pass
                spec = ("path", fp) if (fp and os.path.isfile(fp)) else ("mesh", mesh)
                payloads.append((spec, titles[i], opts))

            from concurrent.futures import ProcessPoolExecutor
            workers = min(len(meshes), (os.cpu_count() or 4))  # max available
            log.info("[PreviewMeshBatch] software render: %d mesh(es) @ %dpx across %d processes",
                     len(meshes), res, workers)
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    gen = ex.map(_render_worker, payloads)
                    if tqdm is not None:
                        gen = tqdm(gen, total=len(meshes), desc="Preview Mesh Batch", unit="mesh")
                    for i, out in enumerate(gen):
                        if isinstance(out, tuple) and out and out[0] == "ERR":
                            log.error("[PreviewMeshBatch] render failed for %s: %s", titles[i], out[1])
                            imgs[i] = np.ones((res, res, 3), np.float32)
                        else:
                            imgs[i] = out
                        if pbar is not None:
                            pbar.update(1)
                return True
            except Exception as e:
                log.warning("[PreviewMeshBatch] parallel render failed (%s); serial fallback", e)
                return False

        # Process pool ONLY for the software path; GPU uses the serial per-mesh path.
        done = _run_pool() if ((not gpu) and len(meshes) > 1) else False

        if not done:
            # ---- serial path (single mesh, or pool fallback) ----
            # Fresh plotter per mesh (default lights -> shading). Reuse+clear() dropped
            # the lights AND barely helped (cost is geometry upload + GPU readback).
            backend = _setup_offscreen(gpu)
            log.info("[PreviewMeshBatch] serial backend: %s | %d mesh(es) @ %dpx%s%s",
                     backend, len(meshes), res, " 3-plane" if three else "", " CC" if show_cc else "")
            it = enumerate(meshes)
            if tqdm is not None:
                it = tqdm(list(it), desc="Preview Mesh Batch", unit="mesh")
            for i, mesh in it:
                try:
                    imgs[i] = _render_mesh_array(mesh, titles[i], opts)
                except Exception as e:
                    log.error("[PreviewMeshBatch] render failed for %s: %s", titles[i], e)
                    imgs[i] = np.ones((res, res, 3), np.float32)
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
