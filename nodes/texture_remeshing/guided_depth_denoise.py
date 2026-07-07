# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Guided Depth Denoise -- edge-preserving denoise of a depth/height map.

Removes high-frequency grain (e.g. the VAE-decode grain from latent-diffusion depth
models like Lotus/Marigold) WITHOUT blurring real depth edges, by a joint-bilateral
filter guided by the input RGB: a pixel is averaged with a neighbour only when they
are close in space AND the GUIDE (RGB) image is similar there -- so flat regions get
smoothed but silhouettes/creases the RGB shows are preserved.

Mask-aware: only pixels inside the mask are denoised, and pixels OUTSIDE the mask
contribute zero weight (normalized convolution) -- so the background never bleeds into
the masked surface, and out-of-mask pixels are left untouched.

Pure numpy (vectorised shift-accumulate), no opencv-contrib needed. If no guide image
is given it self-guides (plain bilateral on the depth).
"""

import logging

import numpy as np
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.array(x)


def _to_gray(arr):
    """Any IMAGE/MASK tensor -> (H,W) float in [0,1]."""
    a = _to_numpy(arr).astype(np.float64)
    if a.ndim == 4:                              # (B,H,W,C)
        a = a[0]
    if a.ndim == 3:
        if a.shape[2] in (3, 4):                 # (H,W,C) -> luminance
            a = a[:, :, :3].mean(axis=2)
        elif a.shape[2] == 1:
            a = a[:, :, 0]
        else:                                    # (B,H,W) mask
            a = a[0]
    if a.ndim != 2:
        a = a.squeeze()
    if a.max() > 1.0:
        a = a / 255.0
    return a


def _joint_bilateral_masked(depth, guide, mask, radius, sig_s, sig_c, iters):
    """Masked, guide-weighted bilateral. depth/guide/mask are (H,W) float.
    Only mask>0.5 pixels are updated; out-of-mask pixels carry zero weight."""
    H, W = depth.shape
    P = int(radius)
    inside = mask > 0.5
    # precompute spatial weights
    sig_s2 = 2.0 * sig_s * sig_s
    sig_c2 = 2.0 * max(sig_c, 1e-6) ** 2
    offsets = []
    for dy in range(-P, P + 1):
        for dx in range(-P, P + 1):
            ws = np.exp(-(dx * dx + dy * dy) / sig_s2)
            if ws >= 1e-4:
                offsets.append((dy, dx, ws))

    cur = depth.copy()
    gpad = np.pad(guide, P, mode="reflect")
    mpad = np.pad(mask, P, mode="reflect")
    for _ in range(int(iters)):
        dpad = np.pad(cur, P, mode="reflect")
        acc = np.zeros((H, W), dtype=np.float64)
        wsum = np.zeros((H, W), dtype=np.float64)
        for dy, dx, ws in offsets:
            g_sh = gpad[P + dy:P + dy + H, P + dx:P + dx + W]
            d_sh = dpad[P + dy:P + dy + H, P + dx:P + dx + W]
            m_sh = mpad[P + dy:P + dy + H, P + dx:P + dx + W]
            wr = np.exp(-((guide - g_sh) ** 2) / sig_c2)
            w = ws * wr * m_sh                   # m_sh: out-of-mask neighbours -> 0
            acc += w * d_sh
            wsum += w
        out = np.where(wsum > 1e-8, acc / np.maximum(wsum, 1e-8), cur)
        cur = np.where(inside, out, cur)          # only update inside the mask
    return cur.astype(np.float32)


def _joint_bilateral_masked_torch(depth, guide, mask, radius, sig_s, sig_c, iters, dev):
    """GPU/torch version of the masked joint-bilateral (identical math to the numpy one)."""
    import torch
    import torch.nn.functional as F
    H, W = depth.shape
    P = int(radius)
    d = torch.as_tensor(depth, dtype=torch.float32, device=dev)
    g = torch.as_tensor(guide, dtype=torch.float32, device=dev)
    m = torch.as_tensor(mask, dtype=torch.float32, device=dev)
    inside = m > 0.5
    sig_s2 = 2.0 * sig_s * sig_s
    sig_c2 = 2.0 * max(sig_c, 1e-6) ** 2
    offsets = []
    for dy in range(-P, P + 1):
        for dx in range(-P, P + 1):
            ws = float(np.exp(-(dy * dy + dx * dx) / sig_s2))
            if ws >= 1e-4:
                offsets.append((dy, dx, ws))

    def _pad(t):
        return F.pad(t[None, None], (P, P, P, P), mode="reflect")[0, 0]

    gpad = _pad(g)
    mpad = _pad(m)
    cur = d.clone()
    for _ in range(int(iters)):
        dpad = _pad(cur)
        acc = torch.zeros((H, W), dtype=torch.float32, device=dev)
        wsum = torch.zeros((H, W), dtype=torch.float32, device=dev)
        for dy, dx, ws in offsets:
            g_sh = gpad[P + dy:P + dy + H, P + dx:P + dx + W]
            d_sh = dpad[P + dy:P + dy + H, P + dx:P + dx + W]
            m_sh = mpad[P + dy:P + dy + H, P + dx:P + dx + W]
            wr = torch.exp(-((g - g_sh) ** 2) / sig_c2)
            w = ws * wr * m_sh
            acc += w * d_sh
            wsum += w
        out = torch.where(wsum > 1e-8, acc / wsum.clamp_min(1e-8), cur)
        cur = torch.where(inside, out, cur)
    return cur.detach().cpu().numpy().astype(np.float32)


class GuidedDepthDenoiseNode(io.ComfyNode):
    """Edge-preserving (joint-bilateral) denoise of a depth map, guided by RGB, masked."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackGuidedDepthDenoise",
            display_name="Guided Depth Denoise",
            category="geompack/texture_remeshing",
            is_output_node=True,
            description=(
                "Edge-preserving denoise of a depth/height map via a joint-bilateral filter "
                "guided by an RGB image. Kills high-frequency grain (e.g. VAE-decode grain from "
                "Lotus/Marigold latent-diffusion depth) while KEEPING real depth edges -- it "
                "only averages pixels that are close in space AND similar in the guide image.\n\n"
                "MASK-aware: only pixels inside the mask are denoised; pixels outside contribute "
                "zero weight (no background bleed) and are left untouched. Run this BEFORE "
                "Depth Map to Mesh so grain doesn't become bumpy geometry. Pure-numpy."
            ),
            inputs=[
                io.Image.Input("depth",
                    tooltip="The depth / height map to denoise (RGB is averaged to grayscale, "
                            "or single channel). Output is the cleaned depth as a grayscale IMAGE."),
                io.Image.Input("guide", optional=True,
                    tooltip="RGB guide image for edge preservation (typically the ORIGINAL "
                            "photo/render). Smoothing stops where the guide shows an edge. If "
                            "omitted, the filter self-guides on the depth (plain bilateral)."),
                io.Mask.Input("mask", optional=True,
                    tooltip="Only denoise pixels where mask > 0.5; outside pixels are untouched "
                            "and excluded from averaging (no background bleed). Omit = whole image."),
                io.Int.Input("radius", default=4, min=1, max=32, step=1, tooltip=(
                    "Filter window radius in pixels. Larger = removes coarser grain / smooths "
                    "more, but slower (cost ~ (2*radius+1)^2). Keep it a bit above sigma_spatial. ~3-6.")),
                io.Float.Input("sigma_spatial", default=3.0, min=0.5, max=32.0, step=0.5,
                    display_mode="number", tooltip=(
                    "Spatial falloff (pixels): how far neighbours influence a pixel. Larger = "
                    "smoother. Should be <= radius. ~2-5.")),
                io.Float.Input("sigma_color", default=0.1, min=0.005, max=1.0, step=0.005,
                    display_mode="number", tooltip=(
                    "Guide range sigma (on the normalized [0,1] guide): how similar guide pixels "
                    "must be to be averaged together. SMALLER = stricter edge preservation (only "
                    "very-similar pixels mix -> keeps fine edges, removes less grain); LARGER = "
                    "more smoothing across mild guide variation. ~0.05-0.2.")),
                io.Int.Input("iterations", default=1, min=1, max=10, step=1, tooltip=(
                    "Apply the filter N times. More passes = stronger denoise (approaches a "
                    "stronger edge-preserving smoothing) without enlarging the window. ~1-3.")),
                io.Combo.Input("device", options=["auto", "cpu", "gpu"], default="auto", tooltip=(
                    "Compute device. 'gpu' (torch/CUDA) is much faster for large images or big "
                    "radius/iterations; 'cpu' is pure-numpy. 'auto' uses the GPU if available. "
                    "Results are identical (same math).")),
            ],
            outputs=[
                io.Image.Output(display_name="denoised_depth"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, depth, guide=None, mask=None, radius=4, sigma_spatial=3.0,
                sigma_color=0.1, iterations=1, device="auto"):
        import torch

        d = _to_gray(depth)
        H, W = d.shape
        g = _to_gray(guide) if guide is not None else d
        if g.shape != (H, W):                    # nearest-resize guide to depth
            ys = np.linspace(0, g.shape[0] - 1, H).round().astype(int)
            xs = np.linspace(0, g.shape[1] - 1, W).round().astype(int)
            g = g[np.ix_(ys, xs)]
        if mask is not None:
            m = _to_gray(mask)
            if m.shape != (H, W):
                ys = np.linspace(0, m.shape[0] - 1, H).round().astype(int)
                xs = np.linspace(0, m.shape[1] - 1, W).round().astype(int)
                m = m[np.ix_(ys, xs)]
        else:
            m = np.ones((H, W), dtype=np.float64)

        use_gpu = device == "gpu" or (device == "auto" and torch.cuda.is_available())
        if use_gpu:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            out = _joint_bilateral_masked_torch(d, g, m, radius, float(sigma_spatial),
                                                float(sigma_color), int(iterations), dev)
            dev_used = "gpu (%s)" % dev
        else:
            out = _joint_bilateral_masked(d, g, m, radius, float(sigma_spatial),
                                          float(sigma_color), int(iterations))
            dev_used = "cpu (numpy)"

        # report how much grain was removed inside the mask
        inside = m > 0.5
        if inside.any():
            before = float(d[inside].std())
            resid = float((d[inside] - out[inside]).std())
        else:
            before = resid = 0.0

        img = torch.from_numpy(np.repeat(out[:, :, None], 3, axis=2)[None].astype(np.float32))
        info = (
            f"Guided Depth Denoise\n\n"
            f"{W}x{H} | mask: {'yes (%.1f%%)' % (100*inside.mean()) if mask is not None else 'whole image'} | "
            f"guide: {'RGB' if guide is not None else 'self'}\n"
            f"radius={radius} sigma_spatial={sigma_spatial} sigma_color={sigma_color} iters={iterations} | device={dev_used}\n"
            f"removed (std of change, in-mask): {resid:.4g}  (input std {before:.4g})\n"
            f"\nOutput: denoised_depth (grayscale IMAGE) -> feed to Depth Map to Mesh"
        )
        log.info("GuidedDepthDenoise: %dx%d, removed std %.4g", W, H, resid)
        return io.NodeOutput(img, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackGuidedDepthDenoise": GuidedDepthDenoiseNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackGuidedDepthDenoise": "Guided Depth Denoise"}
