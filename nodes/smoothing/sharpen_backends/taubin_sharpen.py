# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Pure-Taubin sharpening backend (geometric unsharp masking on a Taubin low-pass).

Taubin's lambda|mu algorithm (Taubin, SIGGRAPH 1995) is a linear filter on the mesh
Laplacian whose transfer function is the polynomial

    f(k) = (1 - lambda*k) * (1 - mu*k)         (k = Laplacian frequency / eigenvalue)

With lambda>0 and mu<-lambda it is a shrink-free LOW-PASS: f~1 on low frequencies,
f~0 on high ones -> SMOOTHING. To SHARPEN with the very same operator we amplify the
band Taubin removes (geometric unsharp masking):

    S      = Taubin_lowpass(V0)                 (N iterations of the lambda then mu step)
    V_out  = V0 + enhancement * (V0 - S)

This is "pure Taubin" -- it is built entirely from the Taubin smoothing operator and is
unconditionally stable. (A raw sign-flipped / negative-lambda Taubin, i.e. inverse
diffusion, would also boost high frequencies but is numerically unstable and blows up
triangles, so we do not use it.) The high-frequency detail (V0 - S) is added back scaled
by `enhancement`: 0 = no change, 1 = double the detail, >1 = aggressive.

The Laplacian is the classic uniform/umbrella operator L x_i = mean_j(x_j) - x_i over the
1-ring (Taubin's original). Boundary vertices are held fixed. The sharpening displacement
is passed through the same signed-area barrier as the foldless guided_normal update so no
triangle inverts, even at large `enhancement`. Runs on CUDA (use_gpu) or CPU torch."""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _boundary_vertices(F, n):
    """Indices of vertices on a boundary edge (edge used by exactly one face)."""
    E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(E, axis=0, return_counts=True)
    be = uniq[cnt == 1]
    bv = np.zeros(n, dtype=bool)
    if len(be):
        bv[be.ravel()] = True
    return bv


def _taubin_sharpen(mesh, lambda_, mu, iterations, enhancement, use_gpu, anti_flip=True):
    """Taubin low-pass + unsharp amplification, foldless. Returns (trimesh, device_str)."""
    import torch
    from .guided_normal_gpu import _anti_flip_step_torch, _torch_device

    V0n = np.asarray(mesh.vertices, dtype=np.float64)
    Ff = np.asarray(mesh.faces, dtype=np.int64)
    if len(V0n) == 0 or len(Ff) == 0:
        return None, "Empty mesh (no vertices or faces)."

    n = len(V0n)
    dev = _torch_device() if use_gpu else torch.device("cpu")

    # --- uniform 1-ring Laplacian connectivity (undirected, symmetric) ---
    E = np.sort(np.vstack([Ff[:, [0, 1]], Ff[:, [1, 2]], Ff[:, [2, 0]]]), axis=1)
    E = np.unique(E, axis=0)
    src = np.concatenate([E[:, 0], E[:, 1]])
    dst = np.concatenate([E[:, 1], E[:, 0]])
    deg = np.bincount(src, minlength=n).astype(np.float64)
    deg[deg == 0] = 1.0

    bv = _boundary_vertices(Ff, n)

    src_t = torch.as_tensor(src, dtype=torch.long, device=dev)
    dst_t = torch.as_tensor(dst, dtype=torch.long, device=dev)
    inv_deg = torch.as_tensor(1.0 / deg, dtype=torch.float32, device=dev).unsqueeze(1)
    bmask = torch.as_tensor(bv, device=dev)

    V0 = torch.as_tensor(V0n, dtype=torch.float32, device=dev)
    F = torch.as_tensor(Ff, dtype=torch.long, device=dev)

    def umbrella(x):
        """L x = mean_j(x_j) - x_i over the 1-ring (uniform Laplacian)."""
        acc = torch.zeros_like(x)
        acc.index_add_(0, src_t, x[dst_t])
        Lx = acc * inv_deg - x
        Lx[bmask] = 0.0                      # hold boundary fixed
        return Lx

    # --- Taubin low-pass: N passes of (lambda then mu) ---
    x = V0.clone()
    lam = float(lambda_)
    m = float(mu)
    for _ in range(int(iterations)):
        x = x + lam * umbrella(x)
        x = x + m * umbrella(x)

    # --- unsharp amplification: add the removed detail back, scaled ---
    detail = V0 - x                          # high-frequency band Taubin removed
    disp = float(enhancement) * detail
    disp[bmask] = 0.0

    if anti_flip:
        disp = _anti_flip_step_torch(V0, F, disp)

    Vout = (V0 + disp).detach().cpu().numpy().astype(np.float64)
    out = trimesh_module.Trimesh(vertices=Vout, faces=Ff, process=False)
    dev_str = "cuda" if dev.type == "cuda" else "cpu"
    return out, dev_str


class SharpenTaubinNode(io.ComfyNode):
    """Pure-Taubin sharpening backend (unsharp masking on a Taubin low-pass)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpen_Taubin",
            display_name="Sharpen Taubin (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("lambda_", default=0.5, min=0.01, max=1.0, step=0.01, tooltip=(
                    "Taubin low-pass shrink step. Higher = stronger smoothing per pass "
                    "(larger feature scale removed -> coarser detail amplified).")),
                io.Float.Input("mu", default=-0.53, min=-1.0, max=-0.01, step=0.01, tooltip=(
                    "Taubin un-shrink step (negative). Must satisfy |mu| > lambda for the "
                    "low-pass to be shrink-free. Typical -0.53 for lambda=0.5.")),
                io.Int.Input("iterations", default=10, min=1, max=200, step=1, tooltip=(
                    "Taubin low-pass passes. MORE passes = smoother reference S = sharpens "
                    "BROADER/larger-scale features; FEWER passes = sharpens fine detail.")),
                io.Float.Input("enhancement", default=0.6, min=0.0, max=5.0, step=0.05, display_mode="number", tooltip=(
                    "Unsharp strength alpha: V_out = V0 + alpha*(V0 - S). 0 = no change, "
                    "1 = double the high-frequency detail, >1 = aggressive sharpening. The "
                    "foldless barrier prevents triangle inversion regardless.")),
                io.Combo.Input("anti_flip", options=["true", "false"], default="true", tooltip=(
                    "Pass the sharpening displacement through the signed-area barrier so no "
                    "triangle folds (recommended; only matters at high enhancement).")),
                io.Combo.Input("use_gpu", options=["true", "false"], default="true", tooltip=(
                    "Run on CUDA (recommended). false = CPU torch.")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, lambda_=0.5, mu=-0.53, iterations=10, enhancement=0.6,
                anti_flip="true", use_gpu="true"):
        log.info("Backend: taubin_sharpen")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: lambda=%.3f, mu=%.3f, iterations=%d, enhancement=%.3f, gpu=%s",
                 lambda_, mu, iterations, enhancement, use_gpu)

        if mu >= lambda_:
            log.warning("Taubin sharpen: |mu| (%.3f) <= lambda (%.3f); low-pass not "
                        "shrink-free, results may drift.", abs(mu), lambda_)

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        sharpened, dev_str = _taubin_sharpen(
            trimesh, float(lambda_), float(mu), int(iterations), float(enhancement),
            use_gpu == "true", anti_flip == "true",
        )
        if sharpened is None:
            raise ValueError(f"Sharpening failed (taubin_sharpen): {dev_str}")

        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": "taubin_sharpen",
            "lambda": lambda_, "mu": mu, "iterations": iterations,
            "enhancement": enhancement,
            "original_vertices": initial_vertices,
            "original_faces": initial_faces,
        }

        disp = np.linalg.norm(
            np.asarray(sharpened.vertices) - np.asarray(trimesh.vertices), axis=1)
        avg_disp = float(np.mean(disp))
        max_disp = float(np.max(disp))

        log.info("Output: %d vertices, %d faces (device=%s)",
                 len(sharpened.vertices), len(sharpened.faces), dev_str)
        log.info("Avg vertex displacement: %.6f, max: %.6f", avg_disp, max_disp)

        info = f"""Sharpen Mesh Results (taubin_sharpen):

Lambda: {lambda_}
Mu: {mu}
Low-pass iterations: {iterations}
Enhancement (alpha): {enhancement}
Anti-flip: {anti_flip} | Device: {dev_str}

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        if len(sharpened.vertices) == len(trimesh.vertices):
            sharpened.vertex_attributes["sharpen_displacement_magnitude"] = disp.astype(np.float32)

        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_Taubin": SharpenTaubinNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_Taubin": "Sharpen Taubin (backend)"}
