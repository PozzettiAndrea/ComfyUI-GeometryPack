# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Sharpen backend that literally DECREASES Gaussian curvature (toward developable).

Discrete Gaussian curvature at a vertex is the angle defect K_i = 2*pi - sum of the
incident triangle angles. This backend does gradient-descent on the Gaussian-curvature
energy

    E = sum_i K_i^2  +  lambda * ||x - x0||^2

so the pointwise |K| is driven toward zero -> the surface becomes locally DEVELOPABLE
(planes / cylinders / cones, K=0), which is what most CAD surfaces are. (Gauss-Bonnet
fixes the TOTAL sum_i K_i = 2*pi*chi, so the flow spreads that topological remainder
thinly and flattens the bulk; the anchor lambda keeps it near the input so the shape
isn't destroyed.) The exact analytic angle-defect gradient is used (verified vs finite
differences), and every step is passed through the same signed-area barrier as the
foldless guided_normal update, so no triangle inverts.

Related: developable surface flows (Stein, Grinspun, Crane 2018). Here we use the simple
L2 angle-defect energy (decrease |K|); an L1 variant would instead CONCENTRATE K onto
sparse seams."""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")

_TWO_PI = 2.0 * np.pi


def _boundary_vertices(F, n):
    """Indices of vertices on a boundary edge (edge used by exactly one face)."""
    E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(E, axis=0, return_counts=True)
    be = uniq[cnt == 1]
    bv = np.zeros(n, dtype=bool)
    if len(be):
        bv[be.ravel()] = True
    return bv


def _decrease_gaussian_gpu(mesh, iterations, strength, anchor_weight, use_gpu,
                           regularizer="reduce", irls_eps=0.01):
    import torch
    from .guided_normal_gpu import _anti_flip_step_torch, _torch_device

    V0 = np.asarray(mesh.vertices, dtype=np.float64)
    Ff = np.asarray(mesh.faces, dtype=np.int64)
    if len(V0) == 0 or len(Ff) == 0:
        return None, "Empty mesh (no vertices or faces).", "cpu"

    n = len(V0)
    dev = _torch_device() if use_gpu else __import__("torch").device("cpu")
    eps = 1e-12
    bv = _boundary_vertices(Ff, n)

    V = torch.as_tensor(V0, dtype=torch.float32, device=dev)
    X0 = V.clone()
    F = torch.as_tensor(Ff, dtype=torch.long, device=dev)
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    bmask = torch.as_tensor(bv, device=dev)
    mean_edge = float(np.mean(np.linalg.norm(V0[Ff[:, 1]] - V0[Ff[:, 0]], axis=1)))
    lam = float(anchor_weight)

    def ang_grad(p0, p1, p2):
        """Angle at p0 between edges p0->p1, p0->p2, and its gradient w.r.t. p0,p1,p2."""
        e1 = p1 - p0
        e2 = p2 - p0
        N = torch.cross(e1, e2, dim=1)
        N = N / (N.norm(dim=1, keepdim=True) + eps)
        u = e1 / (e1.norm(dim=1, keepdim=True) + eps)
        w = e2 / (e2.norm(dim=1, keepdim=True) + eps)
        ang = torch.arccos(torch.clamp((u * w).sum(1), -1.0 + 1e-7, 1.0 - 1e-7))
        g1 = torch.cross(N, p0 - p1, dim=1) / ((p0 - p1).pow(2).sum(1, keepdim=True) + eps)
        g2 = torch.cross(N, p2 - p0, dim=1) / ((p2 - p0).pow(2).sum(1, keepdim=True) + eps)
        g0 = -(g1 + g2)
        return ang, g0, g1, g2

    for _ in range(int(iterations)):
        a, b, c = V[i0], V[i1], V[i2]
        al, gaa, gab, gac = ang_grad(a, b, c)      # angle at vertex i0
        be, gbb, gbc, gba = ang_grad(b, c, a)      # angle at vertex i1 (p0=b,p1=c,p2=a)
        ga, gcc, gca, gcb = ang_grad(c, a, b)      # angle at vertex i2

        K = torch.full((n,), _TWO_PI, device=dev)
        K.index_add_(0, i0, -al)
        K.index_add_(0, i1, -be)
        K.index_add_(0, i2, -ga)
        K[bmask] = 0.0                              # boundary defect isn't curvature
        # 'reduce' = L2 (minimize sum K^2: lowers |K| everywhere, spreads it thin).
        # 'developable' = L1 via IRLS reweight w=1/(|K|+eps): pushes small K to 0
        # (flat/cylinder/cone patches) and CONCENTRATES K onto sparse seams -> the
        # surface becomes piecewise zero-Gaussian-curvature (developable).
        if regularizer == "developable":
            w = 1.0 / (K.abs() + float(irls_eps))
        else:
            w = torch.ones_like(K)
        WK = w * K
        Ka, Kb, Kc = WK[i0], WK[i1], WK[i2]

        # dE/dx = sum_v 2 K_v * grad K_v,  grad K_v = -sum_{angle at v} grad(angle)
        grad = torch.zeros((n, 3), device=dev)
        ca = (2.0 * Ka).unsqueeze(1)
        grad.index_add_(0, i0, ca * (-gaa)); grad.index_add_(0, i1, ca * (-gab)); grad.index_add_(0, i2, ca * (-gac))
        cb = (2.0 * Kb).unsqueeze(1)
        grad.index_add_(0, i1, cb * (-gbb)); grad.index_add_(0, i2, cb * (-gbc)); grad.index_add_(0, i0, cb * (-gba))
        cc = (2.0 * Kc).unsqueeze(1)
        grad.index_add_(0, i2, cc * (-gcc)); grad.index_add_(0, i0, cc * (-gca)); grad.index_add_(0, i1, cc * (-gcb))

        grad = grad + 2.0 * lam * (V - X0)         # Tikhonov anchor (keeps shape / gauge)
        grad[bmask] = 0.0

        # descent step: cap the LARGEST vertex move to strength*edge (conservative
        # but guarantees monotone descent on this stiff non-convex energy; median-
        # based stepping overshoots and drives |K| back up). Barrier => no fold.
        gmax = grad.norm(dim=1).max()
        eta = strength * mean_edge / (gmax + eps)
        move = -eta * grad
        V = V + _anti_flip_step_torch(V, F, move)  # foldless barrier

    out = trimesh_module.Trimesh(vertices=V.detach().cpu().numpy().astype(np.float64),
                                 faces=Ff.astype(np.int32), process=False)
    return out, "", str(dev)


class SharpenDecreaseGaussianNode(io.ComfyNode):
    """Decrease Gaussian curvature (developable flow) sharpening backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpen_DecreaseGaussian",
            display_name="Sharpen Decrease Gaussian (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("iterations", default=20, min=1, max=2000, step=1, tooltip=(
                    "Gradient-descent steps on the Gaussian-curvature energy sum K_i^2 "
                    "(K_i = vertex angle defect). More steps = lower |Gaussian curvature| = "
                    "more developable (flatter / more cylinder-and-cone-like).")),
                io.Float.Input("strength", default=0.05, min=0.001, max=1.0, step=0.001, display_mode="number", tooltip=(
                    "Per-step size, as a fraction of the average edge length (the largest "
                    "vertex move each step is ~strength * edge). Higher = faster but coarser; "
                    "the foldless barrier still prevents triangle inversion. Default 0.05.")),
                io.Float.Input("anchor_weight", default=0.5, min=0.0, max=50.0, step=0.01, display_mode="number", tooltip=(
                    "Tikhonov lambda -- how strongly vertices are held to their ORIGINAL "
                    "positions while Gaussian curvature is reduced. LOWER = flatten harder "
                    "(bigger shape change); HIGHER = stay close to input. Default 0.5.")),
                io.Combo.Input("regularizer", options=["developable", "reduce"], default="developable", tooltip=(
                    "developable = L1/sparsity on K (push small K to 0, CONCENTRATE it onto "
                    "sparse seams) -> piecewise ZERO-Gaussian-curvature: planes + cylinders + "
                    "cones kept smooth, only the seams curve. This is the CAD-friendly mode "
                    "(keeps fillets/cylinders, doesn't facet them). reduce = L2 (lower |K| "
                    "everywhere, spreads it thin -- gentler, less structured).")),
                io.Float.Input("irls_eps", default=0.005, min=0.0005, max=0.5, step=0.0005, display_mode="number", tooltip=(
                    "(developable mode) Sparsity epsilon of the IRLS L1 reweight w=1/(|K|+eps). "
                    "SMALLER = more aggressively sparse / L0-like (crisper developable patches, "
                    "sharper seams); LARGER = softer. Default 0.005.")),
                io.Combo.Input("use_gpu", options=["true", "false"], default="true", tooltip=(
                    "Run on CUDA (recommended). false = CPU torch.")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, iterations=20, strength=0.05, anchor_weight=0.5,
                regularizer="developable", irls_eps=0.005, use_gpu="true"):
        import time
        iv, ifc = len(trimesh.vertices), len(trimesh.faces)
        log.info("Backend: decrease_gaussian | %d verts %d faces | iters=%d strength=%.3f anchor=%.3f reg=%s eps=%.4f gpu=%s",
                 iv, ifc, iterations, strength, anchor_weight, regularizer, irls_eps, use_gpu)
        t0 = time.perf_counter()
        sharpened, error, dev = _decrease_gaussian_gpu(
            trimesh, iterations, strength, anchor_weight, use_gpu == "true",
            regularizer=regularizer, irls_eps=irls_eps)
        elapsed = time.perf_counter() - t0
        if sharpened is None:
            raise ValueError(f"Sharpening failed (decrease_gaussian): {error}")

        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": "decrease_gaussian", "device": str(dev),
            "iterations": iterations, "strength": strength, "anchor_weight": anchor_weight,
            "regularizer": regularizer, "irls_eps": irls_eps,
        }
        disp = np.linalg.norm(np.asarray(sharpened.vertices) - np.asarray(trimesh.vertices), axis=1)
        info = (f"Sharpen Mesh Results (decrease_gaussian, device={dev}):\n\n"
                f"Iterations: {iterations}\nStrength: {strength}\nAnchor weight: {anchor_weight}\n"
                f"Time: {elapsed:.2f}s\n\nVertices: {iv:,} (unchanged)\nFaces: {ifc:,} (unchanged)\n\n"
                f"Displacement:\n  Average: {float(np.mean(disp)):.6f}\n  Maximum: {float(np.max(disp)):.6f}\n")
        if len(sharpened.vertices) == len(trimesh.vertices):
            sharpened.vertex_attributes["sharpen_displacement_magnitude"] = disp.astype(np.float32)
        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_DecreaseGaussian": SharpenDecreaseGaussianNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_DecreaseGaussian": "Sharpen Decrease Gaussian (backend)"}
