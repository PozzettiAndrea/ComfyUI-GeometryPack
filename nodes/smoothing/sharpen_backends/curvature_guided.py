# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Curvature-domain guided sharpening backend (GPU, prototype).

The second-order analogue of guided_normal. Where guided_normal filters face
NORMALS (first order) and reconstructs by plane projection, this filters the
mesh CURVATURE and reconstructs by a Laplacian-domain solve.

Pipeline (all on torch / CUDA):
  1. delta = L x          -- the differential (Laplacian) coordinates; |delta_i|
                             is the (area-weighted) mean-curvature normal, so
                             H_i = |delta_i| / (2 A_i) is pointwise mean curvature.
  2. Edge-aware bilateral filter of the SCALAR curvature |H| over the 1-ring,
     range-weighted by NORMAL difference so it diffuses curvature WITHIN a region
     but stops at sharp edges -> "increase curvature agreement within regions".
     We filter the magnitude only and keep each vertex's own delta DIRECTION, so
     a perfect sphere/cylinder is a fixed point (no flattening).
  3. Rebuild target delta_f = (2 H_f A_i) * (delta_i / |delta_i|).
  4. Reconstruct positions: minimise ||L x - delta_f||^2 + lambda||x - x0||^2,
     i.e. solve (L^T L + lambda I) x = L^T delta_f + lambda x0 with a diagonally
     preconditioned conjugate gradient (sparse mat-vecs -> GPU friendly).

References: Laplacian/differential-coordinate surface editing (Sorkine et al.
2004; Lipman et al. 2004); curvature-domain shape processing (Eigensatz et al.
2008). This is a linearised (quadratic-bending) prototype.
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _torch_device(use_gpu):
    import torch
    if use_gpu:
        try:
            import comfy.model_management as mm
            dev = mm.get_torch_device()
            if dev is not None and dev.type == "cuda":
                return dev
        except Exception:
            pass
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def _curvature_guided(mesh, iterations, sigma_s, sigma_r, anchor_weight, cg_iters, use_gpu,
                      direction_blend=0.0):
    import torch
    import igl

    V0 = np.asarray(mesh.vertices, dtype=np.float64)
    Ff = np.asarray(mesh.faces, dtype=np.int64)
    if len(V0) == 0 or len(Ff) == 0:
        return None, "Empty mesh (no vertices or faces).", "cpu"

    n = len(V0)
    dev = _torch_device(use_gpu)
    eps = 1e-12

    # --- cotangent Laplacian (symmetric) as a torch sparse tensor ---
    L = igl.cotmatrix(V0, Ff).tocoo()
    Lidx = torch.tensor(np.vstack([L.row, L.col]), dtype=torch.long, device=dev)
    Lval = torch.tensor(L.data, dtype=torch.float32, device=dev)
    Lsp = torch.sparse_coo_tensor(Lidx, Lval, (n, n)).coalesce()

    M = igl.massmatrix(V0, Ff, igl.MASSMATRIX_TYPE_VORONOI)
    Mdiag = torch.tensor(np.asarray(M.diagonal(), dtype=np.float64), dtype=torch.float32, device=dev)
    Mdiag = torch.clamp(Mdiag, min=1e-12)

    X0 = torch.tensor(V0, dtype=torch.float32, device=dev)

    def spmm(A, X):
        return torch.sparse.mm(A, X)

    delta = spmm(Lsp, X0)                       # (n,3) differential coords
    dmag = delta.norm(dim=1)                    # (n,)
    Hmag = dmag / (2.0 * Mdiag)                 # pointwise |mean curvature|
    direction = delta / (dmag.unsqueeze(1) + eps)

    # --- 1-ring directed edges (both directions) for the bilateral filter ---
    E = np.vstack([Ff[:, [0, 1]], Ff[:, [1, 2]], Ff[:, [2, 0]]])
    ed = np.unique(np.vstack([E, E[:, [1, 0]]]), axis=0)
    src = torch.tensor(ed[:, 0], dtype=torch.long, device=dev)   # accumulate into i
    dst = torch.tensor(ed[:, 1], dtype=torch.long, device=dev)   # neighbour j

    VN = torch.tensor(np.asarray(mesh.vertex_normals, dtype=np.float64), dtype=torch.float32, device=dev)
    avg_edge = float(np.mean(np.linalg.norm(V0[E[:, 0]] - V0[E[:, 1]], axis=1)))
    ss2 = 2.0 * (sigma_s * avg_edge) ** 2
    sr2 = 2.0 * (sigma_r ** 2)
    dpos = (X0[src] - X0[dst]).pow(2).sum(1)
    dnrm = (VN[src] - VN[dst]).pow(2).sum(1)
    w = torch.exp(-dpos / (ss2 + eps)) * torch.exp(-dnrm / (sr2 + eps))   # (E,)
    wsum = torch.ones(n, device=dev)            # self weight 1.0
    wsum.index_add_(0, src, w)

    # --- edge-aware diffusion of the SCALAR curvature magnitude ---
    Hf = Hmag.clone()
    for _ in range(int(iterations)):
        acc = Hf.clone()                        # self term
        acc.index_add_(0, src, w * Hf[dst])
        Hf = acc / wsum

    # Optionally also smooth the delta DIRECTION. magnitude-only (blend=0) keeps each
    # vertex's own orientation -> a sphere/cylinder is a fixed point (no flattening) but
    # positional noise (wrong directions) survives. blend>0 mixes in an edge-aware
    # smoothed direction, which denoises positions at some flattening risk in curved
    # regions (the sigma_r range weight still protects sharp edges either way).
    if direction_blend > 0.0:
        vdf = delta.clone()
        for _ in range(int(iterations)):
            acc = vdf.clone()
            acc.index_add_(0, src, w.unsqueeze(1) * vdf[dst])
            vdf = acc / wsum.unsqueeze(1)
        unit_v = vdf / (vdf.norm(dim=1, keepdim=True) + eps)
        direction = direction * (1.0 - direction_blend) + unit_v * direction_blend
        direction = direction / (direction.norm(dim=1, keepdim=True) + eps)

    # rebuild target delta with smoothed magnitude and (optionally) smoothed direction
    delta_f = (2.0 * Hf * Mdiag).unsqueeze(1) * direction

    # --- reconstruct: (L^T L + lam I) x = L^T delta_f + lam x0  (L symmetric) ---
    scale = float((Lval * Lval).mean().item())   # ~ magnitude of (L^T L) diagonal
    lam = float(anchor_weight) * scale
    b = spmm(Lsp, delta_f) + lam * X0

    def matvec(X):
        return spmm(Lsp, spmm(Lsp, X)) + lam * X

    # diagonal (Jacobi) preconditioner: diag(L^T L)_i = sum_k L_ik^2
    diagLL = torch.zeros(n, device=dev)
    diagLL.index_add_(0, Lidx[0], Lval * Lval)
    Minv = 1.0 / (diagLL + lam)

    # preconditioned CG over the 3 position columns simultaneously
    X = X0.clone()
    r = b - matvec(X)
    z = Minv.unsqueeze(1) * r
    p = z.clone()
    rz = (r * z).sum(0)
    r0 = torch.sqrt((r * r).sum(0)).max().item()
    used = 0
    for _ in range(int(cg_iters)):
        used += 1
        Ap = matvec(p)
        alpha = rz / ((p * Ap).sum(0) + 1e-20)
        X = X + p * alpha
        r = r - Ap * alpha
        if torch.sqrt((r * r).sum(0)).max().item() < 1e-6 * (r0 + 1e-12):
            break
        z = Minv.unsqueeze(1) * r
        rz_new = (r * z).sum(0)
        p = z + p * (rz_new / (rz + 1e-20))
        rz = rz_new

    Vout = X.detach().cpu().numpy().astype(np.float64)
    result = trimesh_module.Trimesh(vertices=Vout, faces=np.asarray(mesh.faces, dtype=np.int32), process=False)
    info_dev = f"{dev} (cg {used} iters)"
    return result, "", info_dev


class SharpenCurvatureGuidedNode(io.ComfyNode):
    """Curvature-domain guided sharpening backend (prototype, GPU)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpen_CurvatureGuided",
            display_name="Sharpen Curvature Guided (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("iterations", default=5, min=0, max=100, step=1, tooltip=(
                    "Edge-aware diffusion passes on the curvature field. More = stronger "
                    "curvature agreement within regions / wider reach. 0 = identity.")),
                io.Float.Input("sigma_s", default=2.0, min=0.1, max=10.0, step=0.1, tooltip=(
                    "Spatial scale (x average edge length) for the curvature diffusion.")),
                io.Float.Input("sigma_r_degrees", default=20.0, min=1.0, max=120.0, step=1.0, tooltip=(
                    "Range scale in DEGREES: normal-difference angle at which curvature "
                    "STOPS diffusing across an edge. Smaller = sharper feature preservation.")),
                io.Float.Input("anchor_weight", default=0.1, min=0.001, max=10.0, step=0.001, display_mode="number", tooltip=(
                    "How strongly the reconstruction sticks to the input positions "
                    "(Tikhonov lambda, relative to the Laplacian scale). LOWER = stronger "
                    "curvature-domain reshaping; HIGHER = stay close to input. Default 0.1.")),
                io.Float.Input("direction_blend", default=0.0, min=0.0, max=1.0, step=0.05, tooltip=(
                    "0 = filter curvature MAGNITUDE only, keeping each vertex's own normal "
                    "direction -> spheres/cylinders are a fixed point (no flattening), but "
                    "positional noise survives. >0 mixes in an edge-aware smoothed direction "
                    "to also denoise positions, at some flattening risk in curved regions. "
                    "Try 0.3-0.6 to denoise; 0 to purely uniformize curvature.")),
                io.Int.Input("cg_iters", default=200, min=10, max=2000, step=10, tooltip=(
                    "Max preconditioned-CG iterations for the reconstruction solve. "
                    "Increase if the result looks under-converged on large meshes.")),
                io.Combo.Input("use_gpu", options=["true", "false"], default="true", tooltip=(
                    "Run on CUDA (recommended -- this backend is torch sparse mat-vecs). "
                    "false = CPU torch (works, slower).")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, iterations=5, sigma_s=2.0, sigma_r_degrees=20.0,
                anchor_weight=0.1, direction_blend=0.0, cg_iters=200, use_gpu="true"):
        import math
        import time
        sigma_r = 2.0 * math.sin(math.radians(sigma_r_degrees) / 2.0)
        iv, ifc = len(trimesh.vertices), len(trimesh.faces)
        log.info("Backend: curvature_guided | %d verts %d faces | iters=%d sigma_s=%.2f sigma_r=%.1fdeg anchor=%.3f dirblend=%.2f gpu=%s",
                 iv, ifc, iterations, sigma_s, sigma_r_degrees, anchor_weight, direction_blend, use_gpu)

        t0 = time.perf_counter()
        sharpened, error, dev = _curvature_guided(
            trimesh, iterations, sigma_s, sigma_r, anchor_weight, cg_iters, use_gpu == "true",
            direction_blend=direction_blend)
        elapsed = time.perf_counter() - t0
        if sharpened is None:
            raise ValueError(f"Sharpening failed (curvature_guided): {error}")

        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": "curvature_guided", "device": str(dev),
            "iterations": iterations, "sigma_s": sigma_s,
            "sigma_r_degrees": sigma_r_degrees, "anchor_weight": anchor_weight,
        }

        disp = np.linalg.norm(np.asarray(sharpened.vertices) - np.asarray(trimesh.vertices), axis=1)
        info = (f"Sharpen Mesh Results (curvature_guided, device={dev}):\n\n"
                f"Iterations: {iterations}\nSigma S: {sigma_s}\nSigma R: {sigma_r_degrees} deg\n"
                f"Anchor weight: {anchor_weight}\nDirection blend: {direction_blend}\n"
                f"Time: {elapsed:.2f}s\n\n"
                f"Vertices: {iv:,} (unchanged)\nFaces: {ifc:,} (unchanged)\n\n"
                f"Displacement:\n  Average: {float(np.mean(disp)):.6f}\n  Maximum: {float(np.max(disp)):.6f}\n")
        log.info("[curvature_guided] device=%s avg_disp=%.6f", dev, float(np.mean(disp)))
        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_CurvatureGuided": SharpenCurvatureGuidedNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_CurvatureGuided": "Sharpen Curvature Guided (backend)"}
