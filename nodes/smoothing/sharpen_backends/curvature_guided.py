# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Curvature-domain guided sharpening backend (GPU).

The second-order analogue of guided_normal. Where guided_normal filters face
NORMALS (first order) and reconstructs by plane projection, this regularises the
mesh CURVATURE field and reconstructs by a Laplacian-domain solve.

Pipeline (all on torch / CUDA):
  1. delta = L x  -- differential (Laplacian) coordinates; delta_i is the (integrated)
     mean-curvature NORMAL (|delta_i| ~ 2 H_i A_i). We work with the signed VECTOR delta
     (not |H|): filtering the magnitude rectifies noise to a positive floor and injects
     curvature, whereas the signed vector lets opposite-pointing noise cancel.
  2. Regularise the delta field, two modes:
       - 'tv'        : TOTAL-VARIATION denoising via Chambolle-Pock
                       (min 1/2||u-delta||^2 + alpha*sum_edges||u_j-u_i||). L1 of the
                       gradient => PIECEWISE-CONSTANT curvature with crisp jumps = genuine
                       regions of constant curvature (planes/cylinders/spheres).
       - 'bilateral' : edge-aware bilateral diffusion of delta (curvature-range weighted).
                       Denoises (variance down) but yields smooth RAMPS, not plateaus.
  3. Reconstruct positions: minimise ||L x - delta_f||^2 + lambda||x - x0||^2 via CGLS on
     the stacked [L; sqrt(lambda) I] -- never forms L^T L (whose condition number ~1/h^4
     under-converges in float32).

References: differential-coordinate / Laplacian surface editing (Sorkine et al. 2004;
Lipman et al. 2004); curvature-domain shape processing (Eigensatz, Sumner, Pauly 2008);
TV denoising (Rudin-Osher-Fatemi 1992; Chambolle-Pock 2011). Bilateral mode mirrors
guided normal filtering (Zhang et al. 2015) one order up; TV mode is the piecewise-
constant analog (the L0/TV second-order twin of piecewise-flat normals)."""

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


def _curvature_guided(mesh, iterations, sigma_s, curvature_sigma, anchor_weight, cg_iters, use_gpu,
                      regularizer="tv", tv_weight=0.5):
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

    delta = spmm(Lsp, X0)                       # (n,3) differential coords (mean-curv normal)
    dmag = delta.norm(dim=1)                    # (n,)
    Hmag = dmag / (2.0 * Mdiag)                 # pointwise |mean curvature| (for guidance)

    # --- 1-ring directed edges (both directions) for the bilateral filter ---
    E = np.vstack([Ff[:, [0, 1]], Ff[:, [1, 2]], Ff[:, [2, 0]]])
    ed = np.unique(np.vstack([E, E[:, [1, 0]]]), axis=0)
    src = torch.tensor(ed[:, 0], dtype=torch.long, device=dev)   # accumulate into i
    dst = torch.tensor(ed[:, 1], dtype=torch.long, device=dev)   # neighbour j

    avg_edge = float(np.mean(np.linalg.norm(V0[E[:, 0]] - V0[E[:, 1]], axis=1)))
    ss2 = 2.0 * (sigma_s * avg_edge) ** 2
    dpos = (X0[src] - X0[dst]).pow(2).sum(1)
    ws = torch.exp(-dpos / (ss2 + eps))            # spatial weight (E,)
    ws_sum = torch.ones(n, device=dev)
    ws_sum.index_add_(0, src, ws)

    # GUIDANCE curvature: a couple of spatial-only smoothing passes of |H| give a
    # robust region signal -- the second-order analog of guided_normal's guidance
    # normal (a denoised estimate of "what curvature is this region").
    Hguide = Hmag.clone()
    for _ in range(2):
        acc = Hguide.clone()
        acc.index_add_(0, src, ws * Hguide[dst])
        Hguide = acc / ws_sum

    # RANGE weight on CURVATURE difference (data-scaled): diffuse curvature WITHIN a
    # curvature-region and STOP at curvature steps (e.g. flat<->fillet). This is the
    # faithful second-order analog of guided_normal's normal range weight -- it pushes
    # the surface toward PIECEWISE-CONSTANT CURVATURE, keeping the region boundaries
    # crisp instead of blurring them.
    curv_scale = torch.clamp(Hguide.std(), min=eps)
    sigma_curv2 = 2.0 * (curvature_sigma * curv_scale) ** 2
    dcurv = (Hguide[src] - Hguide[dst]).pow(2)
    wr = torch.exp(-dcurv / (sigma_curv2 + eps))
    w = ws * wr                                    # (E,) spatial x curvature-range
    wsum = torch.ones(n, device=dev)               # self weight 1.0
    wsum.index_add_(0, src, w)

    # --- edge-aware diffusion of the mean-curvature-normal VECTOR (delta) ---
    # Filter the SIGNED vector, not the magnitude |H|: opposite-pointing noise then
    # CANCELS. A noisy flat (delta = random directions) averages toward 0 -> truly flat
    # (spurious bumps removed); a noisy curved region averages toward its consistent
    # curvature normal -> filled. (Averaging |H|, a magnitude, instead rectifies noise to
    # a positive floor and INJECTS curvature everywhere -- the bug this replaces.) The
    # curvature-range weight `w` stops diffusion at curvature steps, so a real fillet next
    # to a flat keeps its curvature instead of being flattened by the flat.
    if regularizer == "tv":
        # --- TOTAL-VARIATION denoise of the curvature-normal field (Chambolle-Pock) ----
        # min_u 1/2||u - delta||^2 + alpha * sum_{edges} ||u_j - u_i||_2.  The L1 of the
        # gradient yields a PIECEWISE-CONSTANT field with crisp jumps -> genuine regions of
        # constant curvature (a bilateral low-pass only makes ramps, never plateaus). Fully
        # matrix-free: gradient = u[j]-u[i], transpose = scatter. (Chambolle & Pock 2011.)
        und = np.unique(np.sort(np.vstack([Ff[:, [0, 1]], Ff[:, [1, 2]], Ff[:, [2, 0]]]), axis=1), axis=0)
        ui = torch.tensor(und[:, 0], dtype=torch.long, device=dev)
        uj = torch.tensor(und[:, 1], dtype=torch.long, device=dev)
        ne = und.shape[0]
        deg = torch.zeros(n, device=dev)
        deg.index_add_(0, ui, torch.ones(ne, device=dev))
        deg.index_add_(0, uj, torch.ones(ne, device=dev))
        Knorm = float(torch.sqrt(2.0 * deg.max()).clamp(min=1.0).item())   # ||grad|| bound
        tau = 1.0 / Knorm
        sig = 1.0 / Knorm
        med = float(delta.norm(dim=1).median().clamp(min=eps).item())
        alpha = float(tv_weight) * med            # data-scaled => mesh-independent strength
        u = delta.clone()
        ubar = u.clone()
        pdual = torch.zeros((ne, 3), device=dev)
        n_cp = max(50, int(iterations) * 30)
        for _ in range(n_cp):
            pdual = pdual + sig * (ubar[uj] - ubar[ui])         # dual ascent (gradient)
            pn = pdual.norm(dim=1, keepdim=True).clamp(min=eps)
            pdual = pdual * torch.clamp(alpha / pn, max=1.0)    # project onto ||.||_2 <= alpha
            KTp = torch.zeros((n, 3), device=dev)               # divergence (grad^T)
            KTp.index_add_(0, ui, -pdual)
            KTp.index_add_(0, uj, pdual)
            v = u - tau * KTp
            u_new = (tau * delta + v) / (tau + 1.0)             # prox of 1/2||u-delta||^2
            ubar = u_new + (u_new - u)                          # over-relax (theta=1)
            u = u_new
        delta_f = u
        used_filter = f"tv({n_cp})"
    else:
        # --- edge-aware bilateral diffusion of the delta VECTOR (denoise, not constancy) -
        delta_f = delta.clone()
        for _ in range(int(iterations)):
            acc = delta_f.clone()
            acc.index_add_(0, src, w.unsqueeze(1) * delta_f[dst])
            delta_f = acc / wsum.unsqueeze(1)
        used_filter = f"bilateral({int(iterations)})"

    # --- reconstruct: minimize ||L x - delta_f||^2 + lam ||x - x0||^2 via CGLS on the
    # STACKED operator [L; sqrt(lam) I]. Never forms L^T L (which squares the condition
    # number ~1/h^4 and silently under-converges in float32). ---
    scale = float((Lval * Lval).mean().item())
    lam = float(anchor_weight) * scale
    sl = lam ** 0.5

    def Aop(x):                      # A x = [L x ; sqrt(lam) x]
        return spmm(Lsp, x), sl * x

    def ATop(y1, y2):                # A^T [y1; y2] = L y1 + sqrt(lam) y2   (L symmetric)
        return spmm(Lsp, y1) + sl * y2

    X = X0.clone()
    r1 = delta_f - spmm(Lsp, X)
    r2 = sl * X0 - sl * X
    s = ATop(r1, r2)
    p = s.clone()
    gamma = (s * s).sum(0)
    g0 = torch.sqrt(gamma).max().item()
    used = 0
    for _ in range(int(cg_iters)):
        used += 1
        q1, q2 = Aop(p)
        a = gamma / ((q1 * q1).sum(0) + (q2 * q2).sum(0) + 1e-20)
        X = X + p * a
        r1 = r1 - q1 * a
        r2 = r2 - q2 * a
        s = ATop(r1, r2)
        g2 = (s * s).sum(0)
        if torch.sqrt(g2).max().item() < 1e-7 * (g0 + 1e-12):
            break
        p = s + p * (g2 / (gamma + 1e-20))
        gamma = g2

    Vout = X.detach().cpu().numpy().astype(np.float64)
    result = trimesh_module.Trimesh(vertices=Vout, faces=np.asarray(mesh.faces, dtype=np.int32), process=False)
    info_dev = f"{dev} ({used_filter}, cgls {used} iters)"
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
                io.Combo.Input("regularizer", options=["tv", "bilateral"], default="tv", tooltip=(
                    "tv = TOTAL-VARIATION (Chambolle-Pock) on the curvature field -> "
                    "PIECEWISE-CONSTANT curvature with crisp jumps: genuine regions of constant "
                    "curvature (planes/cylinders/spheres). bilateral = edge-aware diffusion: "
                    "denoises but only makes smooth ramps, not plateaus. Use tv for "
                    "region/primitive structure, bilateral for gentle denoise.")),
                io.Float.Input("tv_weight", default=0.5, min=0.02, max=8.0, step=0.02, display_mode="number", tooltip=(
                    "TV strength (tv mode only), relative to the median curvature. HIGHER = "
                    "flatter, fewer/larger constant-curvature regions (merges fine variation); "
                    "LOWER = more, smaller regions (keeps detail). Default 0.5.")),
                io.Int.Input("iterations", default=5, min=0, max=100, step=1, tooltip=(
                    "tv mode: scales the number of Chambolle-Pock passes (~iterations x 30). "
                    "bilateral mode: edge-aware diffusion passes. More = stronger / wider reach.")),
                io.Float.Input("sigma_s", default=2.0, min=0.1, max=10.0, step=0.1, tooltip=(
                    "(bilateral mode only) Spatial scale (x average edge length).")),
                io.Float.Input("curvature_sigma", default=0.5, min=0.02, max=5.0, step=0.02, display_mode="number", tooltip=(
                    "(bilateral mode only) CURVATURE range scale, relative to the mesh's curvature "
                    "spread. SMALLER = sharper region boundaries; LARGER = more cross-region "
                    "blending. (tv mode uses tv_weight instead.)")),
                io.Float.Input("anchor_weight", default=0.1, min=0.001, max=10.0, step=0.001, display_mode="number", tooltip=(
                    "How strongly the reconstruction sticks to the input positions "
                    "(Tikhonov lambda, relative to the Laplacian scale). LOWER = stronger "
                    "curvature-domain reshaping; HIGHER = stay close to input. Default 0.1.")),
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
    def execute(cls, trimesh, regularizer="tv", tv_weight=0.5, iterations=5, sigma_s=2.0,
                curvature_sigma=0.5, anchor_weight=0.1, cg_iters=200, use_gpu="true"):
        import time
        iv, ifc = len(trimesh.vertices), len(trimesh.faces)
        log.info("Backend: curvature_guided | %d verts %d faces | reg=%s tv_w=%.2f iters=%d sigma_s=%.2f curv_sigma=%.2f anchor=%.3f gpu=%s",
                 iv, ifc, regularizer, tv_weight, iterations, sigma_s, curvature_sigma, anchor_weight, use_gpu)

        t0 = time.perf_counter()
        sharpened, error, dev = _curvature_guided(
            trimesh, iterations, sigma_s, curvature_sigma, anchor_weight, cg_iters, use_gpu == "true",
            regularizer=regularizer, tv_weight=tv_weight)
        elapsed = time.perf_counter() - t0
        if sharpened is None:
            raise ValueError(f"Sharpening failed (curvature_guided): {error}")

        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": "curvature_guided", "device": str(dev),
            "regularizer": regularizer, "tv_weight": tv_weight,
            "iterations": iterations, "sigma_s": sigma_s,
            "curvature_sigma": curvature_sigma, "anchor_weight": anchor_weight,
        }

        disp = np.linalg.norm(np.asarray(sharpened.vertices) - np.asarray(trimesh.vertices), axis=1)
        info = (f"Sharpen Mesh Results (curvature_guided, device={dev}):\n\n"
                f"Regularizer: {regularizer} (tv_weight={tv_weight})\n"
                f"Iterations: {iterations}\nAnchor weight: {anchor_weight}\n"
                f"Time: {elapsed:.2f}s\n\n"
                f"Vertices: {iv:,} (unchanged)\nFaces: {ifc:,} (unchanged)\n\n"
                f"Displacement:\n  Average: {float(np.mean(disp)):.6f}\n  Maximum: {float(np.max(disp)):.6f}\n")
        log.info("[curvature_guided] device=%s avg_disp=%.6f", dev, float(np.mean(disp)))
        if len(sharpened.vertices) == len(trimesh.vertices):
            sharpened.vertex_attributes["sharpen_displacement_magnitude"] = disp.astype(np.float32)

        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_CurvatureGuided": SharpenCurvatureGuidedNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_CurvatureGuided": "Sharpen Curvature Guided (backend)"}
