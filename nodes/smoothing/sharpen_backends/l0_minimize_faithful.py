# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Faithful He & Schaefer 2013 "Mesh Denoising via L0 Minimization".

Unlike the local-normal-threshold `l0_minimize` backend, this is the REAL method:
a GLOBAL solve. It minimizes  |p - p*|^2 + alpha*|R(p)|^2 + beta*|D(p) - delta|^2 + lambda*|delta|_0
by half-quadratic splitting:

  * D = the area-based EDGE operator (paper Eq. 2): a linear, translation-invariant
    operator over each interior edge's 4-vertex "diamond" that is exactly 0 when the
    two triangles are coplanar and large at a crease. (4 nonzeros per edge.)
  * R = the {+1,-1,+1,-1} edge regularizer (Eq. 3) that stops triangles folding.
  * delta-step (Eq. 5): hard-threshold each edge -- keep D_e if ||D_e||^2 >= lambda/beta,
    else 0. THIS is the L0: it counts nonzero edges, so flat regions go to 0 (flatten)
    while the few real creases survive (stay sharp).
  * p-step (Eq. 6): solve the global sparse SPD system (I + alpha*R^T R + beta*D^T D) p
    = p* + beta*D^T delta  -- one linear solve over the whole mesh per iteration.
  * continuation: beta *= mu each iteration (10^-3 -> 10^3); alpha *= 0.5 (fades).

lambda is scale-normalised:  lambda = gamma_factor * avg_edge_len^2 * avg_dihedral.

Validated on a synthetic 90-degree creased patch: faces flatten to <1 deg off-axis while
the crease stays >88 deg (gamma_factor=0.2). Topology is unchanged (vertices only move),
so vertex/face attributes (e.g. cad_face_id) are carried through.

CPU (scipy sparse). Cost ~ one sparse solve x ~40 iterations; decimate very large meshes
first. Refs: He & Schaefer, ACM TOG 32(4) 2013; reference impl bldeng/GuidedDenoising.
"""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _diamonds(mesh):
    """Interior-edge diamonds: (v1,v3)=shared edge, v2=apex of face a, v4=apex of face b."""
    F = np.asarray(mesh.faces)
    adj = np.asarray(mesh.face_adjacency)
    fae = np.asarray(mesh.face_adjacency_edges)
    v1, v3 = fae[:, 0], fae[:, 1]
    v2 = F[adj[:, 0]].sum(1) - v1 - v3
    v4 = F[adj[:, 1]].sum(1) - v1 - v3
    return v1, v2, v3, v4


def _D_coefs(V, v1, v2, v3, v4):
    """Area-based edge-operator coefficients on (v1,v2,v3,v4); rows sum to 0."""
    P1, P2, P3, P4 = V[v1], V[v2], V[v3], V[v4]
    a123 = 0.5 * np.linalg.norm(np.cross(P2 - P1, P3 - P1), axis=1)
    a134 = 0.5 * np.linalg.norm(np.cross(P3 - P1, P4 - P1), axis=1)
    tot = a123 + a134 + 1e-20
    p13 = P1 - P3; p34 = P3 - P4; p23 = P2 - P3; p14 = P1 - P4; p12 = P1 - P2
    L2 = np.sum(p13 * p13, axis=1) + 1e-20
    dot = lambda a, b: np.sum(a * b, axis=1)
    c0 = (a123 * dot(p34, p13) - a134 * dot(p13, p23)) / (L2 * tot)
    c1 = a134 / tot
    c2 = (-a123 * dot(p13, p14) - a134 * dot(p12, p13)) / (L2 * tot)
    c3 = a123 / tot
    return np.stack([c0, c1, c2, c3], axis=1)


def _l0_faithful(mesh, gamma_factor, alpha_factor, mu, beta0, beta_max):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    V = np.asarray(mesh.vertices, dtype=np.float64)
    if len(V) == 0 or len(mesh.faces) == 0:
        return None, 0, "Empty mesh."
    v1, v2, v3, v4 = _diamonds(mesh)
    E = len(v1)
    if E == 0:
        return None, 0, "No interior edges (mesh is all boundary / disconnected)."
    n = len(V)
    cols = np.stack([v1, v2, v3, v4], axis=1)
    rows = np.repeat(np.arange(E), 4)

    le = float(np.mean(mesh.edges_unique_length))
    gbar = float(np.mean(np.abs(mesh.face_adjacency_angles)))
    lam = float(gamma_factor) * le * le * gbar
    alpha = float(alpha_factor) * gbar

    Rdata = np.tile(np.array([1.0, -1.0, 1.0, -1.0]), E)
    R = sp.coo_matrix((Rdata, (rows, cols.ravel())), shape=(E, n)).tocsr()
    RtR = (R.T @ R).tocsc()
    I = sp.identity(n, format="csc")

    Pstar = V.copy()
    p = V.copy()
    beta = float(beta0)
    iters = 0
    while beta < float(beta_max):
        coefs = _D_coefs(p, v1, v2, v3, v4)
        D = sp.coo_matrix((coefs.ravel(), (rows, cols.ravel())), shape=(E, n)).tocsr()
        Dp = D @ p
        mag2 = np.sum(Dp * Dp, axis=1)
        delta = np.where((mag2 >= lam / beta)[:, None], Dp, 0.0)
        A = (I + alpha * RtR + beta * (D.T @ D)).tocsc()
        b = Pstar + beta * (D.T @ delta)
        p = spl.spsolve(A, b)
        beta *= float(mu)
        alpha *= 0.5
        iters += 1

    out = trimesh_module.Trimesh(vertices=p, faces=np.asarray(mesh.faces, dtype=np.int32),
                                 process=False)
    return out, iters, ""


def _cg(matvec, B, X, tol=1e-5, maxiter=600):
    """Block conjugate gradient for SPD A (3 RHS columns share the matrix). Warm-started by X."""
    import torch
    R = B - matvec(X)
    P = R.clone()
    rs = (R * R).sum(0, keepdim=True)
    bnorm = B.norm(dim=0, keepdim=True).clamp_min(1e-30)
    for _ in range(maxiter):
        AP = matvec(P)
        a = rs / (P * AP).sum(0, keepdim=True).clamp_min(1e-30)
        X = X + a * P
        R = R - a * AP
        rs_new = (R * R).sum(0, keepdim=True)
        if bool(torch.all(rs_new.sqrt() / bnorm < tol)):
            break
        P = R + (rs_new / rs.clamp_min(1e-30)) * P
        rs = rs_new
    return X


def _l0_faithful_gpu(mesh, gamma_factor, alpha_factor, beta_mu, beta0, beta_max):
    """Same algorithm as _l0_faithful, on torch (CUDA if available, else CPU torch), with a
    warm-started CG solve instead of a sparse direct factorization. float32."""
    import torch

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None, 0, "cpu", "Empty mesh."
    v1, v2, v3, v4 = _diamonds(mesh)
    E = len(v1)
    if E == 0:
        return None, 0, "cpu", "No interior edges."
    n = len(mesh.vertices)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = torch.float32

    le = float(np.mean(mesh.edges_unique_length))
    gbar = float(np.mean(np.abs(mesh.face_adjacency_angles)))
    lam = float(gamma_factor) * le * le * gbar
    alpha = float(alpha_factor) * gbar

    cols = torch.as_tensor(np.ascontiguousarray(np.stack([v1, v2, v3, v4], 1).ravel()),
                           dtype=torch.long, device=dev)
    rows = torch.arange(E, device=dev).repeat_interleave(4)
    idx = torch.stack([rows, cols])
    vi = [torch.as_tensor(np.ascontiguousarray(x), dtype=torch.long, device=dev)
          for x in (v1, v2, v3, v4)]

    def spmm(A, x):
        return torch.sparse.mm(A, x)

    def build(values):
        return torch.sparse_coo_tensor(idx, values, (E, n)).coalesce()

    Rvals = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=f, device=dev).repeat(E)
    R = build(Rvals); Rt = R.t().coalesce()

    Pstar = torch.as_tensor(np.asarray(mesh.vertices), dtype=f, device=dev)
    p = Pstar.clone()
    beta = float(beta0); iters = 0
    while beta < float(beta_max):
        P1, P2, P3, P4 = p[vi[0]], p[vi[1]], p[vi[2]], p[vi[3]]
        a123 = 0.5 * torch.linalg.norm(torch.cross(P2 - P1, P3 - P1, dim=1), dim=1)
        a134 = 0.5 * torch.linalg.norm(torch.cross(P3 - P1, P4 - P1, dim=1), dim=1)
        tot = a123 + a134 + 1e-20
        p13 = P1 - P3; p34 = P3 - P4; p23 = P2 - P3; p14 = P1 - P4; p12 = P1 - P2
        L2 = (p13 * p13).sum(1) + 1e-20
        dot = lambda x, y: (x * y).sum(1)
        c0 = (a123 * dot(p34, p13) - a134 * dot(p13, p23)) / (L2 * tot)
        c1 = a134 / tot
        c2 = (-a123 * dot(p13, p14) - a134 * dot(p12, p13)) / (L2 * tot)
        c3 = a123 / tot
        coefs = torch.stack([c0, c1, c2, c3], 1).reshape(-1)
        D = build(coefs); Dt = D.t().coalesce()

        Dp = spmm(D, p)
        mag2 = (Dp * Dp).sum(1)
        delta = torch.where((mag2 >= lam / beta)[:, None], Dp, torch.zeros_like(Dp))
        b = Pstar + beta * spmm(Dt, delta)
        matvec = lambda x: x + alpha * spmm(Rt, spmm(R, x)) + beta * spmm(Dt, spmm(D, x))
        p = _cg(matvec, b, p)
        beta *= float(beta_mu); alpha *= 0.5; iters += 1

    out = trimesh_module.Trimesh(vertices=p.detach().cpu().numpy().astype(np.float64),
                                 faces=np.asarray(mesh.faces, dtype=np.int32), process=False)
    return out, iters, ("cuda" if dev.type == "cuda" else "cpu"), ""


class SharpenL0FaithfulNode(io.ComfyNode):
    """Faithful global He & Schaefer 2013 L0 minimization (edge operator + global solve)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpen_L0Faithful",
            display_name="Sharpen L0 Faithful (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("gamma_factor", default=0.2, min=0.0001, max=30.0, step=0.01,
                               display_mode="number", tooltip=(
                    "MAIN strength. Sets the L0 weight lambda = gamma_factor * avg_edge^2 * "
                    "avg_dihedral (scale-normalised). HIGHER = flattens harder / fewer surviving "
                    "creases (cleaner faces, but can erase shallow features); LOWER = gentler, "
                    "keeps more detail/noise. 0.2 = reference-code default (validated: faces -> "
                    "<1deg, a 90deg crease stays >88deg). 0.02 = the paper's milder value.")),
                io.Float.Input("beta_mu", default=1.414, min=1.05, max=3.0, step=0.01, tooltip=(
                    "beta continuation multiplier (beta *= beta_mu each iteration, 1e-3 -> 1e3). "
                    "SMALLER = more iterations = finer schedule = better features but slower "
                    "(~#iters = ln(1e6)/ln(mu): 1.414->40, 2.0->20, 1.09->160). Paper default "
                    "sqrt(2)~=1.414.")),
                io.Float.Input("alpha_factor", default=0.1, min=0.0, max=2.0, step=0.01, optional=True,
                               tooltip=(
                    "Initial vertex regularizer alpha0 = alpha_factor * avg_dihedral, halved each "
                    "iteration (anti-fold / fairing term R, Eq.3). Decays fast so it mostly "
                    "stabilises early iterations; 0 disables it. Paper default 0.1.")),
                io.Float.Input("beta_max", default=1000.0, min=10.0, max=100000.0, step=10.0,
                               optional=True, tooltip=(
                    "Stop when beta reaches this (paper 1e3). Higher = runs longer, locks in more "
                    "of the current state (preserves residual detail); rarely needs changing.")),
                io.Combo.Input("use_gpu", options=["true", "false"], default="true", optional=True,
                               tooltip=(
                    "true = torch solver (CUDA if available, else CPU torch) with a warm-started "
                    "conjugate-gradient solve -- scales to large meshes (float32, results differ "
                    "slightly). false = exact scipy sparse-direct solve (SuperLU, float64) -- best "
                    "for small/medium meshes; gets slow past ~100k vertices since A is refactorised "
                    "every iteration.")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, gamma_factor=0.2, beta_mu=1.414, alpha_factor=0.1, beta_max=1000.0,
                use_gpu="true"):
        import time
        gpu = (use_gpu == "true")
        log.info("Backend: l0_minimize_faithful (%s)", "gpu" if gpu else "cpu-direct")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: gamma_factor=%.4f, beta_mu=%.3f, alpha_factor=%.3f, beta_max=%.1f, gpu=%s",
                 gamma_factor, beta_mu, alpha_factor, beta_max, use_gpu)

        n0, f0 = len(trimesh.vertices), len(trimesh.faces)
        t0 = time.perf_counter()
        device = "cpu"
        if gpu:
            try:
                out, iters, device, err = _l0_faithful_gpu(trimesh, gamma_factor, alpha_factor,
                                                            beta_mu, 1e-3, beta_max)
            except Exception as e:
                log.warning("GPU path failed (%s); falling back to scipy direct.", e)
                out, iters, err = _l0_faithful(trimesh, gamma_factor, alpha_factor, beta_mu, 1e-3, beta_max)
        else:
            out, iters, err = _l0_faithful(trimesh, gamma_factor, alpha_factor, beta_mu, 1e-3, beta_max)
        elapsed = time.perf_counter() - t0
        if out is None:
            raise ValueError(f"Sharpening failed (l0_minimize_faithful): {err}")

        # Topology is unchanged (vertices only moved) -> carry metadata + attributes through.
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            out.metadata = trimesh.metadata.copy()
        out.metadata["sharpening"] = {
            "algorithm": "l0_minimize_faithful",
            "gamma_factor": gamma_factor, "beta_mu": beta_mu, "alpha_factor": alpha_factor,
            "iterations": iters, "device": device, "original_vertices": n0, "original_faces": f0,
        }
        try:
            for k, v in dict(trimesh.vertex_attributes).items():
                out.vertex_attributes[k] = v
            for k, v in dict(trimesh.face_attributes).items():
                out.face_attributes[k] = v
        except Exception as e:
            log.warning("attribute carry-over skipped: %s", e)

        disp = np.linalg.norm(np.asarray(out.vertices) - np.asarray(trimesh.vertices), axis=1)
        out.vertex_attributes["sharpen_displacement_magnitude"] = disp.astype(np.float32)

        info = f"""Sharpen Mesh Results (l0_minimize_faithful, GLOBAL solve, device={device}):

gamma_factor (lambda scale): {gamma_factor}
beta_mu (beta multiplier): {beta_mu}   ->  {iters} iterations
alpha_factor: {alpha_factor}   beta_max: {beta_max}
Solver: {"GPU/CPU torch CG" if use_gpu == "true" else "scipy sparse-direct"}
Time: {elapsed:.2f}s

Vertices: {n0:,} (unchanged)   Faces: {f0:,} (unchanged)
Displacement: avg {float(np.mean(disp)):.6f}  max {float(np.max(disp)):.6f}

He & Schaefer 2013 edge-operator L0. Flat regions -> flat, creases stay sharp.
"""
        log.info("l0_minimize_faithful: %d iters, %.2fs, max disp %.6f", iters, elapsed, float(np.max(disp)))
        return io.NodeOutput(out, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_L0Faithful": SharpenL0FaithfulNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_L0Faithful": "Sharpen L0 Faithful (backend)"}
