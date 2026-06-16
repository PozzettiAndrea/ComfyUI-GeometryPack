# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""GPU/vectorized L0 normal-minimization sharpening backend (He & Schaefer 2013).

Same scheme as the CPU `l0_minimize` backend but fully vectorized with torch
(runs on CUDA when available, else vectorized CPU torch). The CPU backend loops
in Python over every adjacency edge and every face/vertex each iteration -- here
those become index_add_ scatter ops, so it is orders of magnitude faster on
large meshes. The per-edge snap is done as a symmetric area-weighted bilateral
accumulation (order-independent), which matches the CPU intent more cleanly.
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _torch_device():
    import torch
    try:
        import comfy.model_management as mm
        dev = mm.get_torch_device()
        if dev is not None:
            return dev
    except Exception:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _l0_minimize_gpu(mesh, alpha, beta, iterations):
    import torch

    V0 = np.asarray(mesh.vertices, dtype=np.float64)
    Ff = np.asarray(mesh.faces, dtype=np.int64)
    if len(V0) == 0 or len(Ff) == 0:
        return None, "Empty mesh (no vertices or faces).", "cpu"
    adj = np.asarray(mesh.face_adjacency)
    if len(adj) == 0:
        return None, "Mesh has no face adjacency (disconnected or degenerate).", "cpu"

    dev = _torch_device()
    eps = 1e-12
    V = torch.as_tensor(V0, dtype=torch.float32, device=dev)
    F = torch.as_tensor(Ff, dtype=torch.long, device=dev)
    fi = torch.as_tensor(adj[:, 0], dtype=torch.long, device=dev)
    fj = torch.as_tensor(adj[:, 1], dtype=torch.long, device=dev)
    nF = F.shape[0]
    nV = V.shape[0]
    ones_f = torch.ones(nF, device=dev)
    cur_alpha = float(alpha)

    for _ in range(int(iterations)):
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        cross = torch.cross(v1 - v0, v2 - v0, dim=1)
        area2 = cross.norm(dim=1, keepdim=True)
        normals = cross / (area2 + eps)
        areas = area2.squeeze(1) * 0.5
        centroids = (v0 + v1 + v2) / 3.0

        # snap: area-weighted bilateral over edges whose normal diff^2 < alpha
        diff_sq = ((normals[fi] - normals[fj]) ** 2).sum(1)
        mask = diff_sq < cur_alpha
        mfi, mfj = fi[mask], fj[mask]
        acc = normals * areas.unsqueeze(1)        # self (area-weighted)
        wsum = areas.clone()
        acc.index_add_(0, mfi, normals[mfj] * areas[mfj].unsqueeze(1))
        wsum.index_add_(0, mfi, areas[mfj])
        acc.index_add_(0, mfj, normals[mfi] * areas[mfi].unsqueeze(1))
        wsum.index_add_(0, mfj, areas[mfi])
        tn = acc / (wsum.unsqueeze(1) + eps)
        tn = tn / (tn.norm(dim=1, keepdim=True) + eps)

        # vertex update: p <- mean over incident faces of (p - dot(p-c, n) n)
        new_V = torch.zeros_like(V)
        counts = torch.zeros(nV, device=dev)
        for k in range(3):
            vid = F[:, k]
            Vv = V[vid]
            d = ((Vv - centroids) * tn).sum(1, keepdim=True)
            new_V.index_add_(0, vid, Vv - d * tn)
            counts.index_add_(0, vid, ones_f)
        moved = counts > 0
        new_V[moved] /= counts[moved].unsqueeze(1)
        new_V[~moved] = V[~moved]
        V = new_V
        cur_alpha *= float(beta)

    Vout = V.detach().cpu().numpy().astype(np.float64)
    result = trimesh_module.Trimesh(vertices=Vout,
                                    faces=np.asarray(mesh.faces, dtype=np.int32),
                                    process=False)
    return result, "", str(dev)


class SharpenL0MinimizeGPUNode(io.ComfyNode):
    """GPU/vectorized L0 normal minimization sharpening backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpen_L0MinimizeGPU",
            display_name="Sharpen L0 Minimize GPU (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("alpha", default=0.001, min=0.0001, max=0.1, step=0.0001,
                    tooltip="Initial normal-difference threshold (squared). Larger = more "
                            "aggressive initial flattening."),
                io.Float.Input("beta", default=2.0, min=1.1, max=10.0, step=0.1,
                    tooltip="Growth rate of the threshold each iteration."),
                io.Int.Input("iterations", default=10, min=1, max=50, step=1,
                    tooltip="Number of L0 iterations."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, alpha=0.001, beta=2.0, iterations=10):
        log.info("Backend: l0_minimize_gpu | %d verts, %d faces | alpha=%.4f beta=%.1f iters=%d",
                 len(trimesh.vertices), len(trimesh.faces), alpha, beta, iterations)
        iv, ifc = len(trimesh.vertices), len(trimesh.faces)

        sharpened, error, dev = _l0_minimize_gpu(trimesh, alpha, beta, iterations)
        if sharpened is None:
            raise ValueError(f"Sharpening failed (l0_minimize_gpu): {error}")

        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {"algorithm": "l0_minimize_gpu",
                                            "original_vertices": iv, "original_faces": ifc}

        disp = np.linalg.norm(np.asarray(sharpened.vertices) - np.asarray(trimesh.vertices), axis=1)
        info = (f"Sharpen Mesh Results (l0_minimize_gpu, device={dev}):\n\n"
                f"Alpha: {alpha}\nBeta: {beta}\nIterations: {iterations}\n\n"
                f"Vertices: {iv:,} (unchanged)\nFaces: {ifc:,} (unchanged)\n\n"
                f"Displacement:\n  Average: {float(np.mean(disp)):.6f}\n  Maximum: {float(np.max(disp)):.6f}\n")
        log.info("[l0_minimize_gpu] device=%s avg_disp=%.6f", dev, float(np.mean(disp)))
        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_L0MinimizeGPU": SharpenL0MinimizeGPUNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_L0MinimizeGPU": "Sharpen L0 Minimize GPU (backend)"}
