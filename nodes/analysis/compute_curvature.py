# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Compute per-vertex curvature (mean, Gaussian, principal, shape index).

Two estimators:
  - quadric_fit (default): libigl principal_curvature -- fits a quadric over a
    k-ring neighborhood, returns principal curvatures k1>=k2 (robust on
    irregular CAD tessellations, gives directions). Mean = (k1+k2)/2,
    Gaussian = k1*k2.
  - ddg: discrete-differential-geometry operators -- mean curvature from the
    cotangent Laplacian mean-curvature-normal (Meyer et al. 2003) and Gaussian
    from the angle defect (Gauss-Bonnet). Fast (sparse mat-vec + scatter);
    principal curvatures recovered as H +/- sqrt(max(0, H^2 - K)).

All fields are attached to the output mesh as `vertex_attributes` so the field
viewers (Preview with scalar fields / VTP export) can colour them, plus a
canonical `curvature` field (your chosen primary) for driving adaptive remeshing.
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class ComputeCurvatureNode(io.ComfyNode):
    """Per-vertex curvature estimation (mean / Gaussian / principal / shape index)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackComputeCurvature",
            display_name="Compute Curvature",
            category="geompack/analysis",
            description=(
                "Estimate per-vertex curvature and attach it to the mesh as scalar "
                "vertex fields (viewable in the field preview / VTP export, and usable "
                "as a sizing field for curvature-adaptive remeshing).\n"
                "\n"
                "method = quadric_fit (libigl principal_curvature over a k-ring; robust, "
                "gives true principal curvatures k1>=k2) or ddg (cotangent-Laplacian mean "
                "curvature + angle-defect Gaussian; fast on huge meshes). Mean=(k1+k2)/2, "
                "Gaussian=k1*k2, and principal = H +/- sqrt(max(0,H^2-K)).\n"
                "\n"
                "Outputs fields: curvature_mean, curvature_gaussian, curvature_k1, "
                "curvature_k2, curvature_abs_max (MAGNITUDE of strongest curvature, >=0), "
                "curvature_dominant (same but signed: -=concave, +=convex), "
                "curvature_shape_index, plus a 'curvature' alias set to your chosen primary. radius controls the neighborhood (quadric "
                "only); larger = smoother/less noisy but blurs features. smoothing_iterations "
                "diffuses the field; clamp_percentile clips outliers (slivers) so they don't "
                "dominate the range."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Combo.Input("method", options=["quadric_fit", "ddg"], default="quadric_fit", tooltip=(
                    "quadric_fit = libigl principal_curvature (robust k-ring quadric, best for "
                    "CAD, gives principal directions). ddg = cotangent-Laplacian mean + "
                    "angle-defect Gaussian (fast on large meshes, scalar only).")),
                io.Combo.Input("primary_output", options=[
                    "mean", "gaussian", "max_principal", "min_principal",
                    "abs_max_principal", "dominant_signed", "angle_deg", "shape_index"
                ], default="mean", tooltip=(
                    "Which scalar becomes the canonical 'curvature' field (all fields are "
                    "attached regardless). abs_max_principal = MAGNITUDE of the strongest "
                    "principal curvature (>=0; ~0 flat, ~1/r on a fillet) -- threshold this to "
                    "separate flat from curved. dominant_signed = same but keeps sign "
                    "(negative=concave, positive=convex). angle_deg = curvature as a "
                    "DIHEDRAL-EQUIVALENT angle in degrees (turn over one edge length) -- "
                    "threshold it like a crease angle. shape_index classifies local shape "
                    "(-1 cup .. 0 saddle .. +1 cap).")),
                io.Int.Input("radius", default=5, min=1, max=12, step=1, tooltip=(
                    "Neighborhood size (k-ring, in average edge lengths) for the quadric fit. "
                    "Larger = smoother, more noise-robust, but blurs sharp features. ~3 for "
                    "crisp CAD, ~5 default, 7-8 for noisy scans. Ignored by the ddg method.")),
                io.Int.Input("smoothing_iterations", default=0, min=0, max=50, step=1, tooltip=(
                    "Laplacian (neighbour-averaging) smoothing of the curvature fields. 0 = off. "
                    "A few passes denoise the field without re-fitting. Stable at any resolution.")),
                io.Combo.Input("clamp_to_resolution", options=["true", "false"], default="true", tooltip=(
                    "AUTOMATICALLY cap curvature the mesh can't actually represent. A fillet "
                    "finer than the local triangles is just the fit blowing up at a sharp edge "
                    "/ sliver -- this caps |kappa| per vertex so the radius can't be smaller "
                    "than min_radius_edges x the LOCAL edge length, killing those spikes. "
                    "On by default (sampling theory: curvature is only valid where kappa*edge "
                    "is small).")),
                io.Float.Input("min_radius_edges", default=1.0, min=0.25, max=8.0, step=0.25, tooltip=(
                    "Resolution clamp strength: smallest trustworthy radius of curvature, in "
                    "units of the LOCAL edge length. 1.0 = cap where radius == local edge "
                    "(remove pure sub-triangle artifacts); 2-3 = stricter (only keep fillets "
                    "spanning several triangles). Only used when clamp_to_resolution is on.")),
                io.Float.Input("clamp_percentile", default=1.0, min=0.0, max=20.0, step=0.5, tooltip=(
                    "Additionally clip each field to its [p, 100-p] percentile range so a few "
                    "remaining outliers don't blow out the dynamic range. 0 = off. 1.0 safe.")),
                io.Combo.Input("preclean", options=["true", "false"], default="true", tooltip=(
                    "Merge duplicate vertices, drop degenerate faces, and fix normals before "
                    "estimating (recommended: degenerate triangles are the main failure mode, "
                    "and consistent normals are needed for the mean-curvature sign).")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh_with_curvature"),
                io.String.Output(display_name="info"),
            ],
        )

    @staticmethod
    def _shape_index(k1, k2):
        """Koenderink shape index s in [-1, 1] from principal curvatures (k1>=k2).
        s = (2/pi) atan((k2 + k1) / (k2 - k1)); umbilic (k1==k2) -> 0."""
        denom = k2 - k1
        s = np.zeros_like(k1)
        nz = np.abs(denom) > 1e-12
        s[nz] = (2.0 / np.pi) * np.arctan((k2[nz] + k1[nz]) / denom[nz])
        return s

    @classmethod
    def execute(cls, trimesh, method="quadric_fit", primary_output="mean",
                radius=5, smoothing_iterations=0, clamp_to_resolution="true",
                min_radius_edges=1.0, clamp_percentile=1.0, preclean="true"):
        import igl
        import scipy.sparse as sp

        mesh = trimesh.copy()
        if preclean == "true":
            try:
                mesh.merge_vertices()
            except Exception as e:
                log.debug("merge_vertices skipped: %s", e)
            try:
                mesh.update_faces(mesh.nondegenerate_faces())
                mesh.remove_unreferenced_vertices()
            except Exception as e:
                log.debug("degenerate-face cleanup skipped: %s", e)
            try:
                trimesh_module.repair.fix_normals(mesh)
            except Exception as e:
                log.debug("fix_normals skipped: %s", e)

        V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        n = len(V)
        log.info("Compute Curvature: method=%s, %d verts %d faces", method, n, len(F))

        # Mass + cotangent operators (used by ddg and by field smoothing).
        M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
        Mdiag = np.asarray(M.diagonal(), dtype=np.float64)
        Mdiag[Mdiag <= 0] = 1e-12
        Minv = sp.diags(1.0 / Mdiag)
        L = igl.cotmatrix(V, F)

        if method == "quadric_fit":
            out = igl.principal_curvature(V, F, int(max(1, radius)))
            k1 = np.asarray(out[2], dtype=np.float64)   # max principal
            k2 = np.asarray(out[3], dtype=np.float64)   # min principal
            mean = 0.5 * (k1 + k2)
            gaussian = k1 * k2
        else:  # ddg
            Kint = np.asarray(igl.gaussian_curvature(V, F), dtype=np.float64)
            gaussian = Minv @ Kint
            Hn = np.asarray(Minv @ (L @ V))             # mean-curvature normal = 2 H n
            VN = np.asarray(mesh.vertex_normals, dtype=np.float64)
            # Sign against the outward normal (igl cotmatrix is negative-semidefinite,
            # so the raw Hn points inward -> negate to make convex H > 0).
            mean = -0.5 * np.einsum('ij,ij->i', Hn, VN)
            root = np.sqrt(np.clip(mean * mean - gaussian, 0.0, None))
            k1 = mean + root
            k2 = mean - root

        # --- automatic resolution clamp -------------------------------------------
        # A fillet finer than the LOCAL triangles cannot be real geometry -- it's the
        # quadric fit blowing up at a sharp edge / sliver. Sampling theory says
        # curvature is only trustworthy where kappa*edge is small (Vasa et al. CGF
        # 2016: eps <= 1/(16*K*rho^2)) -- equivalently the radius 1/kappa must exceed
        # the local sample spacing (Amenta-Bern epsilon-sampling / local feature size).
        # So cap |kappa| per vertex so the radius can't be finer than
        # `min_radius_edges` x the LOCAL edge length. Kills the spurious spikes
        # automatically; every derived field inherits the cap.
        if clamp_to_resolution == "true" and len(F):
            ev = np.asarray(mesh.edges_unique)
            el = np.asarray(mesh.edges_unique_length, dtype=np.float64)
            loc = np.zeros(n, dtype=np.float64)
            cnt = np.zeros(n, dtype=np.float64)
            np.add.at(loc, ev[:, 0], el)
            np.add.at(loc, ev[:, 1], el)
            np.add.at(cnt, ev[:, 0], 1.0)
            np.add.at(cnt, ev[:, 1], 1.0)
            loc = loc / np.maximum(cnt, 1.0)                 # per-vertex local edge length
            loc[loc <= 0] = (float(np.mean(el)) if len(el) else 1.0)
            kcap = 1.0 / (max(1e-6, float(min_radius_edges)) * loc)   # max |kappa| per vertex
            k1 = np.clip(k1, -kcap, kcap)
            k2 = np.clip(k2, -kcap, kcap)

        # Recompute dependents from the (possibly clamped) principals so everything is
        # consistent (a no-op when nothing was clamped).
        mean = 0.5 * (k1 + k2)
        gaussian = k1 * k2

        # abs_max = MAGNITUDE of the strongest principal curvature (>= 0): ~0 on a
        # flat, ~1/r on a fillet of radius r -- the field to threshold for
        # flat-vs-curved, independent of concave/convex. `dominant` keeps the sign
        # (negative = concave / inner round, positive = convex / outer round).
        abs_max = np.maximum(np.abs(k1), np.abs(k2))
        dominant = np.where(np.abs(k1) >= np.abs(k2), k1, k2)
        shape_index = cls._shape_index(k1, k2)

        # "Dihedral-equivalent" angle: how many DEGREES the surface turns over one
        # mean edge length (theta = kappa * L). This puts curvature in the SAME units
        # as a per-edge dihedral angle, so you can threshold it the way you'd threshold
        # a crease. ~0 on flats; a fillet of radius r meshed at edge length L reads
        # ~degrees(L/r). (Unlike a raw dihedral it's defined inside a smooth fillet,
        # not just at sharp creases.)
        try:
            mean_edge_len = float(np.mean(mesh.edges_unique_length))
        except Exception:
            mean_edge_len = 1.0
        angle_deg = np.degrees(abs_max * mean_edge_len)

        fields = {
            "curvature_mean": mean,
            "curvature_gaussian": gaussian,
            "curvature_k1": k1,
            "curvature_k2": k2,
            "curvature_abs_max": abs_max,
            "curvature_dominant": dominant,
            "curvature_angle_deg": angle_deg,
            "curvature_shape_index": shape_index,
        }

        # Optional Laplacian smoothing of each scalar field. Row-normalized (umbrella)
        # neighbour averaging -- a CONVEX combination, so it's unconditionally stable at
        # any mesh resolution. (The previous explicit cotangent diffusion
        # `f += 0.5 * Minv @ L @ f` blew up to ~1e12: the Laplace-Beltrami spectral
        # radius scales like 1/edge^2, so a fixed step diverges on fine meshes, and
        # obtuse triangles add negative cotangent weights.)
        if smoothing_iterations > 0:
            ev = np.asarray(mesh.edges_unique)
            deg = np.zeros(n, dtype=np.float64)
            np.add.at(deg, ev[:, 0], 1.0)
            np.add.at(deg, ev[:, 1], 1.0)
            deg = np.maximum(deg, 1.0)
            lam = 0.5
            for name in list(fields.keys()):
                f = np.asarray(fields[name], dtype=np.float64).copy()
                for _ in range(int(smoothing_iterations)):
                    nb = np.zeros(n, dtype=np.float64)
                    np.add.at(nb, ev[:, 0], f[ev[:, 1]])
                    np.add.at(nb, ev[:, 1], f[ev[:, 0]])
                    f = (1.0 - lam) * f + lam * (nb / deg)
                fields[name] = f

        # Optional percentile clamp so slivers don't dominate the range.
        if clamp_percentile and clamp_percentile > 0:
            p = float(clamp_percentile)
            for name in list(fields.keys()):
                lo, hi = np.percentile(fields[name], [p, 100.0 - p])
                fields[name] = np.clip(fields[name], lo, hi)

        # Attach as vertex scalar fields + a canonical 'curvature' alias.
        for name, f in fields.items():
            mesh.vertex_attributes[name] = np.ascontiguousarray(f, dtype=np.float32)
        primary_map = {
            "mean": "curvature_mean", "gaussian": "curvature_gaussian",
            "max_principal": "curvature_k1", "min_principal": "curvature_k2",
            "abs_max_principal": "curvature_abs_max", "dominant_signed": "curvature_dominant",
            "angle_deg": "curvature_angle_deg", "shape_index": "curvature_shape_index",
        }
        mesh.vertex_attributes["curvature"] = mesh.vertex_attributes[primary_map[primary_output]].copy()

        # Also attach per-FACE versions (mean over each face's 3 vertices) so face-field
        # consumers like Preview Mesh Boundaries can threshold edges by curvature the
        # same way they threshold dihedral angle (face_normals + 'angle').
        Fidx = np.asarray(mesh.faces)
        for name, f in fields.items():
            mesh.face_attributes[name] = np.ascontiguousarray(
                np.asarray(f, dtype=np.float32)[Fidx].mean(axis=1), dtype=np.float32)
        mesh.face_attributes["curvature"] = mesh.face_attributes[primary_map[primary_output]].copy()

        mesh.metadata = (mesh.metadata.copy() if mesh.metadata else {})
        mesh.metadata["curvature"] = {
            "method": method, "radius": int(radius), "primary": primary_output,
            "smoothing_iterations": int(smoothing_iterations),
            "clamp_percentile": float(clamp_percentile),
        }

        lines = [
            f"Compute Curvature ({method})",
            f"verts={n:,} faces={len(F):,} | radius={radius} smooth={smoothing_iterations} clamp={clamp_percentile}%",
            f"primary field 'curvature' = {primary_output}",
            "",
        ]
        for name in ["curvature_mean", "curvature_gaussian", "curvature_k1",
                     "curvature_k2", "curvature_shape_index"]:
            f = fields[name]
            lines.append(f"{name:22s} min={float(np.min(f)):+.4g}  max={float(np.max(f)):+.4g}  mean={float(np.mean(f)):+.4g}")
        info = "\n".join(lines)
        log.info("Compute Curvature done: %s", " | ".join(lines[1:3]))

        return io.NodeOutput(mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackComputeCurvature": ComputeCurvatureNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackComputeCurvature": "Compute Curvature"}
