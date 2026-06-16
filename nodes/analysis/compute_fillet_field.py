# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Compute a noise-robust "fillet field": where a surface is genuinely, smoothly
curved (rolling-ball fillets, cylinders, blends) as opposed to flat-but-noisy or
sharp-edged.

The discriminator a fillet needs is NOT raw per-edge curvature -- noise lives at
that scale too, so a noisy flat and a noisy fillet look identical edge-to-edge.
The fix is SCALE: estimate curvature with a least-squares quadric fit over a
k-ring neighborhood (libigl `principal_curvature`). The fit averages out
high-frequency noise (random -> cancels) while the fillet's steady, coherent bend
survives (consistent -> accumulates). So:

  * noisy flat   -> fitted curvature ~ 0          -> fillet_field ~ 0   (let it smooth)
  * noisy fillet -> fitted curvature ~ 1/r > 0    -> fillet_field high  (protect)
  * sharp edge   -> handled by crease angle elsewhere

The tightest principal curvature |k|max is the rolling-ball curvature: its
reciprocal `fillet_radius = 1/|k|max` IS the local rolling-ball radius (in model
units). Multiplying by the bounding-box diagonal makes it dimensionless /
scale-free (the "global" threshold value): `fillet_field = |k|max * bbox_diag`,
so the same part at 1 mm or 1 km reads identically.

Fields attached (as vertex_attributes, viewable / thresholdable downstream):
  fillet_field   = |k|max * bbox_diag        (dimensionless, primary; threshold this)
  fillet_radius  = 1/|k|max                   (model units; the local rolling-ball radius)
  curvedness     = sqrt((k1^2+k2^2)/2)        (raw magnitude, 1/length)
  is_fillet      = 1 where fillet_radius <= fillet_radius_frac * bbox_diag else 0
                   (the ready-to-use "protect these faces" mask)
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class ComputeFilletFieldNode(io.ComfyNode):
    """Noise-robust detector of genuinely-curved (fillet/blend) regions."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackComputeFilletField",
            display_name="Compute Fillet Field",
            category="geompack/analysis",
            description=(
                "Find where a surface is genuinely smoothly curved (rolling-ball fillets, "
                "blends, cylinders) vs flat-but-noisy -- robust to noise ON the fillet.\n"
                "\n"
                "Uses a least-squares quadric fit (libigl principal_curvature) over a k-ring "
                "set by 'radius': the fit averages out high-frequency noise while the fillet's "
                "steady bend survives. The tightest principal curvature is the rolling-ball "
                "curvature, so fillet_radius = 1/|k|max is the local blend radius.\n"
                "\n"
                "Outputs scalar vertex fields: fillet_field (dimensionless = |k|max*bbox_diag, "
                "the scale-free value to threshold), fillet_radius (model units), curvedness, "
                "and is_fillet (a 0/1 protect mask for curves up to fillet_radius_frac of the "
                "object). Feed the mask/field to Preview Mesh Boundaries to eyeball it, or use "
                "it to exempt fillet regions from guided-normal sharpening.\n"
                "\n"
                "radius: bigger = more noise-robust but blurs tighter fillets (needs noise_scale "
                "< radius < fillet_width to work). fillet_radius_frac: protect curves whose "
                "rolling-ball radius is up to this fraction of the bounding box."
            ),
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("radius", default=6, min=1, max=16, step=1, tooltip=(
                    "Quadric-fit neighborhood (k-ring, in avg edge lengths). This is the SCALE "
                    "knob: it must be larger than the noise wavelength (so noise averages out) "
                    "but smaller than the fillet width (so it isn't blurred across). ~6 default; "
                    "raise for noisy scans, lower for crisp CAD with tight fillets.")),
                io.Float.Input("fillet_radius_frac", default=0.1, min=0.0001, max=2.0, step=0.01, tooltip=(
                    "is_fillet mask cutoff: flag a vertex as a fillet/curve to protect when its "
                    "local rolling-ball radius is <= this fraction of the bounding-box diagonal. "
                    "0.1 = protect anything curving tighter than 10% of the object. Scale-free.")),
                io.Int.Input("smoothing_iterations", default=2, min=0, max=50, step=1, tooltip=(
                    "Extra cotangent-Laplacian diffusion of the field after fitting. A couple of "
                    "passes clean up residual speckle on the mask without re-fitting. 0 = off.")),
                io.Float.Input("clamp_percentile", default=1.0, min=0.0, max=20.0, step=0.5, tooltip=(
                    "Clip the curvature fields to their [p, 100-p] percentile so sliver triangles "
                    "don't blow out the range / the bbox-normalization. 0 = off.")),
                io.Combo.Input("preclean", options=["true", "false"], default="true", tooltip=(
                    "Merge duplicate vertices, drop degenerate faces, fix normals before fitting "
                    "(recommended -- degenerate triangles are the main quadric-fit failure mode).")),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh_with_fillet_field"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, radius=6, fillet_radius_frac=0.1,
                smoothing_iterations=2, clamp_percentile=1.0, preclean="true"):
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

        # Scale reference: bounding-box diagonal makes the field dimensionless.
        bbox = V.max(axis=0) - V.min(axis=0)
        diag = float(np.linalg.norm(bbox))
        if diag <= 0:
            diag = 1.0
        log.info("Compute Fillet Field: %d verts %d faces, radius=%d, bbox_diag=%.6g",
                 n, len(F), radius, diag)

        # Noise-robust principal curvatures via least-squares quadric fit.
        out = igl.principal_curvature(V, F, int(max(1, radius)))
        k1 = np.asarray(out[2], dtype=np.float64)   # max principal
        k2 = np.asarray(out[3], dtype=np.float64)   # min principal

        curvedness = np.sqrt(0.5 * (k1 * k1 + k2 * k2))
        kmax_abs = np.maximum(np.abs(k1), np.abs(k2))   # rolling-ball curvature

        fields = {
            "curvedness": curvedness,
            "fillet_field": kmax_abs * diag,            # dimensionless, primary
        }

        # Optional Laplacian diffusion (denoise the field, not the geometry).
        if smoothing_iterations > 0:
            M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
            Mdiag = np.asarray(M.diagonal(), dtype=np.float64)
            Mdiag[Mdiag <= 0] = 1e-12
            Minv = sp.diags(1.0 / Mdiag)
            L = igl.cotmatrix(V, F)
            lam = 0.5
            for name in list(fields.keys()):
                f = np.asarray(fields[name], dtype=np.float64).copy()
                for _ in range(int(smoothing_iterations)):
                    f = f + lam * np.asarray(Minv @ (L @ f))
                fields[name] = f

        # Percentile clamp so slivers don't dominate.
        if clamp_percentile and clamp_percentile > 0:
            p = float(clamp_percentile)
            for name in list(fields.keys()):
                lo, hi = np.percentile(fields[name], [p, 100.0 - p])
                fields[name] = np.clip(fields[name], lo, hi)

        # Derived radius + protect mask, from the (possibly smoothed) fillet_field.
        ff = np.asarray(fields["fillet_field"], dtype=np.float64)
        fillet_radius = diag / np.maximum(ff, 1e-12)     # = 1/|k|max in model units
        is_fillet = (fillet_radius <= float(fillet_radius_frac) * diag).astype(np.float32)
        fields["fillet_radius"] = fillet_radius
        fields["is_fillet"] = is_fillet

        for name, f in fields.items():
            mesh.vertex_attributes[name] = np.ascontiguousarray(f, dtype=np.float32)

        mesh.metadata = (mesh.metadata.copy() if mesh.metadata else {})
        mesh.metadata["fillet_field"] = {
            "radius": int(radius), "bbox_diag": diag,
            "fillet_radius_frac": float(fillet_radius_frac),
            "smoothing_iterations": int(smoothing_iterations),
        }

        n_fillet = int(is_fillet.sum())
        pct = 100.0 * n_fillet / max(1, n)
        cutoff_r = float(fillet_radius_frac) * diag
        lines = [
            "Compute Fillet Field (quadric fit, radius=%d)" % int(radius),
            f"verts={n:,} faces={len(F):,} | bbox_diag={diag:.6g}",
            f"protect cutoff: rolling-ball radius <= {cutoff_r:.6g} "
            f"({fillet_radius_frac:g} x bbox)  ->  fillet_field >= {1.0/float(fillet_radius_frac):.3g}",
            f"flagged as fillet/curve: {n_fillet:,} verts ({pct:.1f}%)",
            "",
            f"fillet_field   min={float(ff.min()):.4g} max={float(ff.max()):.4g} mean={float(ff.mean()):.4g}",
            f"fillet_radius  min={float(fillet_radius.min()):.4g} max={float(fillet_radius.max()):.4g} (model units)",
            f"curvedness     min={float(fields['curvedness'].min()):.4g} max={float(fields['curvedness'].max()):.4g} (1/length)",
        ]
        info = "\n".join(lines)
        log.info("Compute Fillet Field done: %s", " | ".join(lines[1:4]))

        return io.NodeOutput(mesh, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackComputeFilletField": ComputeFilletFieldNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackComputeFilletField": "Compute Fillet Field"}
