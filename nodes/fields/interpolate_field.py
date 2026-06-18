# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors
"""Interpolate Field — transfer per-vertex / per-face attributes from one mesh to another.

Method consensus from two geometry-processing surveys:
  - barycentric (DEFAULT): closest-point on a source triangle + barycentric interpolation
    of the source per-vertex field. Second-order; exact when the meshes coincide. The right
    default for resampling a continuous field after remeshing the SAME object.
  - nearest: copy the nearest source value. The ONLY correct path for integer / label fields
    (cad_face_id, segmentation, materials) — barycentric/idw/rbf would average ids into
    nonexistent labels. Integer/boolean fields are therefore ALWAYS routed to nearest.
  - idw: inverse-distance weighting over k nearest source points (smoothing / point-cloud src).
  - rbf: scipy RBFInterpolator (smooth / sparse source).

Per-face fields transfer by nearest source face (constant per face). A max_distance guard
flags target points whose closest source point is farther than the threshold.

Heavy machinery deliberately skipped (overkill when source/target share the same space):
conservative L2/Galerkin projection (needs a supermesh), functional maps (no registration
problem here). Add later only if a use case demands integral conservation.
"""
import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("GeometryPack")


# ── core transfer helpers (no ComfyUI deps — unit-testable) ──────────────────

def _is_discrete(arr):
    """Integer / boolean fields must never be linearly blended."""
    return np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_


def _closest_point(mesh, pts):
    """Closest point on the mesh surface -> (closest, distance, triangle_id).
    Fast rtree-backed path, with a brute-force fallback if rtree is unavailable."""
    try:
        return trimesh_module.proximity.closest_point(mesh, pts)
    except Exception:
        return trimesh_module.proximity.closest_point_naive(mesh, pts)


def _as2d(arr):
    """(n, ...) -> (n, C) plus the trailing shape, so scalar/vector/matrix fields
    all flow through the same code."""
    a = np.asarray(arr)
    trailing = a.shape[1:]
    return a.reshape(a.shape[0], -1), trailing


def transfer_vertex_field(src, tgt, field, method, max_distance, fill_value,
                          idw_k, idw_power):
    """Transfer a per-source-vertex field to the target vertices.

    Returns (values[n_tgt, ...], n_out_of_range). Discrete fields force 'nearest'.
    """
    P = np.asarray(tgt.vertices, dtype=np.float64)
    discrete = _is_discrete(field)
    F, trailing = _as2d(field)
    eff_method = "nearest" if discrete else method

    if eff_method in ("barycentric", "nearest"):
        # surface-aware: closest point on a source triangle
        closest, dist, tri_id = _closest_point(src, P)
        tri_v = src.faces[tri_id]                                  # (n,3) source vert ids
        bary = trimesh_module.triangles.points_to_barycentric(
            src.triangles[tri_id], closest)                        # (n,3)
        if eff_method == "nearest":
            dom = tri_v[np.arange(len(tri_id)), np.argmax(bary, axis=1)]
            out = F[dom]
        else:
            f = F[tri_v]                                           # (n,3,C)
            out = (bary[:, :, None] * f).sum(axis=1)               # (n,C)
    else:
        # meshless on source vertices
        Vsrc = np.asarray(src.vertices, dtype=np.float64)
        from scipy.spatial import cKDTree
        tree = cKDTree(Vsrc)
        if eff_method == "idw":
            k = int(max(1, min(idw_k, len(Vsrc))))
            d, idx = tree.query(P, k=k)
            d = np.atleast_2d(d).reshape(len(P), k)
            idx = np.atleast_2d(idx).reshape(len(P), k)
            w = 1.0 / (np.power(d, idw_power) + 1e-12)             # (n,k)
            w /= w.sum(axis=1, keepdims=True)
            out = (w[:, :, None] * F[idx].astype(np.float64)).sum(axis=1)
            dist = d[:, 0]
        elif eff_method == "rbf":
            from scipy.interpolate import RBFInterpolator
            nbr = int(min(32, len(Vsrc)))
            rbf = RBFInterpolator(Vsrc, F.astype(np.float64),
                                  neighbors=nbr, kernel="thin_plate_spline")
            out = rbf(P)
            dist, _ = tree.query(P, k=1)
        else:
            raise ValueError(f"unknown method: {method}")

    n_oor = 0
    if max_distance and max_distance > 0:
        oor = dist > float(max_distance)
        n_oor = int(oor.sum())
        if n_oor:
            out = out.astype(np.float64, copy=True) if not discrete else out.copy()
            out[oor] = fill_value
    out = out.astype(field.dtype, copy=False) if discrete else out
    return out.reshape((len(P),) + trailing), n_oor


def transfer_face_field(src, tgt, field, max_distance, fill_value):
    """Transfer a per-source-face field to the target faces by nearest source face
    (constant-per-face is correct for both labels and piecewise face data)."""
    centroids = np.asarray(tgt.triangles_center, dtype=np.float64)
    _, dist, tri_id = _closest_point(src, centroids)
    field = np.asarray(field)
    out = field[tri_id]
    n_oor = 0
    if max_distance and max_distance > 0:
        oor = dist > float(max_distance)
        n_oor = int(oor.sum())
        if n_oor:
            if _is_discrete(field):
                out = out.copy()
            else:
                out = out.astype(np.float64, copy=True)
            out[oor] = fill_value
    return out, n_oor


def _select_fields(src, field_name):
    """Return [(name, kind, array)] to transfer. kind in {'vertex','face'}.
    Blank field_name = every vertex + face attribute."""
    vattr = dict(getattr(src, "vertex_attributes", {}) or {})
    fattr = dict(getattr(src, "face_attributes", {}) or {})
    fn = (field_name or "").strip()
    out = []
    if fn:
        if fn.startswith("face.") and fn[5:] in fattr:
            out.append((fn[5:], "face", fattr[fn[5:]]))
        elif fn in vattr:
            out.append((fn, "vertex", vattr[fn]))
        elif fn in fattr:
            out.append((fn, "face", fattr[fn]))
    else:
        out += [(n, "vertex", v) for n, v in vattr.items()]
        out += [(n, "face", v) for n, v in fattr.items()]
    return out


# ── node ─────────────────────────────────────────────────────────────────────

class InterpolateFieldNode(io.ComfyNode):
    """Transfer scalar/vector fields from one mesh onto another (e.g. after remeshing)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackInterpolateField",
            display_name="Interpolate Field",
            category="geompack/fields",
            inputs=[
                io.Custom("TRIMESH").Input("field_providing_mesh",
                    tooltip="Source mesh carrying the field(s) to transfer (in its vertex_attributes / face_attributes)."),
                io.Custom("TRIMESH").Input("field_target_mesh",
                    tooltip="Target mesh to receive the interpolated field(s). Its geometry is unchanged; only attributes are added."),
                io.String.Input("field_name", default="",
                    tooltip="Single field to transfer. Leave BLANK to transfer ALL vertex + face fields. Address a face field explicitly as 'face.<name>' (e.g. 'face.cad_face_id')."),
                io.Combo.Input("method", options=["barycentric", "nearest", "idw", "rbf"], default="barycentric",
                    tooltip="barycentric: closest-point + barycentric blend (default; accurate for continuous fields, exact when meshes coincide). nearest: copy nearest source value (used automatically for integer/label fields). idw: inverse-distance weighting (smoothing / point clouds). rbf: thin-plate RBF (smooth / sparse source). NOTE: integer & boolean fields are ALWAYS transferred by nearest regardless of this setting — blending labels is meaningless."),
                io.Float.Input("max_distance", default=0.0, min=0.0, max=1e9, step=0.001,
                    tooltip="World-unit cap on the source->target projection distance. Target points whose closest source point is farther than this are out-of-range and set to fill_value (prevents grabbing values across holes / off the surface). 0 = no limit."),
                io.Float.Input("fill_value", default=0.0, min=-1e12, max=1e12, step=0.1, advanced=True,
                    tooltip="Value written where a target point exceeds max_distance (no nearby source)."),
                io.Int.Input("idw_k", default=8, min=1, max=128, step=1, advanced=True,
                    tooltip="IDW only: number of nearest source points averaged."),
                io.Float.Input("idw_power", default=2.0, min=0.1, max=8.0, step=0.1, advanced=True,
                    tooltip="IDW only: inverse-distance exponent (higher = more local)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, field_providing_mesh, field_target_mesh, field_name="",
                method="barycentric", max_distance=0.0, fill_value=0.0,
                idw_k=8, idw_power=2.0):
        src = field_providing_mesh
        tgt = field_target_mesh.copy()

        fields = _select_fields(src, field_name)
        if not fields:
            avail_v = list((getattr(src, "vertex_attributes", {}) or {}).keys())
            avail_f = list((getattr(src, "face_attributes", {}) or {}).keys())
            msg = (f"No matching field. field_name={field_name!r}. "
                   f"Available vertex fields: {avail_v}; face fields: {avail_f}.")
            log.warning("[InterpolateField] %s", msg)
            return io.NodeOutput(tgt, msg, ui={"text": [msg]})

        lines = [f"Interpolate Field: src {len(src.vertices)}v/{len(src.faces)}f "
                 f"-> tgt {len(tgt.vertices)}v/{len(tgt.faces)}f | method={method}"]
        for name, kind, arr in fields:
            arr = np.asarray(arr)
            forced = _is_discrete(arr) and method != "nearest"
            try:
                if kind == "vertex":
                    out, n_oor = transfer_vertex_field(
                        src, tgt, arr, method, max_distance, fill_value, idw_k, idw_power)
                    tgt.vertex_attributes[name] = out
                else:
                    out, n_oor = transfer_face_field(src, tgt, arr, max_distance, fill_value)
                    tgt.face_attributes[name] = out
                lines.append(
                    f"  [{kind}] {name}: dtype={arr.dtype} shape{arr.shape[1:] or '(scalar)'}"
                    f"{' -> nearest (discrete)' if forced else ''}"
                    f"{f' | {n_oor} out-of-range -> fill' if n_oor else ''}")
            except Exception as e:
                lines.append(f"  [{kind}] {name}: FAILED ({type(e).__name__}: {e})")
                log.exception("[InterpolateField] transfer failed for %s", name)

        info = "\n".join(lines)
        log.info("[InterpolateField]\n%s", info)
        return io.NodeOutput(tgt, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackInterpolateField": InterpolateFieldNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackInterpolateField": "Interpolate Field"}
