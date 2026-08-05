# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Shared helpers + driver for the Interpolate Field backends.

Hard rules enforced here for EVERY backend:
- integer / boolean fields are always transferred by (surface-aware) nearest --
  linearly blending labels produces nonexistent ids;
- per-face fields transfer by nearest source face (constant per face);
- the max_distance / fill_value out-of-range guard applies uniformly.

A backend contributes only its continuous per-vertex interpolator, via the
`vertex_interp(src, tgt_points, F)` callable handed to `run_interpolation` --
it returns (values (n,C) float, dist (n,) distances for the range guard).
"""

import logging

import numpy as np
import trimesh as trimesh_module

log = logging.getLogger("GeometryPack")


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


def parse_field_names(interpolate_all_fields, field_names):
    """UI inputs -> None (= all fields) or a list of requested names.
    Accepts comma-separated names, optionally quoted: pressure, "length", 'face.part_id'."""
    if interpolate_all_fields in (True, "true", "True", 1):
        return None
    names = []
    for tok in (field_names or "").split(","):
        tok = tok.strip().strip('"').strip("'").strip()
        if tok:
            names.append(tok)
    return names


def _select_fields(src, field_names):
    """Return [(name, kind, array)] to transfer. kind in {'vertex','face'}.
    field_names: None = every vertex + face attribute; else list of names
    (a 'face.' prefix addresses a face field explicitly)."""
    vattr = dict(getattr(src, "vertex_attributes", {}) or {})
    fattr = dict(getattr(src, "face_attributes", {}) or {})
    out = []
    if field_names is None:
        out += [(n, "vertex", v) for n, v in vattr.items()]
        out += [(n, "face", v) for n, v in fattr.items()]
        return out
    for fn in field_names:
        if fn.startswith("face.") and fn[5:] in fattr:
            out.append((fn[5:], "face", fattr[fn[5:]]))
        elif fn in vattr:
            out.append((fn, "vertex", vattr[fn]))
        elif fn in fattr:
            out.append((fn, "face", fattr[fn]))
        else:
            out.append((fn, "missing", None))
    return out


# -- built-in per-vertex interpolators (surface-aware pair) --------------------

def nearest_vertex_interp(src, P, F):
    """Surface-aware nearest: closest point on a source triangle -> value of the
    dominant-barycentric corner vertex."""
    closest, dist, tri_id = _closest_point(src, P)
    tri_v = src.faces[tri_id]
    bary = trimesh_module.triangles.points_to_barycentric(src.triangles[tri_id], closest)
    dom = tri_v[np.arange(len(tri_id)), np.argmax(bary, axis=1)]
    return F[dom], dist


def barycentric_vertex_interp(src, P, F):
    """Closest point on a source triangle + barycentric blend of its 3 corners."""
    closest, dist, tri_id = _closest_point(src, P)
    tri_v = src.faces[tri_id]
    bary = trimesh_module.triangles.points_to_barycentric(src.triangles[tri_id], closest)
    f = F[tri_v]                                           # (n,3,C)
    return (bary[:, :, None] * f).sum(axis=1), dist


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


def transfer_vertex_field(src, tgt, field, vertex_interp, max_distance, fill_value):
    """Transfer a per-source-vertex field to the target vertices via the backend's
    `vertex_interp` (discrete fields are forced to the nearest interpolator).
    Returns (values[n_tgt, ...], n_out_of_range, forced_nearest)."""
    P = np.asarray(tgt.vertices, dtype=np.float64)
    discrete = _is_discrete(np.asarray(field))
    F, trailing = _as2d(field)

    interp = nearest_vertex_interp if discrete else vertex_interp
    out, dist = interp(src, P, F)

    n_oor = 0
    if max_distance and max_distance > 0:
        oor = dist > float(max_distance)
        n_oor = int(oor.sum())
        if n_oor:
            out = out.astype(np.float64, copy=True) if not discrete else out.copy()
            out[oor] = fill_value
    out = out.astype(np.asarray(field).dtype, copy=False) if discrete else out
    return out.reshape((len(P),) + trailing), n_oor, discrete


def run_interpolation(src, tgt_in, field_names, method_label, vertex_interp,
                      max_distance, fill_value):
    """Shared driver: select fields, transfer each (vertex via the backend's
    interpolator, face via nearest-face), build info. Returns (tgt, info).
    field_names: None = all fields, else list of names (see parse_field_names)."""
    tgt = tgt_in.copy()

    fields = _select_fields(src, field_names)
    if not fields or all(kind == "missing" for _, kind, _ in fields):
        avail_v = list((getattr(src, "vertex_attributes", {}) or {}).keys())
        avail_f = list((getattr(src, "face_attributes", {}) or {}).keys())
        msg = (f"No matching field. requested={field_names!r}. "
               f"Available vertex fields: {avail_v}; face fields: {avail_f}.")
        log.warning("[InterpolateField] %s", msg)
        return tgt, msg

    lines = [f"Interpolate Field: src {len(src.vertices)}v/{len(src.faces)}f "
             f"-> tgt {len(tgt.vertices)}v/{len(tgt.faces)}f | method={method_label}"]
    for name, kind, arr in fields:
        if kind == "missing":
            avail_v = list((getattr(src, "vertex_attributes", {}) or {}).keys())
            avail_f = list((getattr(src, "face_attributes", {}) or {}).keys())
            lines.append(f"  {name}: NOT FOUND on source "
                         f"(vertex fields: {avail_v}; face fields: {avail_f})")
            continue
        arr = np.asarray(arr)
        try:
            if kind == "vertex":
                out, n_oor, forced = transfer_vertex_field(
                    src, tgt, arr, vertex_interp, max_distance, fill_value)
                tgt.vertex_attributes[name] = out
            else:
                out, n_oor = transfer_face_field(src, tgt, arr, max_distance, fill_value)
                tgt.face_attributes[name] = out
                forced = False
            lines.append(
                f"  [{kind}] {name}: dtype={arr.dtype} shape{arr.shape[1:] or '(scalar)'}"
                f"{' -> nearest (discrete)' if forced and method_label != 'nearest' else ''}"
                f"{f' | {n_oor} out-of-range -> fill' if n_oor else ''}")
        except Exception as e:
            lines.append(f"  [{kind}] {name}: FAILED ({type(e).__name__}: {e})")
            log.exception("[InterpolateField] transfer failed for %s", name)

    info = "\n".join(lines)
    log.info("[InterpolateField]\n%s", info)
    return tgt, info
