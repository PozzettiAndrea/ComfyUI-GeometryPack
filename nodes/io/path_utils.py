# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Shared read-path resolution against the ComfyUI directory layout.

Users hand nodes paths in many shapes: real absolute paths, paths relative to
the ComfyUI base (``input/3d/mesh.ply``, ``output/result.ply``), the same with
a leading slash (``/input/3d/mesh.ply`` -- rooted-looking but meant relative to
the ComfyUI base), or bare filenames expected in ``input/3d``. The leading-slash
form defeats a plain ``os.path.join`` (Python discards the base when the second
argument is absolute), so it must be stripped before joining.

``folder_paths`` is imported lazily inside the functions rather than at module
scope: on ComfyUI Desktop the worker rewrites ``folder_paths.base_path`` and the
input/output directories after import, so values captured at import time can go
stale.
"""

import os


class AmbiguousPathError(ValueError):
    """A relative path matched files in more than one ComfyUI folder."""


def _candidate_paths(path):
    """Yield candidate locations for ``path``, most specific first."""
    yield path

    stripped = path.lstrip("/\\")
    base = None
    input_dir = None
    output_dir = None
    try:
        import folder_paths
        base = getattr(folder_paths, "base_path", None)
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()
    except (ImportError, AttributeError):
        base = os.environ.get("COMFYUI_BASE")

    if base:
        yield os.path.join(base, stripped)
    if input_dir:
        yield os.path.join(input_dir, "3d", stripped)
        yield os.path.join(input_dir, stripped)
    if output_dir:
        yield os.path.join(output_dir, stripped)


def resolve_read_path(path):
    """Resolve a user-supplied read path against the ComfyUI layout.

    Tries the path as given, then (with any leading slash stripped) relative
    to the ComfyUI base, ``input/3d``, ``input`` and ``output``. Returns the
    single existing candidate, or None if nothing matches.

    Raises AmbiguousPathError when the path matches distinct files in more
    than one folder -- picking one silently would load the wrong file half
    the time; the user must disambiguate (e.g. ``input/3d/mesh.ply`` vs
    ``output/mesh.ply``).
    """
    if not path or not str(path).strip():
        return None
    path = str(path).strip()

    candidates = list(_candidate_paths(path))
    # The as-given form is explicit (a real absolute or cwd-relative path the
    # user spelled out) -- never treated as ambiguous.
    if os.path.exists(candidates[0]):
        return candidates[0]

    hits = {}
    for candidate in candidates[1:]:
        if os.path.exists(candidate):
            hits.setdefault(os.path.realpath(candidate), candidate)
    matches = list(hits.values())
    if len(matches) > 1:
        base = None
        try:
            import folder_paths
            base = getattr(folder_paths, "base_path", None)
        except (ImportError, AttributeError):
            base = os.environ.get("COMFYUI_BASE")
        def _suggest(p):
            if base:
                try:
                    if os.path.commonpath([os.path.abspath(p), os.path.abspath(base)]) \
                            == os.path.abspath(base):
                        return os.path.relpath(p, base)
                except ValueError:
                    pass
            return p
        options = " or ".join(f"'{_suggest(m)}'" for m in matches)
        raise AmbiguousPathError(
            f"Ambiguous path '{path}': it exists in multiple locations. "
            f"Please specify {options}.")
    return matches[0] if matches else None


def searched_locations(path):
    """The candidate list ``resolve_read_path`` would try, for error messages."""
    if not path or not str(path).strip():
        return []
    return list(_candidate_paths(str(path).strip()))
