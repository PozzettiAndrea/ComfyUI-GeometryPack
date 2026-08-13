"""GeometryPack wire type: trimesh.Trimesh (comfy-env [types], ADR-0015).

Loaded by file path on BOTH sides of the process boundary, so the top level
stays self-contained -- stdlib/comfy_env imports only; trimesh is imported
inside the functions. Geometry (vertices/faces) decomposes into shared-memory
arrays instead of the 3-copy pickle rung; visuals + metadata ride best-effort
via the transport's fallback (the visual's mesh back-reference is dropped, so
it cannot drag the whole geometry back into a pickle).

A side without trimesh registers deserialize=None: comfy-env then holds the
value as a materialized OpaquePayload receipt and re-emits it when forwarding,
so a bare ComfyUI host never needs trimesh. The tag is the TYPE IDENTITY, so
independent packs that also declare trimesh.Trimesh interoperate.
"""

try:  # parent: comfy_env installed. worker: module copied in flat, no package.
    from comfy_env.isolation.workers._ipc_shared import register_serializer
except ImportError:
    from _ipc_shared import register_serializer


def _opt(fn):
    """fn(), or None if this env can't run it (missing deps, unpicklable, ...)."""
    try:
        return fn()
    except Exception:
        return None


def _put(payload, key, value, recurse):
    """Best-effort payload[key] = recurse(value); skip None / unserializable."""
    if value is not None and (v := _opt(lambda: recurse(value))) is not None:
        payload[key] = v


def _serialize_trimesh(mesh, recurse):
    # Pass mesh.vertices/mesh.faces DIRECTLY -- never np.asarray() them. The
    # transport keys its visited-map by id(); a temporary created here can be
    # GC'd mid-walk and its id reused by a later array, which then receives the
    # WRONG frame (seen: faces deserialized as vertices). Keep inputs mesh-lived.
    payload = {"vertices": recurse(mesh.vertices), "faces": recurse(mesh.faces)}

    # Decompose visuals by kind; never copy them whole -- a full copy pulls in
    # optional deps (PIL) and the visual's mesh back-reference re-serializes the
    # geometry. (Vertex colors TODO: their accessor makes temporaries, unsafe
    # under the id-reuse hazard above.)
    visual = getattr(mesh, "visual", None)
    if type(visual).__name__ == "TextureVisuals" and (uv := visual.uv) is not None and len(uv):
        payload["uv"] = recurse(uv)
        _put(payload, "material", visual.material, recurse)

    if getattr(mesh, "metadata", None):
        _put(payload, "metadata", dict(mesh.metadata), recurse)
    return payload


def _deserialize_trimesh(payload, recurse):
    import trimesh
    mesh = trimesh.Trimesh(recurse(payload["vertices"]), recurse(payload["faces"]),
                           process=False)  # exact round-trip; no merge/validate

    if "uv" in payload:
        from trimesh.visual import TextureVisuals
        material = _opt(lambda: recurse(payload["material"])) if "material" in payload else None
        visual = _opt(lambda: TextureVisuals(uv=recurse(payload["uv"]), material=material))
        if visual is not None:
            mesh.visual = visual

    if "metadata" in payload and (meta := _opt(lambda: recurse(payload["metadata"]))):
        mesh.metadata.update(meta)
    return mesh


# Deserializer only where trimesh exists; elsewhere None -> comfy-env keeps the
# value as an OpaquePayload receipt (docstring). Tag is the base type -- MRO
# matching carries Trimesh subclasses along.
try:
    import trimesh
except ImportError:
    trimesh = None

register_serializer(
    "Trimesh", _serialize_trimesh,
    _deserialize_trimesh if trimesh else None,
    tag="trimesh.Trimesh",
)
