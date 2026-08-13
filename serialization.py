"""GeometryPack wire types (comfy-env [types] declaration, ADR-0015).

Declared in comfy-env-root.toml under [types]; loaded by file path on
BOTH sides of the process boundary (parent + workers), so keep the top
level self-contained: stdlib/comfy_env imports only, trimesh imported
inside the functions and probed once for conditional registration.

Registers trimesh.Trimesh -- the type behind all io.Custom("TRIMESH")
sockets in this pack. Geometry (vertices/faces) decomposes into
shared-memory arrays instead of riding the 3-copy pickle rung (which
also needs PIL at pickle time for textured meshes); the small remainder
(visual, metadata) travels via the transport's fallback, with the
visual's mesh BACK-REFERENCE never serialized so it cannot drag the
whole geometry into a pickle again.

A side without trimesh (the bare ComfyUI host env) registers
deserialize=None: comfy-env holds such values as MATERIALIZED
OpaquePayload receipts -- receiver-owned, safe across worker restarts
(comfy-env >= 0.4.15) -- and re-emits fresh frames when forwarding.
If some other pack installs trimesh into the host, the same file
registers the real deserializer there and host-side consumers get
actual Trimesh objects. No configuration either way.

Tag is the TYPE IDENTITY ("trimesh.Trimesh", ADR-0015 convention), so
independent packs that also declare trimesh.Trimesh interoperate: each
side rebuilds with its own registered functions.
"""

try:  # parent process (comfy_env installed)
    from comfy_env.isolation.workers._ipc_shared import register_serializer
except ImportError:  # worker process (copied module, no comfy_env package)
    from _ipc_shared import register_serializer


def _serialize_trimesh(mesh, recurse):
    # NOTE: pass mesh.vertices / mesh.faces DIRECTLY (they are long-lived
    # ndarray subclasses owned by the mesh). Do NOT wrap in np.asarray():
    # the transport's visited-map is keyed by id(), and a temporary created
    # here can be garbage-collected mid-walk, letting a later array reuse
    # its id and receive the WRONG frame (observed: faces deserialized as
    # vertices). Known comfy-env transport hazard; keep inputs long-lived.
    payload = {
        # Bulk geometry as arrays -> shared-memory path via recurse.
        "vertices": recurse(mesh.vertices),
        "faces": recurse(mesh.faces),
    }

    # Visuals are decomposed by kind rather than copied/pickled whole:
    # copying drags in optional deps (PIL via material deepcopy) and the
    # visual object holds a mesh back-reference that would re-serialize
    # the entire geometry. v1 preserves UVs + texture material; vertex
    # colors are TODO (their accessor synthesizes temporaries -- unsafe
    # under the id-reuse hazard above).
    visual = getattr(mesh, "visual", None)
    if type(visual).__name__ == "TextureVisuals":
        uv = getattr(visual, "uv", None)  # stored array, mesh-lifetime
        if uv is not None and len(uv):
            payload["uv"] = recurse(uv)
        material = getattr(visual, "material", None)
        if material is not None:
            try:
                payload["material"] = recurse(material)
            except Exception:
                pass  # material needs deps this env lacks (the transport
                # raises loudly on unpicklable values); uv still travels

    metadata = getattr(mesh, "metadata", None)
    if metadata:
        try:
            payload["metadata"] = recurse(dict(metadata))
        except Exception:
            pass

    return payload


def _deserialize_trimesh(payload, recurse):
    import trimesh
    mesh = trimesh.Trimesh(
        vertices=recurse(payload["vertices"]),
        faces=recurse(payload["faces"]),
        process=False,  # exact geometry round-trip; no merging/validation
    )
    if payload.get("uv") is not None:
        try:
            from trimesh.visual import TextureVisuals
            material = None
            if payload.get("material") is not None:
                try:
                    material = recurse(payload["material"])
                except Exception:
                    pass
            mesh.visual = TextureVisuals(
                uv=recurse(payload["uv"]), material=material)
        except Exception:
            pass
    if payload.get("metadata") is not None:
        try:
            mesh.metadata.update(recurse(payload["metadata"]))
        except Exception:
            pass
    return mesh


# Register deserialize only where trimesh exists; a side without it holds
# materialized OpaquePayload receipts (module docstring). Base-class
# registration: MRO matching makes Trimesh subclasses ride along.
try:
    import trimesh  # noqa: F401
    _DESERIALIZE = _deserialize_trimesh
except ImportError:
    _DESERIALIZE = None

register_serializer(
    "Trimesh", _serialize_trimesh, _DESERIALIZE,
    tag="trimesh.Trimesh",
)
