"""GeometryPack custom wire types (comfy-env serializer registry, ADR-0014).

Declared in nodes/comfy-env.toml under [serializers].modules; imported for
its registration side effects on BOTH sides of the process boundary:
  - parent: comfy-env loads it at register_nodes() (pack root on sys.path)
  - worker: loaded at startup via COMFY_ENV_SERIALIZER_MODULES

Registers trimesh.Trimesh -- the type behind all 305 io.Custom("TRIMESH")
sockets in this pack. Geometry (vertices/faces) decomposes into
shared-memory arrays instead of riding the 3-copy pickle rung; the small
remainder (visual, metadata) still travels via the transport's fallback,
with the visual's mesh BACK-REFERENCE cleared first so it cannot drag the
whole geometry into a pickle again.

Minor socket types (SKELETON, INTRINSICS/EXTRINSICS, VOXELGRID -- 13
sockets total) intentionally stay on the default path for now.
"""

try:  # parent process (comfy-env installed)
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
                frame = recurse(material)
                # The transport's fallback returns the RAW object when it
                # cannot encode it (instead of raising) -- keep only real
                # frames, or the control message stops being JSON-safe.
                if isinstance(frame, (dict, list, str, int, float, bool,
                                      type(None))):
                    payload["material"] = frame
            except Exception:
                pass  # material needs deps this side lacks; uv still travels

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


# Base-class registration: MRO matching makes Trimesh subclasses ride along.
# Prefixed tag per ADR-0014 (global last-wins tag namespace).
register_serializer(
    "Trimesh", _serialize_trimesh, _deserialize_trimesh,
    tag="geompack.Trimesh",
)
