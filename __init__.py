import logging

log = logging.getLogger("geometrypack")

log.info("loading...")
from comfy_env import register_nodes
log.info("calling register_nodes")

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()


# /gpack/getpath -- directory listing for path-autocomplete widgets (e.g. Load
# Mesh Batch's folder picker). Registered here (main process, __init__ runs in
# init_extra_nodes AFTER PromptServer exists; prestartup is too early). Returns
# directory entries: folders as "<name>/", files filtered by ?extensions=a,b,c.
def _register_gpack_routes():
    import os
    try:
        from server import PromptServer
        from aiohttp import web
    except Exception as e:
        log.warning("[GeomPack] route deps unavailable: %s", e)
        return
    inst = getattr(PromptServer, "instance", None)
    if inst is None or getattr(inst, "_gpack_getpath_registered", False):
        return

    @inst.routes.get("/gpack/getpath")
    async def _gpack_getpath(request):
        q = request.rel_url.query
        if "path" not in q:
            return web.Response(status=204)
        p = os.path.abspath(os.path.expanduser(q.get("path") or "."))
        if not os.path.isdir(p):
            p = os.path.dirname(p)
        if not os.path.isdir(p):
            return web.json_response([])
        exts = q.get("extensions")
        ext_set = {e.lower().lstrip(".") for e in exts.split(",")} if exts else None
        items = []
        try:
            for it in os.scandir(p):
                try:
                    if it.is_dir():
                        items.append(it.name + "/")
                    elif ext_set is None or it.name.rsplit(".", 1)[-1].lower() in ext_set:
                        items.append(it.name)
                except OSError:
                    pass
        except OSError:
            return web.json_response([])
        items.sort()
        return web.json_response(items)

    inst._gpack_getpath_registered = True
    log.info("[GeomPack] /gpack/getpath route registered")


try:
    _register_gpack_routes()
except Exception as e:
    log.warning("[GeomPack] could not register /gpack/getpath: %s", e)


WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
