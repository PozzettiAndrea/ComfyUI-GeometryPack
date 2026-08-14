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

    # /gpack/save_preview -- the viewers' "Save mesh" button. Copies a temp
    # preview export (preview_*.stl/vtp/glb, written to ComfyUI output/ or the
    # OS tempdir by the preview nodes) to a stable name in output/ so it
    # survives the next preview run. Body: {"temp_filename": "<basename>"}.
    @inst.routes.post("/gpack/save_preview")
    async def _gpack_save_preview(request):
        import shutil
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"success": False, "error": "invalid JSON body"}, status=400)
        # basename() defuses path traversal -- we only ever serve files the
        # preview nodes themselves wrote into these two directories.
        name = os.path.basename(str(body.get("temp_filename") or ""))
        if not name:
            return web.json_response({"success": False, "error": "temp_filename required"}, status=400)
        import tempfile
        try:
            import folder_paths
            out_dir = folder_paths.get_output_directory()
        except Exception:
            out_dir = None
        candidates = [d for d in (out_dir, tempfile.gettempdir()) if d]
        src = next((os.path.join(d, name) for d in candidates
                    if os.path.isfile(os.path.join(d, name))), None)
        if src is None:
            return web.json_response(
                {"success": False, "error": f"{name} not found (previews are temporary; re-run the node)"},
                status=404)
        if out_dir is None:
            return web.json_response({"success": False, "error": "no ComfyUI output directory"}, status=500)
        saved = f"saved_{name}"
        try:
            shutil.copy2(src, os.path.join(out_dir, saved))
        except OSError as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)
        log.info("[GeomPack] saved preview %s -> %s", name, saved)
        return web.json_response({"success": True, "saved_filename": saved})

    inst._gpack_getpath_registered = True
    log.info("[GeomPack] /gpack/getpath + /gpack/save_preview routes registered")


try:
    _register_gpack_routes()
except Exception as e:
    log.warning("[GeomPack] could not register /gpack/getpath: %s", e)


# Frontend assets are declared in pyproject.toml ([tool.comfy] web = "javascript"),
# served at /extensions/comfyui-geometrypack/. No WEB_DIRECTORY attribute (it would
# double-register the dir under a second mount key).
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
