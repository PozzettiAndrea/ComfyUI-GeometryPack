// Load Mesh Batch: folder-path autocomplete (type to navigate directories),
// inspired by VHS "Load Video (Path)" but using GeometryPack's own /gpack/getpath
// route and restricted to FOLDERS. Uses a CUSTOM floating dropdown (native
// <datalist> is unreliable inside ComfyUI's transformed DOM widgets).
import { app } from "../../../scripts/app.js";

const TAG = "[MeshBatchPath]";
console.log(`${TAG} script loaded`);

// "output/meshes/fo" -> ["output/meshes/", "fo"]  (dir, remainder)
function pathStem(value) {
    const v = String(value || "");
    const i = v.lastIndexOf("/");
    return i < 0 ? ["", v] : [v.slice(0, i + 1), v.slice(i + 1)];
}

app.registerExtension({
    name: "geompack.loadmeshbatch.path",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GeomPackLoadMeshBatch") return;
        console.log(`${TAG} registering for GeomPackLoadMeshBatch`);

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;

            // Replace the plain folder_path string widget with a DOM input of the
            // same name (so it still serializes as folder_path).
            const idx = (node.widgets || []).findIndex((w) => w.name === "folder_path");
            const initial = idx >= 0 ? (node.widgets[idx].value ?? "3d") : "3d";
            if (idx >= 0) node.widgets.splice(idx, 1);

            const wrap = document.createElement("div");
            wrap.style.cssText = "width:100%;box-sizing:border-box;padding:0 6px;";
            const input = document.createElement("input");
            input.type = "text";
            input.value = initial;
            input.placeholder = "folder path — type to autocomplete (input/… output/… or absolute)";
            input.style.cssText = "width:100%;box-sizing:border-box;background:#222;color:#ddd;" +
                "border:1px solid #444;border-radius:4px;padding:3px 6px;font:12px monospace;";
            wrap.appendChild(input);

            const widget = node.addDOMWidget("folder_path", "gpack_path", wrap, {
                getValue() { return input.value; },
                setValue(v) { input.value = v ?? ""; },
                serialize: true,
            });
            widget.computeSize = (w) => [w, 30];

            // Floating dropdown attached to body (avoids canvas clipping/transform).
            const menu = document.createElement("div");
            menu.style.cssText = "position:fixed;z-index:10000;max-height:240px;overflow-y:auto;" +
                "background:#1e1e1e;border:1px solid #555;border-radius:4px;" +
                "box-shadow:0 4px 14px rgba(0,0,0,0.55);font:12px monospace;display:none;";
            document.body.appendChild(menu);

            let items = [];
            let lastDir = null;
            const hide = () => { menu.style.display = "none"; };
            const place = () => {
                const b = input.getBoundingClientRect();
                menu.style.left = b.left + "px";
                menu.style.top = (b.bottom + 2) + "px";
                menu.style.minWidth = b.width + "px";
            };
            const render = (rem) => {
                const remL = rem.toLowerCase();
                const [dir] = pathStem(input.value);
                const matches = items.filter(
                    (it) => it.endsWith("/") && it.toLowerCase().startsWith(remL));
                menu.innerHTML = "";
                if (!matches.length) { hide(); return; }
                for (const it of matches) {
                    const row = document.createElement("div");
                    row.textContent = it;
                    row.style.cssText = "padding:4px 9px;color:#ddd;cursor:pointer;white-space:nowrap;";
                    row.addEventListener("mouseenter", () => { row.style.background = "#37506b"; });
                    row.addEventListener("mouseleave", () => { row.style.background = ""; });
                    row.addEventListener("mousedown", (e) => {
                        e.preventDefault();              // keep input focused
                        input.value = dir + it;          // descend into the folder
                        fetchDir(true);                  // list its contents
                    });
                    menu.appendChild(row);
                }
                place();
                menu.style.display = "block";
            };
            const fetchDir = async (force) => {
                const [dir, rem] = pathStem(input.value);
                if (!force && dir === lastDir) { render(rem); return; }
                lastDir = dir;
                try {
                    const resp = await fetch(`/gpack/getpath?path=${encodeURIComponent(dir || ".")}`);
                    items = resp.ok ? await resp.json() : [];
                } catch (e) { items = []; }
                render(rem);
            };

            input.addEventListener("input", () => fetchDir(false));
            input.addEventListener("focus", () => { lastDir = null; fetchDir(true); });
            input.addEventListener("blur", () => setTimeout(hide, 200));
            input.addEventListener("keydown", (e) => { if (e.key === "Escape") hide(); });

            const origRemoved = node.onRemoved;
            node.onRemoved = function () { menu.remove(); return origRemoved?.apply(this, arguments); };

            return r;
        };
    },
});
