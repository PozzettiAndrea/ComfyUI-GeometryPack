// Load Mesh Batch: folder-path autocomplete (type to navigate directories),
// inspired by VHS "Load Video (Path)" but using GeometryPack's own /gpack/getpath
// route and restricted to FOLDERS. Replaces the plain folder_path text widget
// with a real <input> + <datalist> so the browser shows live folder suggestions.
import { app } from "../../../scripts/app.js";

const TAG = "[MeshBatchPath]";
console.log(`${TAG} script loaded`);

// Split "output/meshes/fo" -> ["output/meshes/", "fo"] (dir, remainder)
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

            // Remove the original folder_path string widget; we replace it with a
            // DOM input of the same name so it still serializes as folder_path.
            const idx = (node.widgets || []).findIndex((w) => w.name === "folder_path");
            const initial = idx >= 0 ? (node.widgets[idx].value ?? "3d") : "3d";
            if (idx >= 0) node.widgets.splice(idx, 1);

            const wrap = document.createElement("div");
            wrap.style.cssText = "width:100%;box-sizing:border-box;padding:0 6px;";
            const input = document.createElement("input");
            input.type = "text";
            input.value = initial;
            input.placeholder = "folder path — type to autocomplete (e.g. output/…)";
            const listId = `gpack_paths_${node.id}_${Math.floor(performance.now())}`;
            input.setAttribute("list", listId);
            input.style.cssText = "width:100%;box-sizing:border-box;background:#222;color:#ddd;" +
                "border:1px solid #444;border-radius:4px;padding:3px 6px;font:12px monospace;";
            const dl = document.createElement("datalist");
            dl.id = listId;
            wrap.appendChild(input);
            wrap.appendChild(dl);

            const widget = node.addDOMWidget("folder_path", "gpack_path", wrap, {
                getValue() { return input.value; },
                setValue(v) { input.value = v ?? ""; },
                serialize: true,
            });
            widget.computeSize = (w) => [w, 30];

            // Refetch directory contents only when the directory part changes.
            let lastDir = null;
            const refresh = async () => {
                const [dir] = pathStem(input.value);
                if (dir === lastDir) return;
                lastDir = dir;
                try {
                    const resp = await fetch(`/gpack/getpath?path=${encodeURIComponent(dir || ".")}`);
                    if (!resp.ok) return;
                    const items = await resp.json();
                    dl.innerHTML = "";
                    for (const it of items) {
                        if (!it.endsWith("/")) continue;       // folders only
                        const opt = document.createElement("option");
                        opt.value = dir + it;                  // full path so it chains
                        dl.appendChild(opt);
                    }
                } catch (e) { /* route missing / offline -> stays plain text */ }
            };
            input.addEventListener("input", refresh);
            input.addEventListener("focus", () => { lastDir = null; refresh(); });

            return r;
        };
    },
});
