// Load Mesh: drag-n-drop / upload + inline "View 3D", for GeomPackLoadMesh.
//
// Upload mirrors ComfyUI's official Load Image: drop a mesh (or click the
// button) -> POST to /upload/image with subfolder=3d -> lands in input/3d/ and
// is added to the file_path combo. View 3D mounts the shared vtk.js viewer
// (viewer.html) inline in the node and postMessages the selected file's URL.
import { app } from "../../../scripts/app.js";

const EXTENSION_FOLDER = (() => {
    const m = import.meta.url.match(/\/extensions\/([^/]+)\//);
    return m ? m[1] : "ComfyUI-GeometryPack";
})();

const EXTS = [".obj", ".ply", ".stl", ".off", ".gltf", ".glb", ".fbx", ".dae", ".3ds", ".vtp"];
const ACCEPT = EXTS.join(",");
const isMesh = (name) => EXTS.some((x) => name.toLowerCase().endsWith(x));

// Upload one file to input/3d/ via ComfyUI's generic upload route.
// Returns the combo value (relative-to-input path, e.g. "3d/foo.obj").
async function uploadMesh(file) {
    const body = new FormData();
    body.append("image", file, file.name);   // route's field name is "image"
    body.append("subfolder", "3d");
    body.append("type", "input");
    body.append("overwrite", "true");          // skip image-hash dedup path
    const r = await fetch("/upload/image", { method: "POST", body });
    if (r.status !== 200) throw new Error(`${r.status} ${await r.text()}`);
    const d = await r.json();
    const sub = (d.subfolder || "").replace(/\\/g, "/");
    return sub ? `${sub}/${d.name}` : d.name;
}

// Build a /view URL for an input-relative combo value ("3d/foo.obj").
function viewUrlFor(value) {
    const parts = String(value).replace(/\\/g, "/").split("/");
    const fname = parts.pop();
    const subfolder = parts.join("/");
    return `/view?filename=${encodeURIComponent(fname)}&type=input&subfolder=${encodeURIComponent(subfolder)}`;
}

function fileWidget(node) {
    return node.widgets?.find((w) => w.name === "file_path");
}

function selectValue(node, val) {
    const w = fileWidget(node);
    if (!w) return;
    w.options = w.options || {};
    w.options.values = w.options.values || [];
    if (!w.options.values.includes(val)) w.options.values.push(val);
    w.value = val;
    try { w.callback?.(val); } catch (e) { /* noop */ }
    node.setDirtyCanvas(true, true);
}

async function uploadList(node, files) {
    const meshes = [...files].filter((f) => isMesh(f.name));
    if (!meshes.length) return false;
    for (const f of meshes) {
        try { selectValue(node, await uploadMesh(f)); }
        catch (e) { console.error("[GeomPack] mesh upload failed", e); alert("Mesh upload failed: " + e.message); }
    }
    return true;
}

app.registerExtension({
    name: "geompack.loadmesh.upload",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GeomPackLoadMesh") return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;

            // hidden file picker
            const input = document.createElement("input");
            input.type = "file";
            input.accept = ACCEPT;
            input.multiple = true;
            input.style.display = "none";
            input.addEventListener("change", async () => {
                await uploadList(node, input.files);
                input.value = "";
            });
            document.body.appendChild(input);

            node.addWidget("button", "⬆ upload / drop mesh", null, () => input.click());

            // --- inline View 3D (collapsed by default) ---
            let iframe = null;
            let viewWidget = null;
            const collapse = () => {
                if (viewWidget && node.widgets) {
                    const i = node.widgets.indexOf(viewWidget);
                    if (i >= 0) node.widgets.splice(i, 1);
                }
                if (iframe) iframe.remove();
                iframe = null; viewWidget = null;
                node.setDirtyCanvas(true, true);
            };
            const sendMesh = () => {
                const w = fileWidget(node);
                if (iframe?.contentWindow && w?.value) {
                    iframe.contentWindow.postMessage(
                        { type: "LOAD_MESH", filepath: viewUrlFor(w.value), timestamp: Date.now() }, "*");
                }
            };
            const toggleView = () => {
                if (iframe) { collapse(); return; }
                const w = fileWidget(node);
                if (!w || !w.value) { alert("Pick or upload a mesh first."); return; }
                iframe = document.createElement("iframe");
                iframe.style.cssText = "width:100%;height:100%;border:none;background:#2a2a2a;";
                iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer.html?v=` + Date.now();
                viewWidget = node.addDOMWidget("loadmesh_view3d", "MESH_PREVIEW", iframe, {
                    getValue() { return ""; }, setValue() { /* noop */ },
                });
                viewWidget.computeSize = (width) => [width || 360, width || 360];
                iframe.addEventListener("load", () => setTimeout(sendMesh, 150));
                setTimeout(sendMesh, 600);
                node.setSize([Math.max(node.size[0], 380), node.size[1] + 380]);
                node.setDirtyCanvas(true, true);
            };
            node.addWidget("button", "👁 view 3d", null, toggleView);
            node._gpReloadView = () => { if (iframe) sendMesh(); };

            return r;
        };

        // drag-drop a mesh straight onto the node
        const onDragOver = nodeType.prototype.onDragOver;
        nodeType.prototype.onDragOver = function (e) {
            if ([...(e?.dataTransfer?.items || [])].some((it) => it.kind === "file")) return true;
            return onDragOver?.apply(this, arguments) ?? false;
        };
        const onDragDrop = nodeType.prototype.onDragDrop;
        nodeType.prototype.onDragDrop = async function (e) {
            const handled = await uploadList(this, e?.dataTransfer?.files || []);
            if (handled) return true;
            return onDragDrop?.apply(this, arguments) ?? false;
        };
    },
});
