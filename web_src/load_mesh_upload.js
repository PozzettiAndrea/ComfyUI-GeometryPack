// Load Mesh: drag-n-drop / upload (with progress bar) + inline "View 3D",
// for GeomPackLoadMesh. Upload mirrors ComfyUI's official Load Image
// (POST /upload/image, subfolder=3d) but via XHR so we get upload progress.
import { app } from "../../../scripts/app.js";

const TAG = "[LoadMeshUpload]";
console.log(`${TAG} script loaded`);

const EXTENSION_FOLDER = (() => {
    const m = import.meta.url.match(/\/extensions\/([^/]+)\//);
    return m ? m[1] : "ComfyUI-GeometryPack";
})();

const EXTS = [".obj", ".ply", ".stl", ".off", ".gltf", ".glb", ".fbx", ".dae", ".3ds", ".vtp"];
const ACCEPT = EXTS.join(",");
const isMesh = (name) => EXTS.some((x) => name.toLowerCase().endsWith(x));

// XHR upload so we can report upload progress (fetch can't). Returns combo value.
function uploadMesh(file, onProgress) {
    return new Promise((resolve, reject) => {
        const body = new FormData();
        body.append("image", file, file.name);
        body.append("subfolder", "3d");
        body.append("type", "input");
        body.append("overwrite", "true");
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/upload/image");
        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total, e.loaded, e.total);
        };
        xhr.onload = () => {
            if (xhr.status === 200) {
                try {
                    const d = JSON.parse(xhr.responseText);
                    const sub = (d.subfolder || "").replace(/\\/g, "/");
                    resolve(sub ? `${sub}/${d.name}` : d.name);
                } catch (e) { reject(e); }
            } else {
                reject(new Error(`${xhr.status} ${xhr.responseText}`));
            }
        };
        xhr.onerror = () => reject(new Error("network error"));
        if (onProgress) onProgress(0, 0, file.size);
        xhr.send(body);
    });
}

function viewUrlFor(value) {
    const parts = String(value).replace(/\\/g, "/").split("/");
    const fname = parts.pop();
    return `/view?filename=${encodeURIComponent(fname)}&type=input&subfolder=${encodeURIComponent(parts.join("/"))}`;
}

function fileWidget(node) {
    return node.widgets?.find((x) => x.name === "file_path");
}

function selectValue(node, val) {
    const w = fileWidget(node);
    if (!w) { console.warn(`${TAG} no file_path widget`); return; }
    w.options = w.options || {};
    w.options.values = w.options.values || [];
    if (!w.options.values.includes(val)) w.options.values.push(val);
    w.value = val;
    try { w.callback?.(val); } catch (e) { /* noop */ }
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "geompack.loadmesh.upload",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GeomPackLoadMesh") return;
        console.log(`${TAG} registering for GeomPackLoadMesh`);

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;

            // --- progress bar (DOM widget; collapses to 0 height when idle) ---
            const wrap = document.createElement("div");
            wrap.style.cssText = "width:100%;padding:0 6px;box-sizing:border-box;display:none;";
            const label = document.createElement("div");
            label.style.cssText = "font:10px monospace;color:#bbb;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
            const track = document.createElement("div");
            track.style.cssText = "width:100%;height:6px;background:rgba(255,255,255,0.18);border-radius:3px;overflow:hidden;";
            const bar = document.createElement("div");
            bar.style.cssText = "width:0%;height:100%;background:#00c8ff;transition:width 0.1s linear;";
            track.appendChild(bar); wrap.appendChild(label); wrap.appendChild(track);
            const progWidget = node.addDOMWidget("upload_progress", "div", wrap, {
                getValue() { return ""; }, setValue() { },
            });
            progWidget.computeSize = (w) => (wrap.style.display === "none" ? [w, 0] : [w, 26]);
            const showProgress = (name, frac) => {
                wrap.style.display = "block";
                const pct = Math.max(0, Math.min(100, Math.round((frac || 0) * 100)));
                label.textContent = `⬆ ${name} — ${pct}%`;
                bar.style.width = pct + "%";
                node.setDirtyCanvas(true, true);
            };
            const hideProgress = () => { wrap.style.display = "none"; node.setDirtyCanvas(true, true); };

            async function uploadList(files) {
                const meshes = [...files].filter((f) => isMesh(f.name));
                if (!meshes.length) { console.warn(`${TAG} no mesh files in selection/drop`); return false; }
                for (const f of meshes) {
                    try {
                        showProgress(f.name, 0);
                        const val = await uploadMesh(f, (frac) => showProgress(f.name, frac));
                        selectValue(node, val);
                        console.log(`${TAG} uploaded -> ${val}`);
                    } catch (e) {
                        console.error(`${TAG} upload failed for ${f.name}`, e);
                        alert("Mesh upload failed: " + e.message);
                    } finally {
                        hideProgress();
                    }
                }
                return true;
            }
            node._gpUploadList = uploadList;

            // hidden file picker + upload button
            const input = document.createElement("input");
            input.type = "file"; input.accept = ACCEPT; input.multiple = true; input.style.display = "none";
            input.addEventListener("change", async () => { await uploadList(input.files); input.value = ""; });
            document.body.appendChild(input);
            node.addWidget("button", "⬆ upload / drop mesh", null, () => { console.log(`${TAG} upload button clicked`); input.click(); });

            // inline View 3D
            let iframe = null, viewWidget = null;
            const collapse = () => {
                if (viewWidget && node.widgets) {
                    const i = node.widgets.indexOf(viewWidget); if (i >= 0) node.widgets.splice(i, 1);
                }
                if (iframe) iframe.remove(); iframe = null; viewWidget = null; node.setDirtyCanvas(true, true);
            };
            const sendMesh = () => {
                const w = fileWidget(node);
                if (iframe?.contentWindow && w?.value)
                    iframe.contentWindow.postMessage({ type: "LOAD_MESH", filepath: viewUrlFor(w.value), timestamp: Date.now() }, "*");
            };
            node.addWidget("button", "👁 view 3d", null, () => {
                if (iframe) { collapse(); return; }
                const w = fileWidget(node);
                if (!w || !w.value) { alert("Pick or upload a mesh first."); return; }
                iframe = document.createElement("iframe");
                iframe.style.cssText = "width:100%;height:100%;border:none;background:#2a2a2a;";
                // VTK viewer: robust OBJ/PLY/STL/VTP reader (+ the download progress bar),
                // unlike the Three.js viewer.html whose OBJLoader chokes on some OBJs.
                iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_vtk.html?v=` + Date.now();
                viewWidget = node.addDOMWidget("loadmesh_view3d", "MESH_PREVIEW", iframe, { getValue() { return ""; }, setValue() { } });
                viewWidget.computeSize = (width) => [width || 360, width || 360];
                iframe.addEventListener("load", () => setTimeout(sendMesh, 150));
                setTimeout(sendMesh, 600);
                node.setSize([Math.max(node.size[0], 380), node.size[1] + 380]); node.setDirtyCanvas(true, true);
            });

            return r;
        };

        const onDragOver = nodeType.prototype.onDragOver;
        nodeType.prototype.onDragOver = function (e) {
            if ([...(e?.dataTransfer?.items || [])].some((it) => it.kind === "file")) return true;
            return onDragOver?.apply(this, arguments) ?? false;
        };
        const onDragDrop = nodeType.prototype.onDragDrop;
        nodeType.prototype.onDragDrop = async function (e) {
            console.log(`${TAG} drop; files=${e?.dataTransfer?.files?.length}`);
            if (this._gpUploadList && await this._gpUploadList(e?.dataTransfer?.files || [])) return true;
            return onDragDrop?.apply(this, arguments) ?? false;
        };
    },
});
