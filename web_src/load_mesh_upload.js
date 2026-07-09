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
            //
            // IMPORTANT: ComfyUI's DOM-widget host <div> ("dom-widget size-full",
            // position:fixed) is positioned/sized by its OWN reactive layer
            // (Vue), driven by widget.isVisible()/widget.hidden -- it does NOT
            // look at this element's own inline CSS (wrap.style.display) at all.
            // Toggling only wrap.style.display leaves the *host* div fully
            // "visible" from ComfyUI's point of view, at which point its height
            // falls back to the "size-full" CSS class default (100% of the
            // *viewport*, since it's position:fixed) with pointer-events:auto --
            // an invisible, click-eating overlay stretching far below the node,
            // permanently, regardless of whether anything is actually showing.
            // Confirmed empirically (Playwright): computeSize()'s return value
            // is not what drives this host div's real layout height either.
            // The only thing that correctly collapses the host div (display:none
            // AND pointer-events:none together) is the widget's own `hidden`
            // flag, so that's the actual on/off switch -- wrap's inline style
            // just controls the *content* once the host div is shown.
            const wrap = document.createElement("div");
            wrap.style.cssText = "width:100%;padding:0 6px;box-sizing:border-box;";
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
            progWidget.computeSize = (w) => [w, 26];
            progWidget.hidden = true; // starts idle -- see note above
            const showProgress = (name, frac) => {
                progWidget.hidden = false;
                const pct = Math.max(0, Math.min(100, Math.round((frac || 0) * 100)));
                label.textContent = `⬆ ${name} — ${pct}%`;
                bar.style.width = pct + "%";
                node.setDirtyCanvas(true, true);
            };
            const hideProgress = () => { progWidget.hidden = true; node.setDirtyCanvas(true, true); };

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
            node._gpFileInput = input; // removed in onRemoved below
            node.addWidget("button", "⬆ upload / drop mesh", null, () => { console.log(`${TAG} upload button clicked`); input.click(); });

            // inline View 3D
            let iframe = null, viewWidget = null, savedSize = null;
            const collapse = () => {
                if (viewWidget && node.widgets) {
                    const i = node.widgets.indexOf(viewWidget); if (i >= 0) node.widgets.splice(i, 1);
                }
                if (iframe) iframe.remove(); iframe = null; viewWidget = null;
                // Restore the pre-expand size instead of leaving the +380px
                // stuck on the node -- otherwise every open/close cycle
                // permanently grows the node's bounding box downward, which
                // ends up silently overlapping (and blocking clicks on)
                // whatever node sits below it on the canvas.
                if (savedSize) { node.setSize(savedSize); savedSize = null; }
                node.setDirtyCanvas(true, true);
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
                savedSize = [...node.size];
                iframe = document.createElement("iframe");
                iframe.style.cssText = "width:100%;height:100%;border:none;background:#2a2a2a;";
                // VTK viewer: robust OBJ/PLY/STL/VTP reader (+ the download progress bar),
                // unlike the Three.js viewer.html whose OBJLoader chokes on some OBJs.
                iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_vtk.html?v=` + Date.now();
                viewWidget = node.addDOMWidget("loadmesh_view3d", "MESH_PREVIEW", iframe, { getValue() { return ""; }, setValue() { } });
                viewWidget.computeSize = (width) => [width || 360, width || 360];
                iframe.addEventListener("load", () => setTimeout(sendMesh, 150));
                setTimeout(sendMesh, 600);
                node.setSize([Math.max(node.size[0], 380), savedSize[1] + 380]); node.setDirtyCanvas(true, true);
            });

            return r;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this._gpFileInput?.remove();
            return onRemoved?.apply(this, arguments);
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
