// Preview Mesh Slicer: interactive clip/slice viewer (viewer_slicer.html) in-node.
// On execute the Python node writes the mesh .vtp to temp; we mount the slicer
// iframe and postMessage it the file URL.
import { app } from "../../../scripts/app.js";

const TAG = "[PreviewMeshSlicer]";
console.log(`${TAG} script loaded`);

const EXTENSION_FOLDER = (() => {
    const m = import.meta.url.match(/\/extensions\/([^/]+)\//);
    return m ? m[1] : "ComfyUI-GeometryPack";
})();

function viewUrl(name) {
    const norm = String(name).replace(/\\/g, "/");
    const m = norm.match(/(?:^|\/)(output|input|temp)\/(.+)$/);
    if (m) {
        const [, type, rel] = m;
        const parts = rel.split("/");
        const fname = parts.pop();
        return `/view?filename=${encodeURIComponent(fname)}&type=${type}&subfolder=${encodeURIComponent(parts.join("/"))}`;
    }
    return `/view?filename=${encodeURIComponent(norm.split("/").pop())}&type=temp&subfolder=`;
}

app.registerExtension({
    name: "geompack.previewmeshslicer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GeomPackPreviewMeshSlicer") return;
        console.log(`${TAG} registering for GeomPackPreviewMeshSlicer`);

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;
            const iframe = document.createElement("iframe");
            iframe.style.cssText = "width:100%;height:100%;border:none;background:#2a2a2a;";
            iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_slicer.html?v=` + Date.now();
            const widget = node.addDOMWidget("slicer", "MESH_PREVIEW", iframe, {
                getValue() { return ""; }, setValue() { },
            });
            widget.computeSize = (w) => [w || 480, Math.round((w || 480) * 1.15)];
            node.size = [Math.max(node.size?.[0] || 0, 420), Math.max(node.size?.[1] || 0, 540)];
            node._gpSlicerIframe = iframe;

            // plane_json (a real schema input -> auto-created widget) is a machine-written
            // mirror of whatever clip plane is picked interactively in the iframe, not
            // something meant to be hand-edited -- hide it from the node body.
            const planeWidget = node.widgets?.find((w) => w.name === "plane_json");
            if (planeWidget) {
                planeWidget.hidden = true;
                planeWidget.computeSize = () => [0, -4];
            }

            // The clip plane only ever exists as local JS state inside the iframe
            // (viewer_slicer.html) -- execute() has no other way to see it, so mirror it
            // into the (hidden) plane_json widget on every change; that's what gets
            // serialized into the prompt on the next queue.
            window.addEventListener("message", (event) => {
                if (event.source !== iframe.contentWindow) return;
                if (event.data?.type !== "PLANE_UPDATE") return;
                const w = node.widgets?.find((x) => x.name === "plane_json");
                if (!w) return;
                w.value = JSON.stringify(event.data);
                node.setDirtyCanvas(true, true);
            });

            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const iframe = this._gpSlicerIframe;
            const name = message?.mesh_file?.[0];
            if (!iframe || !name) return;
            const url = viewUrl(name);
            setTimeout(() => {
                if (iframe.contentWindow) {
                    console.log(`${TAG} loading ${url}`);
                    iframe.contentWindow.postMessage({ type: "LOAD_MESH", filepath: url, timestamp: Date.now() }, "*");
                }
            }, 250);
        };
    },
});
