// Preview Mesh Boundaries: interactive VTK.js viewer in the node. On execute the
// Python node writes a .vtp (surface + thresholded boundary edges) to temp; here
// we mount viewer_vtk.html and postMessage it the file URL to load.
import { app } from "../../../scripts/app.js";

const TAG = "[PreviewMeshBoundaries]";
console.log(`${TAG} script loaded`);

const EXTENSION_FOLDER = (() => {
    const m = import.meta.url.match(/\/extensions\/([^/]+)\//);
    return m ? m[1] : "ComfyUI-GeometryPack";
})();

// Build a /view URL from a payload filename. Handles bare temp names and
// type-prefixed relative paths (output/.. input/.. temp/..).
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
    name: "geompack.previewmeshboundaries",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GeomPackPreviewMeshBoundaries") return;
        console.log(`${TAG} registering for GeomPackPreviewMeshBoundaries`);

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;

            const iframe = document.createElement("iframe");
            iframe.style.cssText = "width:100%;height:100%;border:none;background:#2a2a2a;aspect-ratio:1;";
            iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_vtk.html?v=` + Date.now();

            const widget = node.addDOMWidget("preview", "MESH_PREVIEW", iframe, {
                getValue() { return ""; }, setValue() { },
            });
            widget.computeSize = (w) => [w || 512, w || 512];
            node.size = [Math.max(node.size?.[0] || 0, 380), Math.max(node.size?.[1] || 0, 420)];
            node._gpBoundIframe = iframe;
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const iframe = this._gpBoundIframe;
            const name = message?.mesh_file?.[0];
            if (!iframe || !name) {
                console.warn(`${TAG} no mesh_file in payload`, message && Object.keys(message));
                return;
            }
            const url = viewUrl(name);
            const send = () => {
                if (iframe.contentWindow) {
                    console.log(`${TAG} loading ${url}`);
                    iframe.contentWindow.postMessage(
                        { type: "LOAD_MESH", filepath: url, timestamp: Date.now() }, "*");
                }
            };
            setTimeout(send, 250);   // give the iframe time if just mounted
        };
    },
});
