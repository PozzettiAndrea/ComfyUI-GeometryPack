/**
 * ComfyUI GeomPack - 3D Mesh Preview Widget
 * Minimal implementation based on ComfyUI-3D-Pack pattern
 */

import { app } from "../../../scripts/app.js";

// Auto-detect extension folder name (handles ComfyUI-GeometryPack or comfyui-geometrypack)
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-GeometryPack";
})();

console.log("[GeomPack] Loading mesh preview extension...");

app.registerExtension({
    name: "geompack.meshpreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackPreviewMesh") {
            console.log("[GeomPack] Registering Preview Mesh node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log("[GeomPack DEBUG] this.widgets before:", this.widgets);
                console.log("[GeomPack DEBUG] nodeData:", nodeData);
                console.log("[GeomPack] Node created, adding widget");

                // Create iframe for 3D viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.height = "100%";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";
                iframe.style.aspectRatio = "1";

                // Point to our HTML viewer (with cache buster)
                iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer.html?v=` + Date.now();

                // Add widget with required options
                console.log("[GeomPack DEBUG] About to call addDOMWidget");
                console.log("[GeomPack DEBUG] typeof this.addDOMWidget:", typeof this.addDOMWidget);

                const widget = this.addDOMWidget("preview", "MESH_PREVIEW", iframe, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                console.log("[GeomPack DEBUG] Widget created:", widget);
                console.log("[GeomPack DEBUG] Widget properties:", Object.keys(widget || {}));
                console.log("[GeomPack DEBUG] Widget.id:", widget?.id);
                console.log("[GeomPack DEBUG] this.widgets after addDOMWidget:", this.widgets);

                // Set widget size - computeSize returns [width, height]
                widget.computeSize = function(width) {
                    console.log("[GeomPack DEBUG] computeSize called with width:", width);
                    const size = [width || 512, width || 512];
                    console.log("[GeomPack DEBUG] computeSize returning:", size);
                    return size;
                };

                // Also try setting element size directly
                widget.element = iframe;

                console.log("[GeomPack DEBUG] Widget computeSize set");
                console.log("[GeomPack DEBUG] Iframe dimensions:", iframe.style.width, iframe.style.height);

                // Store iframe reference
                this.meshViewerIframe = iframe;

                // Set initial node size to be square
                this.setSize([512, 512]);
                console.log("[GeomPack DEBUG] Node size set to [512, 512]");

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[GeomPack] onExecuted called with message:", message);
                    onExecuted?.apply(this, arguments);

                    // The message IS the UI data (not message.ui)
                    if (message?.mesh_file && message.mesh_file[0]) {
                        const filename = message.mesh_file[0];
                        console.log(`[GeomPack] Loading mesh: ${filename}`);

                        // ComfyUI serves files via /view API — parse type/subfolder/filename
                        // Handles both absolute (/home/.../output/sub/file) and relative (output/sub/file) paths
                        let filepath;
                        const normalized = filename.replace(/\\/g, '/');
                        const pathMatch = normalized.match(/(?:^|\/)(output|input|temp)\/(.+)$/);
                        if (pathMatch) {
                            const [, type, relPath] = pathMatch;
                            const parts = relPath.split('/');
                            const fname = parts.pop();
                            const subfolder = parts.join('/');
                            filepath = `/view?filename=${encodeURIComponent(fname)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`;
                        } else {
                            // Fallback: use just the basename
                            const basename = normalized.split('/').pop();
                            filepath = `/view?filename=${encodeURIComponent(basename)}&type=output&subfolder=`;
                        }

                        // Send message to iframe (with delay to ensure iframe is loaded)
                        setTimeout(() => {
                            if (iframe.contentWindow) {
                                console.log(`[GeomPack] Sending postMessage to iframe: ${filepath}`);
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_MESH",
                                    filepath: filepath,
                                    timestamp: Date.now()
                                }, "*");
                            } else {
                                console.error("[GeomPack] Iframe contentWindow not available");
                            }
                        }, 100);
                    } else {
                        console.log("[GeomPack] No mesh_file in message data. Keys:", Object.keys(message || {}));
                    }
                };

                return r;
            };
        }
    }
});

console.log("[GeomPack] Mesh preview extension registered");
