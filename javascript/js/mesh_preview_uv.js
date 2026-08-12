/**
 * ComfyUI GeomPack - UV Mesh Preview Widget
 * Synchronized 3D mesh and 2D UV layout viewer with interactive picking
 */

import { app } from "../../../scripts/app.js";

// Auto-detect extension folder name (handles ComfyUI-GeometryPack or comfyui-geometrypack)
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-GeometryPack";
})();

console.log("[GeomPack] Loading UV mesh preview extension...");

app.registerExtension({
    name: "geompack.meshpreviewuv",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackPreviewMeshUV") {
            console.log("[GeomPack] Registering Preview Mesh UV node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log("[GeomPack UV] Node created, adding widget");

                // Create iframe for UV viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.height = "100%";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";
                iframe.style.aspectRatio = "2";  // Wide aspect for split view

                // Point to our UV viewer HTML
                iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_uv.html?v=` + Date.now();

                // Add widget
                const widget = this.addDOMWidget("uv_preview", "MESH_UV_PREVIEW", iframe, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                // Set widget size - wider for split view
                widget.computeSize = function(width) {
                    const w = width || 600;
                    const h = w / 2;  // 2:1 aspect ratio
                    return [w, h];
                };

                widget.element = iframe;

                // Store iframe reference
                this.uvViewerIframe = iframe;

                // Set initial node size (wide for split view)
                this.setSize([600, 350]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[GeomPack UV] onExecuted called with message:", message);
                    onExecuted?.apply(this, arguments);

                    if (message?.mesh_file && message.mesh_file[0]) {
                        const meshFilename = message.mesh_file[0];
                        const uvDataFilename = message.uv_data_file ? message.uv_data_file[0] : null;
                        const showChecker = message.show_checker ? message.show_checker[0] : false;
                        const showWireframe = message.show_wireframe ? message.show_wireframe[0] : true;

                        console.log(`[GeomPack UV] Loading mesh: ${meshFilename}`);
                        console.log(`[GeomPack UV] UV data: ${uvDataFilename}`);
                        console.log(`[GeomPack UV] Checker: ${showChecker}, Wireframe: ${showWireframe}`);

                        // Build file URLs
                        const meshPath = `/view?filename=${encodeURIComponent(meshFilename)}&type=output&subfolder=`;
                        const uvDataPath = uvDataFilename ?
                            `/view?filename=${encodeURIComponent(uvDataFilename)}&type=output&subfolder=` : null;

                        // Send message to iframe
                        setTimeout(() => {
                            if (iframe.contentWindow) {
                                console.log(`[GeomPack UV] Sending postMessage to iframe`);
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_MESH_UV",
                                    meshPath: meshPath,
                                    uvDataPath: uvDataPath,
                                    checker: showChecker,
                                    wireframe: showWireframe,
                                    timestamp: Date.now()
                                }, "*");
                            } else {
                                console.error("[GeomPack UV] Iframe contentWindow not available");
                            }
                        }, 100);

                        // Update node title with UV info
                        if (message.has_uvs && message.has_uvs[0]) {
                            const coverage = message.uv_coverage ? message.uv_coverage[0].toFixed(3) : "?";
                            const inUnit = message.uv_in_unit_square ? message.uv_in_unit_square[0] : false;
                            console.log(`[GeomPack UV] UV Coverage: ${coverage}, In Unit Square: ${inUnit}`);
                        } else {
                            console.warn("[GeomPack UV] No UV data found on mesh");
                        }
                    } else {
                        console.log("[GeomPack UV] No mesh_file in message data. Keys:", Object.keys(message || {}));
                    }
                };

                return r;
            };
        }
    }
});

console.log("[GeomPack] UV mesh preview extension registered");
