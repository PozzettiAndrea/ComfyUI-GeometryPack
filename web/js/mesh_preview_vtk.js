/**
 * ComfyUI GeomPack - VTK.js Mesh Preview Widget
 * Scientific visualization with VTK.js
 */

import { app } from "../../../scripts/app.js";

console.log("[GeomPack] Loading VTK.js mesh preview extension...");

app.registerExtension({
    name: "geompack.meshpreview.vtk",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackPreviewMeshVTK") {
            console.log("[GeomPack] Registering Preview Mesh (VTK) node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log("[GeomPack] VTK node created, adding widget");

                // Create iframe for VTK.js viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.height = "500px";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";

                // Point to VTK.js HTML viewer (with cache buster)
                iframe.src = "/extensions/ComfyUI-GeomPack/viewer_vtk.html?v=" + Date.now();

                // Add widget
                const widget = this.addDOMWidget("preview_vtk", "MESH_PREVIEW_VTK", iframe);
                widget.computeSize = () => [512, 520];

                // Store iframe reference
                this.meshViewerIframeVTK = iframe;

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[GeomPack VTK] onExecuted called with message:", message);
                    onExecuted?.apply(this, arguments);

                    // The message IS the UI data (not message.ui)
                    if (message?.mesh_file && message.mesh_file[0]) {
                        const filename = message.mesh_file[0];
                        console.log(`[GeomPack VTK] Loading mesh: ${filename}`);

                        // ComfyUI serves output files via /view API endpoint
                        const filepath = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;

                        // Send message to iframe (with delay to ensure iframe is loaded)
                        setTimeout(() => {
                            if (iframe.contentWindow) {
                                console.log(`[GeomPack VTK] Sending postMessage to iframe: ${filepath}`);
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_MESH",
                                    filepath: filepath,
                                    timestamp: Date.now()
                                }, "*");
                            } else {
                                console.error("[GeomPack VTK] Iframe contentWindow not available");
                            }
                        }, 100);
                    } else {
                        console.log("[GeomPack VTK] No mesh_file in message data. Keys:", Object.keys(message || {}));
                    }
                };

                return r;
            };
        }
    }
});

console.log("[GeomPack] VTK.js mesh preview extension registered");
