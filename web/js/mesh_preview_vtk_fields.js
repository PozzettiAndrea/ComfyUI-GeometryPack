/**
 * ComfyUI GeomPack - VTK.js Mesh Preview with Scalar Fields Widget
 * Optimized for scalar field visualization with color mapping
 */

import { app } from "../../../scripts/app.js";

console.log("[GeomPack] Loading VTK.js mesh preview with scalar fields extension...");

app.registerExtension({
    name: "geompack.meshpreview.vtk.fields",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackPreviewMeshVTKFields") {
            console.log("[GeomPack] Registering Preview Mesh (VTK with Fields) node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log("[GeomPack VTK Fields] Node created, adding widget");

                // Create container for viewer + info panel
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#2a2a2a";

                // Create iframe for VTK.js viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";
                iframe.style.aspectRatio = "1";

                // Point to VTK.js HTML viewer with filters (with cache buster)
                // Use viewer_vtk_filters.html which has full scalar field visualization support
                iframe.src = "/extensions/ComfyUI-GeometryPack/viewer_vtk_filters.html?v=" + Date.now();

                // Create mesh info panel
                const infoPanel = document.createElement("div");
                infoPanel.style.backgroundColor = "#1a1a1a";
                infoPanel.style.borderTop = "1px solid #444";
                infoPanel.style.padding = "8px 12px";
                infoPanel.style.fontSize = "11px";
                infoPanel.style.fontFamily = "monospace";
                infoPanel.style.color = "#ccc";
                infoPanel.style.lineHeight = "1.4";
                infoPanel.innerHTML = '<span style="color: #888;">Mesh info will appear here after execution</span>';

                // Add iframe and info panel to container
                container.appendChild(iframe);
                container.appendChild(infoPanel);

                // Add widget with required options
                const widget = this.addDOMWidget("preview_vtk_fields", "MESH_PREVIEW_VTK_FIELDS", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                widget.computeSize = () => [512, 560];  // Increased height for info panel

                // Store iframe and info panel references
                this.meshViewerIframeVTKFields = iframe;
                this.meshInfoPanelVTKFields = infoPanel;

                // Track iframe load state
                let iframeLoaded = false;
                iframe.addEventListener('load', () => {
                    console.log("[GeomPack VTK Fields] Iframe loaded");
                    iframeLoaded = true;
                });

                // Set initial node size
                this.setSize([512, 560]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[GeomPack VTK Fields] onExecuted called with message:", message);
                    onExecuted?.apply(this, arguments);

                    // The message IS the UI data (not message.ui)
                    if (message?.mesh_file && message.mesh_file[0]) {
                        const filename = message.mesh_file[0];
                        console.log(`[GeomPack VTK Fields] Loading mesh: ${filename}`);

                        // Update mesh info panel with metadata
                        const vertices = message.vertex_count?.[0] || 'N/A';
                        const faces = message.face_count?.[0] || 'N/A';
                        const boundsMin = message.bounds_min?.[0] || [];
                        const boundsMax = message.bounds_max?.[0] || [];
                        const extents = message.extents?.[0] || [];
                        const fieldNames = message.field_names?.[0] || [];

                        // Format bounds
                        let boundsStr = 'N/A';
                        if (boundsMin.length === 3 && boundsMax.length === 3) {
                            boundsStr = `[${boundsMin.map(v => v.toFixed(2)).join(', ')}] to [${boundsMax.map(v => v.toFixed(2)).join(', ')}]`;
                        }

                        // Format extents
                        let extentsStr = 'N/A';
                        if (extents.length === 3) {
                            extentsStr = `${extents.map(v => v.toFixed(2)).join(' × ')}`;
                        }

                        // Build info HTML
                        let infoHTML = `
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 4px 8px;">
                                <span style="color: #888;">Vertices:</span>
                                <span>${vertices.toLocaleString()}</span>

                                <span style="color: #888;">Faces:</span>
                                <span>${faces.toLocaleString()}</span>

                                <span style="color: #888;">Bounds:</span>
                                <span style="font-size: 10px;">${boundsStr}</span>

                                <span style="color: #888;">Extents:</span>
                                <span>${extentsStr}</span>
                        `;

                        // Add watertight status
                        if (message.is_watertight !== undefined) {
                            const watertight = message.is_watertight[0] ? 'Yes' : 'No';
                            const color = message.is_watertight[0] ? '#6c6' : '#c66';
                            infoHTML += `
                                <span style="color: #888;">Watertight:</span>
                                <span style="color: ${color};">${watertight}</span>
                            `;
                        }

                        // Add scalar fields with highlighting
                        if (fieldNames.length > 0) {
                            const fields = fieldNames.join(', ');
                            infoHTML += `
                                <span style="color: #888;">Scalar Fields:</span>
                                <span style="font-size: 10px; color: #6c6; font-weight: bold;">${fields}</span>
                            `;
                        } else {
                            infoHTML += `
                                <span style="color: #888;">Scalar Fields:</span>
                                <span style="color: #c66;">None (use dropdown to select)</span>
                            `;
                        }

                        infoHTML += '</div>';

                        infoPanel.innerHTML = infoHTML;

                        // ComfyUI serves output files via /view API endpoint
                        const filepath = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;

                        // Function to send message
                        const sendMessage = () => {
                            if (iframe.contentWindow) {
                                console.log(`[GeomPack VTK Fields] Sending postMessage to iframe: ${filepath}`);
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_MESH",
                                    filepath: filepath,
                                    timestamp: Date.now()
                                }, "*");
                            } else {
                                console.error("[GeomPack VTK Fields] Iframe contentWindow not available");
                            }
                        };

                        // Send message after iframe is loaded
                        if (iframeLoaded) {
                            console.log("[GeomPack VTK Fields] Iframe already loaded, sending immediately");
                            sendMessage();
                        } else {
                            console.log("[GeomPack VTK Fields] Waiting for iframe to load...");
                            setTimeout(sendMessage, 500);
                        }
                    } else {
                        console.log("[GeomPack VTK Fields] No mesh_file in message data. Keys:", Object.keys(message || {}));
                    }
                };

                return r;
            };
        }
    }
});

console.log("[GeomPack] VTK.js mesh preview with scalar fields extension registered");
