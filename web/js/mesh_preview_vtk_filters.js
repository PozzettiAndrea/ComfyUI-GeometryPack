/**
 * ComfyUI GeomPack - VTK.js Mesh Preview with Filters Widget
 * Scientific visualization with VTK.js filters
 */

import { app } from "../../../scripts/app.js";

console.log("[GeomPack] Loading VTK.js mesh preview with filters extension...");

app.registerExtension({
    name: "geompack.meshpreview.vtk.filters",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackPreviewMeshVTKFilters") {
            console.log("[GeomPack] Registering Preview Mesh (VTK with Filters) node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log("[GeomPack VTK Filters DEBUG] this.widgets before:", this.widgets);
                console.log("[GeomPack VTK Filters DEBUG] nodeData:", nodeData);
                console.log("[GeomPack] VTK Filters node created, adding widget");

                // Create container for viewer + info panel
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#2a2a2a";

                // Create iframe for VTK.js viewer with filters
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";
                iframe.style.aspectRatio = "1";

                // Point to VTK.js HTML viewer with filters (with cache buster)
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
                console.log("[GeomPack VTK Filters DEBUG] About to call addDOMWidget");
                console.log("[GeomPack VTK Filters DEBUG] typeof this.addDOMWidget:", typeof this.addDOMWidget);

                const widget = this.addDOMWidget("preview_vtk_filters", "MESH_PREVIEW_VTK_FILTERS", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                console.log("[GeomPack VTK Filters DEBUG] Widget created:", widget);
                console.log("[GeomPack VTK Filters DEBUG] Widget properties:", Object.keys(widget || {}));
                console.log("[GeomPack VTK Filters DEBUG] Widget.id:", widget?.id);
                console.log("[GeomPack VTK Filters DEBUG] this.widgets after addDOMWidget:", this.widgets);

                widget.computeSize = () => [512, 560];  // Increased height for info panel

                // Store iframe and info panel references
                this.meshViewerIframeVTKFilters = iframe;
                this.meshInfoPanelVTKFilters = infoPanel;

                // Track iframe load state
                let iframeLoaded = false;
                iframe.addEventListener('load', () => {
                    console.log("[GeomPack VTK Filters DEBUG] Iframe loaded");
                    iframeLoaded = true;
                });

                // Listen for error messages from iframe
                window.addEventListener('message', (event) => {
                    if (event.data.type === 'MESH_ERROR' && event.data.error) {
                        console.error('[GeomPack VTK Filters] Error from viewer:', event.data.error);
                        if (infoPanel) {
                            infoPanel.innerHTML = `<div style="color: #ff6b6b; padding: 8px;">Error: ${event.data.error}</div>`;
                        }
                    }
                });

                // Set initial node size (increased for info panel)
                this.setSize([512, 560]);
                console.log("[GeomPack VTK Filters DEBUG] Node size set to [512, 560]");

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[GeomPack VTK Filters] onExecuted called with message:", message);
                    onExecuted?.apply(this, arguments);

                    // The message IS the UI data (not message.ui)
                    if (message?.mesh_file && message.mesh_file[0]) {
                        const filename = message.mesh_file[0];
                        console.log(`[GeomPack VTK Filters] Loading mesh: ${filename}`);

                        // Update mesh info panel with metadata
                        const vertices = message.vertex_count?.[0] || 'N/A';
                        const faces = message.face_count?.[0] || 'N/A';
                        const boundsMin = message.bounds_min?.[0] || [];
                        const boundsMax = message.bounds_max?.[0] || [];
                        const extents = message.extents?.[0] || [];

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

                        // Add optional fields if available
                        if (message.is_watertight !== undefined) {
                            const watertight = message.is_watertight[0] ? 'Yes' : 'No';
                            const color = message.is_watertight[0] ? '#6c6' : '#c66';
                            infoHTML += `
                                <span style="color: #888;">Watertight:</span>
                                <span style="color: ${color};">${watertight}</span>
                            `;
                        }

                        if (message.field_names && message.field_names[0]?.length > 0) {
                            const fields = message.field_names[0].join(', ');
                            infoHTML += `
                                <span style="color: #888;">Fields:</span>
                                <span style="font-size: 10px;">${fields}</span>
                            `;
                        }

                        infoHTML += '</div>';

                        infoPanel.innerHTML = infoHTML;

                        // ComfyUI serves output files via /view API endpoint
                        const filepath = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;

                        // Function to send message
                        const sendMessage = () => {
                            if (iframe.contentWindow) {
                                console.log(`[GeomPack VTK Filters] Sending postMessage to iframe: ${filepath}`);
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_MESH",
                                    filepath: filepath,
                                    timestamp: Date.now()
                                }, "*");
                            } else {
                                console.error("[GeomPack VTK Filters] Iframe contentWindow not available");
                            }
                        };

                        // Send message after iframe is loaded
                        if (iframeLoaded) {
                            console.log("[GeomPack VTK Filters DEBUG] Iframe already loaded, sending immediately");
                            sendMessage();
                        } else {
                            console.log("[GeomPack VTK Filters DEBUG] Waiting for iframe to load...");
                            setTimeout(sendMessage, 500);
                        }
                    } else {
                        console.log("[GeomPack VTK Filters] No mesh_file in message data. Keys:", Object.keys(message || {}));
                    }
                };

                return r;
            };
        }
    }
});

console.log("[GeomPack] VTK.js mesh preview with filters extension registered");
