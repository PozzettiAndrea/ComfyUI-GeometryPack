/**
 * ComfyUI GeomPack - VTK.js Bounding Box Preview Widget
 * Displays meshes with bounding boxes using VTK.js
 */

import { app } from "../../../scripts/app.js";

console.log("[GeomPack] Loading VTK.js bounding box preview extension...");

app.registerExtension({
    name: "geompack.bboxpreview.vtk",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackPreviewBoundingBoxesVTK") {
            console.log("[GeomPack] Registering Preview Bounding Boxes (VTK) node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log("[GeomPack BBox VTK] Node created, adding widget");

                // Create container for viewer + info panel
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#2a2a2a";
                container.style.overflow = "hidden";

                // Create iframe for VTK.js viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "0";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";

                // Point to VTK.js HTML viewer (with cache buster)
                iframe.src = "/extensions/ComfyUI-GeometryPack/viewer_vtk.html?v=" + Date.now();

                // Create bbox info panel
                const infoPanel = document.createElement("div");
                infoPanel.style.backgroundColor = "#1a1a1a";
                infoPanel.style.borderTop = "1px solid #444";
                infoPanel.style.padding = "6px 12px";
                infoPanel.style.fontSize = "10px";
                infoPanel.style.fontFamily = "monospace";
                infoPanel.style.color = "#ccc";
                infoPanel.style.lineHeight = "1.3";
                infoPanel.style.flexShrink = "0";
                infoPanel.style.overflow = "hidden";
                infoPanel.innerHTML = '<span style="color: #888;">Bounding box preview will appear here after execution</span>';

                // Add iframe and info panel to container
                container.appendChild(iframe);
                container.appendChild(infoPanel);

                // Add widget with required options
                const widget = this.addDOMWidget("preview_bbox_vtk", "BBOX_PREVIEW_VTK", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                widget.computeSize = () => [512, 640];  // Increased height for viewer + info panel

                // Store iframe and info panel references
                this.bboxViewerIframeVTK = iframe;
                this.bboxInfoPanelVTK = infoPanel;

                // Track iframe load state
                let iframeLoaded = false;
                iframe.addEventListener('load', () => {
                    console.log("[GeomPack BBox VTK] Iframe loaded");
                    iframeLoaded = true;
                });

                // Listen for messages from iframe
                window.addEventListener('message', async (event) => {
                    // Handle screenshot messages
                    if (event.data.type === 'SCREENSHOT' && event.data.image) {
                        console.log('[GeomPack BBox VTK] Received screenshot from iframe');

                        try {
                            // Convert base64 data URL to blob
                            const base64Data = event.data.image.split(',')[1];
                            const byteString = atob(base64Data);
                            const arrayBuffer = new ArrayBuffer(byteString.length);
                            const uint8Array = new Uint8Array(arrayBuffer);

                            for (let i = 0; i < byteString.length; i++) {
                                uint8Array[i] = byteString.charCodeAt(i);
                            }

                            const blob = new Blob([uint8Array], { type: 'image/png' });

                            // Generate filename with timestamp
                            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                            const filename = `vtk-bbox-screenshot-${timestamp}.png`;

                            // Create FormData for upload
                            const formData = new FormData();
                            formData.append('image', blob, filename);
                            formData.append('type', 'output');  // Save to output directory
                            formData.append('subfolder', '');   // Root of output folder

                            // Upload to ComfyUI backend
                            console.log('[GeomPack BBox VTK] Uploading screenshot to server...');
                            const response = await fetch('/upload/image', {
                                method: 'POST',
                                body: formData
                            });

                            if (response.ok) {
                                const result = await response.json();
                                console.log('[GeomPack BBox VTK] Screenshot saved to outputs folder:', result.name);
                            } else {
                                throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
                            }

                        } catch (error) {
                            console.error('[GeomPack BBox VTK] Error saving screenshot:', error);
                        }
                    }
                    // Handle error messages from iframe
                    else if (event.data.type === 'MESH_ERROR' && event.data.error) {
                        console.error('[GeomPack BBox VTK] Error from viewer:', event.data.error);
                        if (infoPanel) {
                            infoPanel.innerHTML = `<div style="color: #ff6b6b; padding: 8px;">Error: ${event.data.error}</div>`;
                        }
                    }
                });

                // Set initial node size (increased for info panel)
                this.setSize([512, 640]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[GeomPack BBox VTK] onExecuted called with message:", message);
                    onExecuted?.apply(this, arguments);

                    // The message IS the UI data (not message.ui)
                    if (message?.mesh_file && message.mesh_file[0]) {
                        const filename = message.mesh_file[0];
                        console.log(`[GeomPack BBox VTK] Loading mesh with bboxes: ${filename}`);

                        // Update bbox info panel with metadata
                        const vertices = message.vertex_count?.[0] || 'N/A';
                        const faces = message.face_count?.[0] || 'N/A';
                        const numBboxes = message.num_bboxes?.[0] || 'N/A';
                        const lineWidth = message.line_width?.[0] || 2.0;
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

                        // Build info HTML with bbox count highlighted
                        let infoHTML = `
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 8px;">
                                <span style="color: #888;">Bounding Boxes:</span>
                                <span style="color: #6cf; font-weight: bold;">${numBboxes}</span>

                                <span style="color: #888;">Vertices:</span>
                                <span>${vertices.toLocaleString()}</span>

                                <span style="color: #888;">Faces:</span>
                                <span>${faces.toLocaleString()}</span>

                                <span style="color: #888;">Bounds:</span>
                                <span style="font-size: 9px;">${boundsStr}</span>

                                <span style="color: #888;">Extents:</span>
                                <span>${extentsStr}</span>
                            </div>
                        `;

                        infoPanel.innerHTML = infoHTML;

                        // ComfyUI serves output files via /view API endpoint
                        const filepath = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;

                        // Function to send message
                        const sendMessage = () => {
                            if (iframe.contentWindow) {
                                console.log(`[GeomPack BBox VTK] Sending postMessage to iframe: ${filepath}, lineWidth: ${lineWidth}`);
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_MESH",
                                    filepath: filepath,
                                    lineWidth: lineWidth,
                                    timestamp: Date.now()
                                }, "*");
                            } else {
                                console.error("[GeomPack BBox VTK] Iframe contentWindow not available");
                            }
                        };

                        // Send message after iframe is loaded
                        if (iframeLoaded) {
                            console.log("[GeomPack BBox VTK] Iframe already loaded, sending immediately");
                            sendMessage();
                        } else {
                            console.log("[GeomPack BBox VTK] Waiting for iframe to load...");
                            setTimeout(sendMessage, 500);
                        }
                    } else {
                        console.log("[GeomPack BBox VTK] No mesh_file in message data. Keys:", Object.keys(message || {}));
                    }
                };

                return r;
            };
        }
    }
});

console.log("[GeomPack] VTK.js bounding box preview extension registered");
