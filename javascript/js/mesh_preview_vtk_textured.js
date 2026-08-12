/**
 * ComfyUI GeomPack - VTK.js Textured Mesh Preview Widget
 * Supports GLB/GLTF with textures and PBR materials via VTK.js GLTFImporter
 */

import { app } from "../../../scripts/app.js";

// Auto-detect extension folder name (handles ComfyUI-GeometryPack or comfyui-geometrypack)
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-GeometryPack";
})();

console.log("[GeomPack] Loading VTK.js textured mesh preview extension...");

app.registerExtension({
    name: "geompack.meshpreview.vtk.textured",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackPreviewMeshVTKWithTexture") {
            console.log("[GeomPack] Registering Preview Mesh (VTK with Textures) node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create container for viewer + info panel
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#2a2a2a";
                container.style.overflow = "hidden";

                // Create iframe for VTK.js textured viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "0";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";

                // Point to VTK.js textured HTML viewer (with cache buster)
                // Use unified v2 viewer with modular architecture
                iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_vtk_textured.html?v=` + Date.now();

                // Create mesh info panel
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
                infoPanel.innerHTML = '<span style="color: #888;">Textured mesh info will appear here after execution</span>';

                // Add iframe and info panel to container
                container.appendChild(iframe);
                container.appendChild(infoPanel);

                // Add widget with required options
                const widget = this.addDOMWidget("preview_vtk_textured", "MESH_PREVIEW_VTK_TEXTURED", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                widget.computeSize = () => [512, 640];  // Increased height for viewer + info panel

                // Store iframe and info panel references
                this.meshViewerIframeVTKTextured = iframe;
                this.meshInfoPanelVTKTextured = infoPanel;

                // Track iframe load state
                let iframeLoaded = false;
                iframe.addEventListener('load', () => {
                    iframeLoaded = true;
                });

                // Listen for messages from iframe
                window.addEventListener('message', async (event) => {
                    // Handle screenshot messages
                    if (event.data.type === 'SCREENSHOT' && event.data.image) {
                        console.log('[GeomPack VTK Textured] Received screenshot from iframe');

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
                            const filename = `vtk-textured-screenshot-${timestamp}.png`;

                            // Create FormData for upload
                            const formData = new FormData();
                            formData.append('image', blob, filename);
                            formData.append('type', 'output');  // Save to output directory
                            formData.append('subfolder', '');   // Root of output folder

                            // Upload to ComfyUI backend
                            console.log('[GeomPack VTK Textured] Uploading screenshot to server...');
                            const response = await fetch('/upload/image', {
                                method: 'POST',
                                body: formData
                            });

                            if (response.ok) {
                                const result = await response.json();
                                console.log('[GeomPack VTK Textured] Screenshot saved to outputs folder:', result.name);
                            } else {
                                throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
                            }

                        } catch (error) {
                            console.error('[GeomPack VTK Textured] Error saving screenshot:', error);
                        }
                    }
                    // Handle error messages from iframe
                    else if (event.data.type === 'MESH_ERROR' && event.data.error) {
                        console.error('[GeomPack VTK Textured] Error from viewer:', event.data.error);
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
                    onExecuted?.apply(this, arguments);

                    // The message IS the UI data (not message.ui)
                    if (message?.mesh_file && message.mesh_file[0]) {
                        const filename = message.mesh_file[0];

                        // Update mesh info panel with metadata
                        const vertices = message.vertex_count?.[0] || 'N/A';
                        const faces = message.face_count?.[0] || 'N/A';
                        const boundsMin = message.bounds_min?.[0] || [];
                        const boundsMax = message.bounds_max?.[0] || [];
                        const extents = message.extents?.[0] || [];

                        // Texture/material info
                        const hasTexture = message.has_texture?.[0] || false;
                        const hasVertexColors = message.has_vertex_colors?.[0] || false;
                        const hasMaterial = message.has_material?.[0] || false;
                        const visualKind = message.visual_kind?.[0] || 'none';

                        // Format bounds
                        let boundsStr = 'N/A';
                        if (boundsMin.length === 3 && boundsMax.length === 3) {
                            boundsStr = `[${boundsMin.map(v => v.toFixed(2)).join(', ')}] to [${boundsMax.map(v => v.toFixed(2)).join(', ')}]`;
                        }

                        // Format extents
                        let extentsStr = 'N/A';
                        if (extents.length === 3) {
                            extentsStr = `${extents.map(v => v.toFixed(2)).join(' x ')}`;
                        }

                        // Build info HTML
                        let infoHTML = `
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 8px;">
                                <span style="color: #888;">Vertices:</span>
                                <span>${vertices.toLocaleString()}</span>

                                <span style="color: #888;">Faces:</span>
                                <span>${faces.toLocaleString()}</span>

                                <span style="color: #888;">Bounds:</span>
                                <span style="font-size: 9px;">${boundsStr}</span>

                                <span style="color: #888;">Extents:</span>
                                <span>${extentsStr}</span>

                                <span style="color: #888;">Visual Type:</span>
                                <span style="color: #4fc3f7;">${visualKind}</span>

                                <span style="color: #888;">Texture:</span>
                                <span style="color: ${hasTexture ? '#6c6' : '#888'};">${hasTexture ? 'Yes' : 'No'}</span>

                                <span style="color: #888;">Vertex Colors:</span>
                                <span style="color: ${hasVertexColors ? '#6c6' : '#888'};">${hasVertexColors ? 'Yes' : 'No'}</span>

                                <span style="color: #888;">Material:</span>
                                <span style="color: ${hasMaterial ? '#6c6' : '#888'};">${hasMaterial ? 'Yes' : 'No'}</span>
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

                        infoHTML += '</div>';

                        infoPanel.innerHTML = infoHTML;

                        // ComfyUI serves output files via /view API endpoint
                        const filepath = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;

                        // Prepare metadata to pass to viewer
                        const metadata = {
                            has_texture: hasTexture,
                            has_vertex_colors: hasVertexColors,
                            has_material: hasMaterial,
                            visual_kind: visualKind
                        };

                        // Function to send message
                        const sendMessage = () => {
                            if (iframe.contentWindow) {
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_MESH",
                                    filepath: filepath,
                                    metadata: metadata,
                                    timestamp: Date.now()
                                }, "*");
                            } else {
                                console.error("[GeomPack VTK Textured] Iframe contentWindow not available");
                            }
                        };

                        // Send message after iframe is loaded
                        if (iframeLoaded) {
                            sendMessage();
                        } else {
                            setTimeout(sendMessage, 500);
                        }
                    } else {
                        console.log("[GeomPack VTK Textured] No mesh_file in message data. Keys:", Object.keys(message || {}));
                    }
                };

                return r;
            };
        }
    }
});

console.log("[GeomPack] VTK.js textured mesh preview extension registered");
