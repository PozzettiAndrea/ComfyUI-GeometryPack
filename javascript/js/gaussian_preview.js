/**
 * ComfyUI GeomPack - Gaussian Splat Preview Widget
 * Interactive 3D Gaussian Splatting viewer using gsplat.js
 */

import { app } from "../../../scripts/app.js";

// Auto-detect extension folder name (handles ComfyUI-GeometryPack or comfyui-geometrypack)
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-GeometryPack";
})();

console.log("[GeomPack Gaussian] Loading extension...");

app.registerExtension({
    name: "geompack.gaussianpreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackPreviewGaussian") {
            console.log("[GeomPack Gaussian] Registering Preview Gaussian node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create container for viewer + info panel
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#1a1a1a";
                container.style.overflow = "hidden";

                // Create iframe for gsplat.js viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "0";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#1a1a1a";

                // Point to gsplat.js HTML viewer (with cache buster)
                iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_gaussian.html?v=` + Date.now();

                // Create info panel
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
                infoPanel.innerHTML = '<span style="color: #888;">Gaussian splat info will appear here after execution</span>';

                // Add iframe and info panel to container
                container.appendChild(iframe);
                container.appendChild(infoPanel);

                // Add widget with required options
                const widget = this.addDOMWidget("preview_gaussian", "GAUSSIAN_PREVIEW", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                // Store reference to node for dynamic resizing
                const node = this;
                let currentNodeSize = [512, 580];

                widget.computeSize = () => currentNodeSize;

                // Store references
                this.gaussianViewerIframe = iframe;
                this.gaussianInfoPanel = infoPanel;

                // Function to resize node dynamically
                this.resizeToAspectRatio = function(imageWidth, imageHeight) {
                    const aspectRatio = imageWidth / imageHeight;
                    const nodeWidth = 512;
                    const viewerHeight = Math.round(nodeWidth / aspectRatio);
                    const nodeHeight = viewerHeight + 60;  // Add space for info panel

                    currentNodeSize = [nodeWidth, nodeHeight];
                    node.setSize(currentNodeSize);
                    node.setDirtyCanvas(true, true);
                    app.graph.setDirtyCanvas(true, true);

                    console.log("[GeomPack Gaussian] Resized node to:", nodeWidth, "x", nodeHeight, "(aspect ratio:", aspectRatio.toFixed(2), ")");
                };

                // Track iframe load state
                let iframeLoaded = false;
                iframe.addEventListener('load', () => {
                    iframeLoaded = true;
                });

                // Listen for messages from iframe
                window.addEventListener('message', async (event) => {
                    // Handle screenshot messages
                    if (event.data.type === 'SCREENSHOT' && event.data.image) {
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
                            const filename = `gaussian-screenshot-${timestamp}.png`;

                            // Create FormData for upload
                            const formData = new FormData();
                            formData.append('image', blob, filename);
                            formData.append('type', 'output');
                            formData.append('subfolder', '');

                            // Upload to ComfyUI backend
                            const response = await fetch('/upload/image', {
                                method: 'POST',
                                body: formData
                            });

                            if (response.ok) {
                                const result = await response.json();
                                console.log('[GeomPack Gaussian] Screenshot saved:', result.name);
                            } else {
                                throw new Error(`Upload failed: ${response.status}`);
                            }

                        } catch (error) {
                            console.error('[GeomPack Gaussian] Error saving screenshot:', error);
                        }
                    }
                    // Handle copy image to clipboard messages
                    else if (event.data.type === 'COPY_IMAGE' && event.data.success) {
                        console.log('[GeomPack Gaussian] Image copied to clipboard successfully');
                    }
                    else if (event.data.type === 'COPY_IMAGE' && !event.data.success) {
                        console.error('[GeomPack Gaussian] Failed to copy image to clipboard:', event.data.error);
                    }
                    // Handle error messages from iframe
                    else if (event.data.type === 'MESH_ERROR' && event.data.error) {
                        console.error('[GeomPack Gaussian] Error from viewer:', event.data.error);
                        if (infoPanel) {
                            infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error: ${event.data.error}</div>`;
                        }
                    }
                });

                // Set initial node size
                this.setSize([512, 580]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[GeomPack Gaussian] onExecuted called with:", message);
                    onExecuted?.apply(this, arguments);

                    // Check for errors
                    if (message?.error && message.error[0]) {
                        infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error: ${message.error[0]}</div>`;
                        return;
                    }

                    // The message IS the UI data (not message.ui)
                    if (message?.ply_file && message.ply_file[0]) {
                        const filename = message.ply_file[0];
                        const displayName = message.filename?.[0] || filename;
                        const fileSizeMb = message.file_size_mb?.[0] || 'N/A';

                        // Extract camera parameters if provided
                        const extrinsics = message.extrinsics?.[0] || null;
                        const intrinsics = message.intrinsics?.[0] || null;

                        // Resize node to match image aspect ratio from intrinsics
                        if (intrinsics && intrinsics[0] && intrinsics[1]) {
                            const imageWidth = intrinsics[0][2] * 2;   // cx * 2
                            const imageHeight = intrinsics[1][2] * 2;  // cy * 2
                            this.resizeToAspectRatio(imageWidth, imageHeight);
                        }

                        // Update info panel
                        infoPanel.innerHTML = `
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 8px;">
                                <span style="color: #888;">File:</span>
                                <span style="color: #6cc;">${displayName}</span>
                                <span style="color: #888;">Size:</span>
                                <span>${fileSizeMb} MB</span>
                            </div>
                        `;

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

                        // Send URL to iframe — gsplat.js fetches directly with streaming
                        const sendUrl = () => {
                            if (!iframe.contentWindow) {
                                console.error("[GeomPack Gaussian] Iframe contentWindow not available");
                                return;
                            }
                            iframe.contentWindow.postMessage({
                                type: "LOAD_MESH_URL",
                                url: filepath,
                                filename: filename,
                                extrinsics: extrinsics,
                                intrinsics: intrinsics,
                                timestamp: Date.now()
                            }, "*");
                        };

                        if (iframeLoaded) {
                            sendUrl();
                        } else {
                            setTimeout(sendUrl, 500);
                        }
                    }
                };

                return r;
            };
        }
    }
});
