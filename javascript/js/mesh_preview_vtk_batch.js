/**
 * ComfyUI GeomPack - VTK.js Batch Mesh Preview Widget
 * Scientific visualization with VTK.js and batch navigation
 */

import { app } from "../../../scripts/app.js";

// Auto-detect extension folder name (handles ComfyUI-GeometryPack or comfyui-geometrypack)
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-GeometryPack";
})();

app.registerExtension({
    name: "geompack.meshpreview.vtk.batch",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackPreviewMeshVTKBatch") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Viewer state persisted via DOM widget serialization
                const viewerState = { show_edges: false, camera_state: "", selected_field: "", selected_channel: "magnitude", selected_colormap: "erdc_rainbow_bright" };

                // Create container for viewer + navigation + info panel
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
                // Note: viewer will be dynamically switched based on mode in onExecuted
                // Use unified v2 viewer with modular architecture
                iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_vtk.html?v=` + Date.now();

                // Track current viewer type to avoid unnecessary reloads
                let currentViewerType = "fields";

                // Create navigation bar for batch controls
                const navBar = document.createElement("div");
                navBar.style.backgroundColor = "#1a1a1a";
                navBar.style.borderTop = "1px solid #444";
                navBar.style.padding = "8px 12px";
                navBar.style.display = "flex";
                navBar.style.alignItems = "center";
                navBar.style.justifyContent = "center";
                navBar.style.gap = "12px";
                navBar.style.fontSize = "12px";
                navBar.style.color = "#ccc";
                navBar.style.flexShrink = "0";

                // Previous button
                const prevButton = document.createElement("button");
                prevButton.textContent = "< Previous";
                prevButton.style.padding = "4px 12px";
                prevButton.style.cursor = "pointer";
                prevButton.style.backgroundColor = "#333";
                prevButton.style.color = "#ccc";
                prevButton.style.border = "1px solid #555";
                prevButton.style.borderRadius = "3px";
                prevButton.style.fontSize = "11px";

                // Index display
                const indexLabel = document.createElement("span");
                indexLabel.textContent = "1 / 1";
                indexLabel.style.minWidth = "60px";
                indexLabel.style.textAlign = "center";
                indexLabel.style.fontFamily = "monospace";
                indexLabel.style.fontWeight = "bold";

                // Next button
                const nextButton = document.createElement("button");
                nextButton.textContent = "Next >";
                nextButton.style.padding = "4px 12px";
                nextButton.style.cursor = "pointer";
                nextButton.style.backgroundColor = "#333";
                nextButton.style.color = "#ccc";
                nextButton.style.border = "1px solid #555";
                nextButton.style.borderRadius = "3px";
                nextButton.style.fontSize = "11px";

                // Assemble navigation bar
                navBar.appendChild(prevButton);
                navBar.appendChild(indexLabel);
                navBar.appendChild(nextButton);

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
                infoPanel.innerHTML = '<span style="color: #888;">Mesh info will appear here after execution</span>';

                // Add iframe, navigation bar, and info panel to container
                container.appendChild(iframe);
                container.appendChild(navBar);
                container.appendChild(infoPanel);

                // Add widget with required options

                const widget = this.addDOMWidget("preview_vtk_batch", "MESH_PREVIEW_VTK_BATCH", container, {
                    getValue() { return JSON.stringify(viewerState); },
                    setValue(v) {
                        try { Object.assign(viewerState, JSON.parse(v)); } catch(e) {}
                    }
                });


                widget.computeSize = () => [512, 680];  // Increased height for viewer + navigation + info panel

                // Store iframe and info panel references
                this.meshViewerIframeVTKBatch = iframe;
                this.meshInfoPanelVTKBatch = infoPanel;
                this.meshNavBarVTKBatch = navBar;

                // Bidirectional sync: viewer → node widgets (viewerState + real widgets)
                const node = this;
                window.addEventListener('message', (event) => {
                    // Without this check, every open GeomPackPreviewMeshVTKBatch
                    // instance's listener fires for every iframe's messages, not
                    // just its own -- e.g. toggling "show edges" in one viewer
                    // updates every other open viewer's state too.
                    if (event.source !== iframe.contentWindow) return;
                    if (event.data.type === 'WIDGET_UPDATE') {
                        const { widget: name, value } = event.data;
                        if (name in viewerState) viewerState[name] = value;
                        const w = node.widgets?.find(w => w.name === name);
                        if (w) w.value = value;
                    }
                });

                // Track iframe load state
                let iframeLoaded = false;
                iframe.addEventListener('load', () => {
                    iframeLoaded = true;
                });

                // Find the index widget (created by ComfyUI from INPUT_TYPES)
                const indexWidget = this.widgets.find(w => w.name === "index");

                // Track batch state
                let currentBatchSize = 1;
                let currentIndex = 0;

                // Add callback to auto-execute when index changes manually
                if (indexWidget) {
                    const originalCallback = indexWidget.callback;
                    indexWidget.callback = function(value) {
                        // Call original callback if it exists
                        const result = originalCallback?.apply(this, arguments);

                        // Update tracked index
                        currentIndex = value;

                        // Trigger re-execution (same as button clicks)
                        app.queuePrompt();

                        return result;
                    };
                }

                // Button click handlers
                prevButton.addEventListener("click", () => {
                    if (indexWidget && currentIndex > 0) {
                        indexWidget.value = currentIndex - 1;
                        // Trigger node queue/re-execution
                        app.queuePrompt();
                    }
                });

                nextButton.addEventListener("click", () => {
                    if (indexWidget && currentIndex < currentBatchSize - 1) {
                        indexWidget.value = currentIndex + 1;
                        // Trigger node queue/re-execution
                        app.queuePrompt();
                    }
                });

                // Update button states
                const updateNavigationButtons = () => {
                    prevButton.disabled = currentIndex === 0;
                    nextButton.disabled = currentIndex >= currentBatchSize - 1;

                    // Style disabled buttons
                    if (prevButton.disabled) {
                        prevButton.style.opacity = "0.4";
                        prevButton.style.cursor = "not-allowed";
                    } else {
                        prevButton.style.opacity = "1";
                        prevButton.style.cursor = "pointer";
                    }

                    if (nextButton.disabled) {
                        nextButton.style.opacity = "0.4";
                        nextButton.style.cursor = "not-allowed";
                    } else {
                        nextButton.style.opacity = "1";
                        nextButton.style.cursor = "pointer";
                    }
                };

                // Listen for messages from iframe
                window.addEventListener('message', async (event) => {
                    if (event.source !== iframe.contentWindow) return;
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
                            const filename = `vtk-screenshot-${timestamp}.png`;

                            // Create FormData for upload
                            const formData = new FormData();
                            formData.append('image', blob, filename);
                            formData.append('type', 'output');  // Save to output directory
                            formData.append('subfolder', '');   // Root of output folder

                            // Upload to ComfyUI backend
                            const response = await fetch('/upload/image', {
                                method: 'POST',
                                body: formData
                            });

                            if (response.ok) {
                                const result = await response.json();
                            } else {
                                throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
                            }

                        } catch (error) {
                            console.error('[GeomPack VTK Batch] Error saving screenshot:', error);
                        }
                    }
                    // Handle error messages from iframe
                    else if (event.data.type === 'MESH_ERROR' && event.data.error) {
                        console.error('[GeomPack VTK Batch] Error from viewer:', event.data.error);
                        if (infoPanel) {
                            infoPanel.innerHTML = `<div style="color: #ff6b6b; padding: 8px;">Error: ${event.data.error}</div>`;
                        }
                    }
                });

                // Set initial node size (increased for info panel + navigation)
                this.setSize([512, 680]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    // The message IS the UI data (not message.ui)
                    if (message?.mesh_file && message.mesh_file[0]) {
                        const filename = message.mesh_file[0];
                        const viewerType = message.viewer_type?.[0] || "fields";
                        const mode = message.mode?.[0] || "fields";

                        // Get batch metadata
                        currentBatchSize = message.batch_size?.[0] || 1;
                        currentIndex = message.current_index?.[0] || 0;

                        // Update navigation controls
                        indexLabel.textContent = `${currentIndex + 1} / ${currentBatchSize}`;
                        updateNavigationButtons();

                        // Dynamically update index widget max based on actual batch size
                        if (indexWidget) {
                            indexWidget.options.max = currentBatchSize - 1;

                            // Clamp current value if out of range
                            if (indexWidget.value >= currentBatchSize) {
                                indexWidget.value = currentBatchSize - 1;
                            }
                        }

                        // Determine which viewer HTML to use (unified v2 viewers)
                        const viewerUrl = viewerType === "texture"
                            ? `/extensions/${EXTENSION_FOLDER}/viewer_vtk_textured.html`
                            : `/extensions/${EXTENSION_FOLDER}/viewer_vtk.html`;

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
                            extentsStr = `${extents.map(v => v.toFixed(2)).join(' x ')}`;
                        }

                        // Build info HTML
                        const modeLabel = mode.charAt(0).toUpperCase() + mode.slice(1);
                        const modeColor = mode === "texture" ? '#c8c' : '#6cc';

                        let infoHTML = `
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 8px;">
                                <span style="color: #888;">Batch:</span>
                                <span style="color: #8c8; font-weight: bold;">${currentIndex + 1} / ${currentBatchSize}</span>

                                <span style="color: #888;">Mode:</span>
                                <span style="color: ${modeColor}; font-weight: bold;">${modeLabel}</span>

                                <span style="color: #888;">Vertices:</span>
                                <span>${vertices.toLocaleString()}</span>

                                <span style="color: #888;">Faces:</span>
                                <span>${faces.toLocaleString()}</span>

                                <span style="color: #888;">Bounds:</span>
                                <span style="font-size: 9px;">${boundsStr}</span>

                                <span style="color: #888;">Extents:</span>
                                <span>${extentsStr}</span>
                        `;

                        // Add watertight status (always shown)
                        if (message.is_watertight !== undefined) {
                            const watertight = message.is_watertight[0] ? 'Yes' : 'No';
                            const color = message.is_watertight[0] ? '#6c6' : '#c66';
                            infoHTML += `
                                <span style="color: #888;">Watertight:</span>
                                <span style="color: ${color};">${watertight}</span>
                            `;
                        }

                        // Add mode-specific info
                        if (viewerType === "texture") {
                            // Texture mode info
                            if (message.visual_kind !== undefined) {
                                const visualKind = message.visual_kind[0] || 'none';
                                infoHTML += `
                                    <span style="color: #888;">Visual Kind:</span>
                                    <span>${visualKind}</span>
                                `;
                            }
                            if (message.has_texture !== undefined) {
                                const hasTexture = message.has_texture[0] ? 'Yes' : 'No';
                                const texColor = message.has_texture[0] ? '#c8c' : '#888';
                                infoHTML += `
                                    <span style="color: #888;">Textures:</span>
                                    <span style="color: ${texColor};">${hasTexture}</span>
                                `;
                            }
                            if (message.has_vertex_colors !== undefined) {
                                const hasColors = message.has_vertex_colors[0] ? 'Yes' : 'No';
                                infoHTML += `
                                    <span style="color: #888;">Vertex Colors:</span>
                                    <span>${hasColors}</span>
                                `;
                            }
                        } else {
                            // Fields mode info
                            if (message.field_names && message.field_names[0]?.length > 0) {
                                const fields = message.field_names[0].join(', ');
                                infoHTML += `
                                    <span style="color: #888;">Fields:</span>
                                    <span style="font-size: 9px; color: #6cc;">${fields}</span>
                                `;
                            } else {
                                infoHTML += `
                                    <span style="color: #888;">Fields:</span>
                                    <span style="color: #888;">None</span>
                                `;
                            }
                        }

                        infoHTML += '</div>';

                        infoPanel.innerHTML = infoHTML;

                        // ComfyUI serves output files via /view API endpoint
                        const filepath = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;

                        // Function to send message
                        const sendMessage = () => {
                            if (iframe.contentWindow) {
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_MESH",
                                    filepath: filepath,
                                    timestamp: Date.now(),
                                    showEdges: viewerState.show_edges,
                                    cameraState: viewerState.camera_state,
                                    selectedField: viewerState.selected_field,
                                    selectedChannel: viewerState.selected_channel,
                                    selectedColormap: viewerState.selected_colormap,
                                }, "*");
                            } else {
                                console.error("[GeomPack VTK Batch] Iframe contentWindow not available");
                            }
                        };

                        // Reload iframe if viewer type changed
                        if (viewerType !== currentViewerType) {
                            currentViewerType = viewerType;
                            iframeLoaded = false;

                            // Set up one-time load listener before changing src
                            const onViewerLoaded = () => {
                                iframeLoaded = true;
                                sendMessage();
                            };
                            iframe.addEventListener('load', onViewerLoaded, { once: true });

                            // Change iframe src to trigger reload
                            iframe.src = viewerUrl + "?v=" + Date.now();
                        } else {
                            // No viewer change needed, send message immediately or after short delay
                            if (iframeLoaded) {
                                sendMessage();
                            } else {
                                setTimeout(sendMessage, 500);
                            }
                        }
                    } else {
                    }
                };

                return r;
            };
        }
    }
});
