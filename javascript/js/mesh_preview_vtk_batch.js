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
    name: "geometrypack.meshpreview.vtk.batch",

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

                // Dropdown to jump directly to any mesh in the batch (client-side)
                const dropdown = document.createElement("select");
                dropdown.style.cssText = "background:#333;color:#ccc;border:1px solid #555;" +
                    "border-radius:3px;font:11px monospace;padding:2px 6px;cursor:pointer;";
                dropdown.innerHTML = '<option value="0">1 / 1</option>';

                // Assemble navigation bar
                navBar.appendChild(prevButton);
                navBar.appendChild(indexLabel);
                navBar.appendChild(dropdown);
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

                // Navigation is CLIENT-SIDE: arrows / dropdown / the index widget
                // switch the shown mesh by posting LOAD_MESH to THIS node's iframe.
                // They never call app.queuePrompt(), so navigating never re-runs the
                // graph or reloads any other node. (showIndex is defined below.)
                let suppressIndexCallback = false;
                if (indexWidget) {
                    const originalCallback = indexWidget.callback;
                    indexWidget.callback = function(value) {
                        const result = originalCallback?.apply(this, arguments);
                        // showIndex() sets the widget value itself -> ignore that echo
                        if (!suppressIndexCallback) showIndex(Number(value));
                        return result;
                    };
                }

                prevButton.addEventListener("click", () => showIndex(currentIndex - 1));
                nextButton.addEventListener("click", () => showIndex(currentIndex + 1));
                dropdown.addEventListener("change", () => showIndex(Number(dropdown.value)));

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

                // ---- client-side batch navigation state + helpers ----
                // Filled by onExecuted with everything needed to switch meshes with
                // no server round-trip: { files:[names], viewerType, mode, meta:[{
                // vertex_count, face_count, bounds_min, bounds_max, extents,
                // is_watertight, field_names, has_texture, has_vertex_colors,
                // visual_kind }, ...] }.
                let batchData = null;

                const postLoad = (i) => {
                    if (!iframe.contentWindow || !batchData) return;
                    const filepath = `/view?filename=${encodeURIComponent(batchData.files[i])}&type=output&subfolder=`;
                    iframe.contentWindow.postMessage({
                        type: "LOAD_MESH",
                        filepath,
                        timestamp: Date.now(),
                        showEdges: viewerState.show_edges,
                        cameraState: viewerState.camera_state,
                        selectedField: viewerState.selected_field,
                        selectedChannel: viewerState.selected_channel,
                        selectedColormap: viewerState.selected_colormap,
                    }, "*");
                };

                const renderInfo = (i) => {
                    const m = batchData?.meta?.[i];
                    if (!m) return;
                    const num = (v) => (typeof v === "number" ? v.toLocaleString() : "N/A");
                    const esc = (s) => String(s ?? "").replace(/[&<>]/g,
                        c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
                    const name = batchData.names?.[i] || `mesh ${i + 1}`;
                    const boundsStr = (Array.isArray(m.bounds_min) && Array.isArray(m.bounds_max)
                        && m.bounds_min.length === 3 && m.bounds_max.length === 3)
                        ? `[${m.bounds_min.map(v => v.toFixed(2)).join(", ")}] to [${m.bounds_max.map(v => v.toFixed(2)).join(", ")}]`
                        : "N/A";
                    const extentsStr = (Array.isArray(m.extents) && m.extents.length === 3)
                        ? m.extents.map(v => v.toFixed(2)).join(" x ") : "N/A";
                    const modeLabel = batchData.mode.charAt(0).toUpperCase() + batchData.mode.slice(1);
                    const modeColor = batchData.viewerType === "texture" ? "#c8c" : "#6cc";
                    let html = `
                        <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 8px;">
                            <span style="color: #888;">Name:</span>
                            <span style="color: #ddd; font-weight: bold; word-break: break-all;">${esc(name)}</span>
                            <span style="color: #888;">Batch:</span>
                            <span style="color: #8c8; font-weight: bold;">${i + 1} / ${currentBatchSize}</span>
                            <span style="color: #888;">Mode:</span>
                            <span style="color: ${modeColor}; font-weight: bold;">${modeLabel}</span>
                            <span style="color: #888;">Vertices:</span><span>${num(m.vertex_count)}</span>
                            <span style="color: #888;">Faces:</span><span>${num(m.face_count)}</span>
                            <span style="color: #888;">Bounds:</span>
                            <span style="font-size: 9px;">${boundsStr}</span>
                            <span style="color: #888;">Extents:</span><span>${extentsStr}</span>
                            <span style="color: #888;">Watertight:</span>
                            <span style="color: ${m.is_watertight ? "#6c6" : "#c66"};">${m.is_watertight ? "Yes" : "No"}</span>`;
                    if (batchData.viewerType === "texture") {
                        html += `
                            <span style="color: #888;">Visual Kind:</span><span>${m.visual_kind ?? "none"}</span>
                            <span style="color: #888;">Textures:</span>
                            <span style="color: ${m.has_texture ? "#c8c" : "#888"};">${m.has_texture ? "Yes" : "No"}</span>
                            <span style="color: #888;">Vertex Colors:</span><span>${m.has_vertex_colors ? "Yes" : "No"}</span>`;
                    } else {
                        const hasFields = Array.isArray(m.field_names) && m.field_names.length > 0;
                        html += `<span style="color: #888;">Fields:</span>` +
                            `<span style="font-size: 9px; color: ${hasFields ? "#6cc" : "#888"};">` +
                            `${hasFields ? m.field_names.join(", ") : "None"}</span>`;
                    }
                    html += "</div>";
                    infoPanel.innerHTML = html;
                };

                // Switch to mesh i entirely client-side (no app.queuePrompt()).
                const showIndex = (i) => {
                    if (!batchData || !batchData.files.length) return;
                    i = Math.max(0, Math.min(i, batchData.files.length - 1));
                    currentIndex = i;
                    suppressIndexCallback = true;
                    if (indexWidget) indexWidget.value = i;   // persists via serialization
                    suppressIndexCallback = false;
                    dropdown.value = String(i);
                    indexLabel.textContent = `${i + 1} / ${currentBatchSize}`;
                    updateNavigationButtons();
                    renderInfo(i);
                    postLoad(i);
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

                // Handle execution: store the WHOLE batch, then show the start index.
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    const files = message?.mesh_files?.[0];
                    if (!files || !files.length) return;

                    const col = (key) => message[key]?.[0] || [];   // a per-index array
                    batchData = {
                        files,
                        names: message.mesh_names?.[0] || [],   // e.g. ["apple.ply", ...]
                        viewerType: message.viewer_type?.[0] || "fields",
                        mode: message.mode?.[0] || "fields",
                        meta: files.map((_, k) => ({
                            vertex_count: col("vertex_counts")[k],
                            face_count: col("face_counts")[k],
                            bounds_min: col("bounds_mins")[k],
                            bounds_max: col("bounds_maxs")[k],
                            extents: col("extents_all")[k],
                            is_watertight: col("is_watertights")[k],
                            field_names: col("field_names_all")[k],
                            has_texture: col("has_textures")[k],
                            has_vertex_colors: col("has_vertex_colors_all")[k],
                            visual_kind: col("visual_kinds")[k],
                        })),
                    };

                    currentBatchSize = message.batch_size?.[0] || files.length;
                    if (indexWidget) indexWidget.options.max = currentBatchSize - 1;

                    // (Re)populate the dropdown: one option per mesh, labelled by its
                    // source name (e.g. "apple.ply"), falling back to its position.
                    // Built with real <option> elements + textContent so a filename
                    // can never inject markup.
                    dropdown.replaceChildren(...files.map((_, k) => {
                        const opt = document.createElement("option");
                        opt.value = String(k);
                        opt.textContent = batchData.names[k] || `${k + 1} / ${files.length}`;
                        return opt;
                    }));

                    let startIndex = message.current_index?.[0] || 0;
                    startIndex = Math.max(0, Math.min(startIndex, files.length - 1));

                    // The iframe only reloads when the GLOBAL viewer type actually
                    // changed (fields <-> texture), never during navigation. Once the
                    // right viewer is loaded, show the start index client-side.
                    const viewerUrl = batchData.viewerType === "texture"
                        ? `/extensions/${EXTENSION_FOLDER}/viewer_vtk_textured.html`
                        : `/extensions/${EXTENSION_FOLDER}/viewer_vtk.html`;
                    if (batchData.viewerType !== currentViewerType) {
                        currentViewerType = batchData.viewerType;
                        iframeLoaded = false;
                        iframe.addEventListener("load", () => {
                            iframeLoaded = true;
                            showIndex(startIndex);
                        }, { once: true });
                        iframe.src = viewerUrl + "?v=" + Date.now();
                    } else if (iframeLoaded) {
                        showIndex(startIndex);
                    } else {
                        iframe.addEventListener("load", () => {
                            iframeLoaded = true;
                            showIndex(startIndex);
                        }, { once: true });
                    }
                };

                return r;
            };
        }
    }
});
