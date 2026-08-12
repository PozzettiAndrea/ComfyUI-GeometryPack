/**
 * ComfyUI GeomPack - Multi Mesh Preview Widget
 * Grid viewer for 1-4 meshes with synchronized cameras
 */

import { app } from "../../../scripts/app.js";

// Auto-detect extension folder name
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-GeometryPack";
})();

console.log('[GeomPack Multi JS] Loading mesh_preview_multi.js extension');

app.registerExtension({
    name: "geompack.meshpreview.multi",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackPreviewMeshMulti") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Viewer state persisted via DOM widget serialization
                const viewerState = { layout: "wipe", show_edges: false, camera_state: "", selected_field: "", selected_channel: "magnitude", selected_colormap: "erdc_rainbow_bright" };
                const viewerUrl = () => (viewerState.layout === "overlay" ? "viewer_multi.html" : "viewer_multi_slider.html");

                console.log('[GeomPack Multi JS] Creating PreviewMeshMulti node widget');

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
                iframe.style.minHeight = "450px";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";
                iframe.src = `/extensions/${EXTENSION_FOLDER}/${viewerUrl()}?v=` + Date.now();

                // Layout toggle bar (Wipe = N-1 draggable dividers, fixed order; Overlay = stacked)
                const bar = document.createElement("div");
                bar.style.cssText = "background:#1a1a1a;border-bottom:1px solid #444;padding:4px 8px;display:flex;gap:8px;align-items:center;font:11px monospace;color:#ccc;flex-shrink:0;";
                const layoutSel = document.createElement("select");
                layoutSel.style.cssText = "background:#333;color:#ccc;border:1px solid #555;border-radius:3px;font:11px monospace;padding:2px 6px;";
                layoutSel.innerHTML = '<option value="wipe">Wipe (sliders)</option><option value="overlay">Overlay</option>';
                bar.appendChild(Object.assign(document.createElement("span"), { textContent: "Layout:" }));
                bar.appendChild(layoutSel);

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

                container.appendChild(bar);
                container.appendChild(iframe);
                container.appendChild(infoPanel);

                // Add widget
                const widget = this.addDOMWidget("preview_multi", "MESH_PREVIEW_MULTI", container, {
                    getValue() { return JSON.stringify(viewerState); },
                    setValue(v) {
                        try { Object.assign(viewerState, JSON.parse(v)); } catch(e) {}
                    }
                });

                widget.computeSize = () => [768, 580];

                // Store references
                this.meshViewerIframeMulti = iframe;
                this.meshInfoPanelMulti = infoPanel;

                this.setSize(this.computeSize());

                // Bidirectional sync: viewer → node widgets (viewerState + real widgets)
                const node = this;
                window.addEventListener('message', (event) => {
                    // Without this check, every open viewer instance's listener
                    // fires for every iframe's messages, not just its own.
                    if (event.source !== iframe.contentWindow) return;
                    if (event.data.type === 'WIDGET_UPDATE') {
                        const { widget: name, value } = event.data;
                        if (name in viewerState) viewerState[name] = value;
                        const w = node.widgets?.find(w => w.name === name);
                        if (w) w.value = value;
                    }
                });

                // Track iframe load state + last loaded meshes (so a layout switch re-sends)
                let iframeLoaded = false;
                let lastLoad = null;   // { numMeshes, filepaths }
                const buildAndSend = () => {
                    if (!lastLoad || !iframe.contentWindow) return;
                    let msg;
                    if (viewerState.layout === "overlay") {
                        msg = { type: 'LOAD_MULTI_MESH', numMeshes: lastLoad.numMeshes, meshFiles: lastLoad.filepaths,
                                timestamp: Date.now(), showEdges: viewerState.show_edges, cameraState: viewerState.camera_state,
                                selectedField: viewerState.selected_field, selectedChannel: viewerState.selected_channel,
                                selectedColormap: viewerState.selected_colormap };
                    } else {
                        msg = { type: 'LOAD_MULTI_SLIDER', mesh_files: lastLoad.filepaths, timestamp: Date.now(),
                                show_edges: viewerState.show_edges, camera_state: viewerState.camera_state };
                    }
                    iframe.contentWindow.postMessage(msg, "*");
                };
                iframe.addEventListener('load', () => { iframeLoaded = true; buildAndSend(); });

                // Layout switch: swap the viewer iframe; buildAndSend fires on its load
                layoutSel.value = viewerState.layout;
                layoutSel.addEventListener('change', () => {
                    viewerState.layout = layoutSel.value;
                    iframeLoaded = false;
                    iframe.src = `/extensions/${EXTENSION_FOLDER}/${viewerUrl()}?v=` + Date.now();
                });

                // Set initial node size
                this.setSize([768, 580]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    if (!message?.num_meshes) {
                        return;
                    }

                    const numMeshes = message.num_meshes[0];
                    const meshFiles = message.mesh_files[0];
                    const vertexCounts = message.vertex_counts[0];
                    const faceCounts = message.face_counts[0];
                    const gridCols = message.grid_cols[0];
                    const gridRows = message.grid_rows[0];

                    console.log(`[GeomPack Multi] onExecuted: ${numMeshes} meshes, grid ${gridCols}x${gridRows}`);

                    // Per-mesh info, one column per mesh (matches the single Preview Mesh panel)
                    const wt = message.is_watertight_list?.[0] || [];
                    const avg = message.avg_edge_lengths?.[0] || [];
                    const bnds = message.bounds_list?.[0] || [];
                    const exts = message.extents_list?.[0] || [];
                    const fields = message.field_names_list?.[0] || null;
                    const num = (v) => (v == null ? '—' : Number(v).toLocaleString());
                    const sig = (v) => (v == null ? '—' : Number(v).toLocaleString(undefined, { maximumSignificantDigits: 4 }));
                    const ext = (e) => (e ? e.map((x) => Number(x).toFixed(2)).join(' × ') : '—');
                    const bnd = (b) => (b
                        ? `<span style="font-size:9px;color:#aaa;">[${b[0].map((x) => Number(x).toFixed(1)).join(', ')}]<br>→ [${b[1].map((x) => Number(x).toFixed(1)).join(', ')}]</span>`
                        : '—');

                    let infoHTML = `<div style="display: grid; grid-template-columns: auto repeat(${numMeshes}, 1fr); gap: 2px 12px; font: 11px monospace;">`;
                    infoHTML += `<span style="color: #888;"></span>`;
                    for (let i = 0; i < numMeshes; i++) {
                        infoHTML += `<span style="color: #999; font-weight: bold; border-bottom: 1px solid #333;">Mesh ${i + 1}</span>`;
                    }
                    const row = (label, valFn) => {
                        infoHTML += `<span style="color: #888;">${label}</span>`;
                        for (let i = 0; i < numMeshes; i++) infoHTML += `<span>${valFn(i)}</span>`;
                    };
                    row('Vertices:', (i) => num(vertexCounts[i]));
                    row('Faces:', (i) => num(faceCounts[i]));
                    row('Watertight:', (i) => { const w = wt[i]; return `<span style="color:${w ? '#7c7' : '#c77'};">${w ? 'Yes' : 'No'}</span>`; });
                    row('Avg edge:', (i) => sig(avg[i]));
                    row('Extents:', (i) => ext(exts[i]));
                    row('Bounds:', (i) => bnd(bnds[i]));
                    if (fields) {
                        row('Fields:', (i) => {
                            const f = fields[i] || [];
                            return f.length ? `<span style="font-size:9px;color:#9bd;">${f.join(', ')}</span>` : '<span style="color:#666;">—</span>';
                        });
                    }
                    infoHTML += '</div>';
                    infoPanel.innerHTML = infoHTML;

                    // Prepare file paths + store for (re)send (also used when layout is switched)
                    const filepaths = meshFiles.map(f => `/view?filename=${encodeURIComponent(f)}&type=output&subfolder=`);
                    lastLoad = { numMeshes, filepaths };
                    buildAndSend();
                };

                return r;
            };
        }
    }
});
