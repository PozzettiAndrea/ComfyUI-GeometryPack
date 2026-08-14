/**
 * ComfyUI GeomPack - Multi Mesh Preview Widget
 * Grid viewer for 1-4 meshes with synchronized cameras
 */

import { app } from "../../../scripts/app.js";
import { createFullscreenButton } from "./utils/uiComponents.js";

// Auto-detect extension folder name
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-GeometryPack";
})();

console.log('[GeomPack Multi JS] Loading mesh_preview_multi.js extension');

app.registerExtension({
    name: "geometrypack.meshpreview.multi",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackPreviewMeshMulti") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Viewer state persisted via DOM widget serialization.
                // layout: "grid" (viewer_multi.html, default) | "wipe" (sliders).
                // grid_cols/grid_rows: null = auto (derived from mesh count).
                const viewerState = { layout: "grid", grid_cols: null, grid_rows: null, bar_collapsed: false, show_edges: false, camera_state: "", selected_field: "", selected_channel: "magnitude", selected_colormap: "erdc_rainbow_bright" };
                const viewerUrl = () => (viewerState.layout === "wipe" ? "viewer_multi_slider.html" : "viewer_multi.html");

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

                // Control bar: [collapse arrow] Layout:[Grid|Wipe]  Cols:[ ] Rows:[ ]
                // Grid = tiled viewports (viewer_multi.html); Wipe = N-1 draggable
                // dividers (viewer_multi_slider.html). The arrow hides everything
                // but itself; cols/rows show only for the grid layout.
                const bar = document.createElement("div");
                bar.style.cssText = "background:#1a1a1a;border-top:1px solid #444;padding:4px 8px;display:flex;gap:8px;align-items:center;font:11px monospace;color:#ccc;flex-shrink:0;";

                const collapseBtn = document.createElement("button");
                collapseBtn.title = "Show/hide controls";
                collapseBtn.style.cssText = "background:none;border:none;color:#ccc;cursor:pointer;font:12px monospace;padding:0 4px;line-height:1;";

                const controls = document.createElement("div");   // everything the arrow hides
                controls.style.cssText = "display:flex;gap:8px;align-items:center;";

                const layoutSel = document.createElement("select");
                layoutSel.style.cssText = "background:#333;color:#ccc;border:1px solid #555;border-radius:3px;font:11px monospace;padding:2px 6px;";
                layoutSel.innerHTML = '<option value="grid">Grid</option><option value="wipe">Wipe (sliders)</option>';

                const mkNum = () => {
                    const inp = document.createElement("input");
                    inp.type = "number"; inp.min = "1"; inp.step = "1";
                    inp.style.cssText = "width:40px;background:#333;color:#ccc;border:1px solid #555;border-radius:3px;font:11px monospace;padding:2px 4px;";
                    return inp;
                };
                const colsInput = mkNum();
                const rowsInput = mkNum();
                const gridControls = document.createElement("span");  // shown only for grid
                gridControls.style.cssText = "display:flex;gap:4px;align-items:center;";
                gridControls.appendChild(Object.assign(document.createElement("span"), { textContent: "Cols:" }));
                gridControls.appendChild(colsInput);
                gridControls.appendChild(Object.assign(document.createElement("span"), { textContent: "Rows:" }));
                gridControls.appendChild(rowsInput);

                // Mode selector (fields | texture) mirroring the node's `mode`
                // input. NOT client-side: mode changes what gets EXPORTED, so
                // changing it queues a re-run (same as editing the node widget).
                const modeSel = document.createElement("select");
                modeSel.style.cssText = "background:#333;color:#ccc;border:1px solid #555;border-radius:3px;font:11px monospace;padding:2px 6px;";
                modeSel.innerHTML = '<option value="fields">Fields</option><option value="texture">Texture</option>';
                modeSel.title = "Visualization mode (re-runs the node)";

                controls.appendChild(Object.assign(document.createElement("span"), { textContent: "Layout:" }));
                controls.appendChild(layoutSel);
                controls.appendChild(Object.assign(document.createElement("span"), { textContent: "Mode:" }));
                controls.appendChild(modeSel);
                controls.appendChild(gridControls);
                bar.appendChild(collapseBtn);
                bar.appendChild(controls);
                bar.appendChild(createFullscreenButton(container));  // outside `controls`: stays visible when collapsed

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

                // Order: canvas on top, controls below it, info panel last
                container.appendChild(iframe);
                container.appendChild(bar);
                container.appendChild(infoPanel);

                // Add widget
                const widget = this.addDOMWidget("preview_multi", "MESH_PREVIEW_MULTI", container, {
                    getValue() { return JSON.stringify(viewerState); },
                    setValue(v) {
                        try { Object.assign(viewerState, JSON.parse(v)); } catch(e) {}
                        if (viewerState.layout === "overlay") viewerState.layout = "grid";  // legacy name
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
                let lastLoad = null;   // { numMeshes, filepaths, autoCols, autoRows }
                const buildAndSend = () => {
                    if (!lastLoad || !iframe.contentWindow) return;
                    let msg;
                    if (viewerState.layout !== "wipe") {   // grid (default; also legacy "overlay")
                        // user override (grid_cols/grid_rows) else the auto dims from Python
                        const cols = viewerState.grid_cols || lastLoad.autoCols;
                        const rows = viewerState.grid_rows || lastLoad.autoRows;
                        msg = { type: 'LOAD_MULTI_MESH', numMeshes: lastLoad.numMeshes, meshFiles: lastLoad.filepaths,
                                gridCols: cols, gridRows: rows,
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

                // ---- control bar wiring ----
                const syncLayoutUI = () => {
                    gridControls.style.display = (viewerState.layout === "wipe") ? "none" : "flex";
                };
                const applyCollapsed = () => {
                    controls.style.display = viewerState.bar_collapsed ? "none" : "flex";
                    collapseBtn.textContent = viewerState.bar_collapsed ? "▸" : "▾";  // ▸ / ▾
                };
                collapseBtn.addEventListener('click', () => {
                    viewerState.bar_collapsed = !viewerState.bar_collapsed;
                    applyCollapsed();
                });

                // Layout switch swaps the viewer iframe (different HTML); buildAndSend
                // fires on its load. Cols/Rows just re-send to the SAME grid iframe.
                layoutSel.value = viewerState.layout;
                layoutSel.addEventListener('change', () => {
                    viewerState.layout = layoutSel.value;
                    syncLayoutUI();
                    iframeLoaded = false;
                    iframe.src = `/extensions/${EXTENSION_FOLDER}/${viewerUrl()}?v=` + Date.now();
                });

                // The edited field is authoritative; the OTHER dimension auto-raises
                // so cols*rows always fits every mesh. Without this, a grid smaller
                // than the mesh count overflows into implicit CSS rows (auto height)
                // and the layout stops matching the inputs. Both inputs are updated
                // so the UI always shows the grid actually in effect.
                const onDim = (edited) => () => {
                    const n = lastLoad?.numMeshes || 1;
                    let cols = Math.max(1, parseInt(colsInput.value, 10) || 0) || (lastLoad?.autoCols ?? 1);
                    let rows = Math.max(1, parseInt(rowsInput.value, 10) || 0) || (lastLoad?.autoRows ?? 1);
                    if (cols * rows < n) {
                        if (edited === "cols") rows = Math.ceil(n / cols);
                        else cols = Math.ceil(n / rows);
                    }
                    colsInput.value = cols;
                    rowsInput.value = rows;
                    viewerState.grid_cols = cols;
                    viewerState.grid_rows = rows;
                    buildAndSend();   // same iframe -> viewer re-tiles, no reload
                };
                colsInput.addEventListener('change', onDim("cols"));
                rowsInput.addEventListener('change', onDim("rows"));

                // Mode: bar select <-> node widget, two-way (bar change re-runs).
                const modeWidget = this.widgets?.find(w => w.name === "mode");
                if (modeWidget) {
                    modeSel.value = modeWidget.value || "fields";
                    const origModeCb = modeWidget.callback;
                    modeWidget.callback = function(value) {
                        const res = origModeCb?.apply(this, arguments);
                        modeSel.value = value;
                        return res;
                    };
                }
                modeSel.addEventListener("change", () => {
                    if (modeWidget) modeWidget.value = modeSel.value;
                    app.queuePrompt();
                });

                syncLayoutUI();
                applyCollapsed();

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
                    lastLoad = { numMeshes, filepaths, autoCols: gridCols, autoRows: gridRows };
                    // Reflect the effective grid dims (user override, else the auto value)
                    colsInput.value = viewerState.grid_cols ?? gridCols;
                    rowsInput.value = viewerState.grid_rows ?? gridRows;
                    buildAndSend();
                };

                return r;
            };
        }
    }
});
