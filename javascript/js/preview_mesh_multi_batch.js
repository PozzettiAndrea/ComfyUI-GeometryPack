/**
 * ComfyUI GeomPack - Multi Mesh Preview Widget (Batch Input)
 * Grid viewer for a mesh batch, with a user-chosen rows x cols layout.
 * Always uses the fields (scalar) viewer -- no wipe/slider/texture modes.
 */

import { app } from "../../../scripts/app.js";

const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-GeometryPack";
})();

console.log('[GeomPack Multi Batch JS] Loading preview_mesh_multi_batch.js extension');

app.registerExtension({
    name: "geompack.meshpreview.multibatch",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackPreviewMeshMultiBatch") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Viewer state persisted via DOM widget serialization
                const viewerState = { show_edges: false, camera_state: "", selected_field: "", selected_channel: "magnitude", selected_colormap: "erdc_rainbow_bright" };

                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#2a2a2a";

                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1";
                iframe.style.minHeight = "450px";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";
                iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_multi.html?v=` + Date.now();

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

                container.appendChild(iframe);
                container.appendChild(infoPanel);

                const widget = this.addDOMWidget("preview_multi_batch", "MESH_PREVIEW_MULTI_BATCH", container, {
                    getValue() { return JSON.stringify(viewerState); },
                    setValue(v) {
                        try { Object.assign(viewerState, JSON.parse(v)); } catch(e) {}
                    }
                });

                widget.computeSize = () => [768, 580];

                this.meshViewerIframeMultiBatch = iframe;
                this.meshInfoPanelMultiBatch = infoPanel;

                this.setSize([768, 580]);

                // Bidirectional sync: viewer -> node widgets (viewerState + real widgets)
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

                let lastLoad = null;   // { numMeshes, filepaths, gridCols, gridRows }
                const buildAndSend = () => {
                    if (!lastLoad || !iframe.contentWindow) return;
                    const msg = {
                        type: 'LOAD_MULTI_MESH', numMeshes: lastLoad.numMeshes, meshFiles: lastLoad.filepaths,
                        gridCols: lastLoad.gridCols, gridRows: lastLoad.gridRows,
                        timestamp: Date.now(), showEdges: viewerState.show_edges, cameraState: viewerState.camera_state,
                        selectedField: viewerState.selected_field, selectedChannel: viewerState.selected_channel,
                        selectedColormap: viewerState.selected_colormap,
                    };
                    iframe.contentWindow.postMessage(msg, "*");
                };
                iframe.addEventListener('load', () => buildAndSend());

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

                    console.log(`[GeomPack Multi Batch] onExecuted: ${numMeshes} meshes, grid ${gridCols}x${gridRows}`);

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

                    const filepaths = meshFiles.map(f => `/view?filename=${encodeURIComponent(f)}&type=output&subfolder=`);
                    lastLoad = { numMeshes, filepaths, gridCols, gridRows };
                    buildAndSend();
                };

                return r;
            };
        }
    }
});
