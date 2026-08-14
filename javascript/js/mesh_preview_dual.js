/**
 * ComfyUI GeomPack - Dual Mesh Preview Widget
 * Unified viewer for side-by-side and overlay dual mesh visualization
 * with full field visualization support
 */

import { app } from "../../../scripts/app.js";
import { EXTENSION_FOLDER, getViewerUrl } from "./utils/extensionFolder.js";
import { createContainer, createIframe, createInfoPanel, createFullscreenButton } from "./utils/uiComponents.js";
import { buildDualMeshInfoHTML, formatExtents } from "./utils/formatting.js";
import { createViewerManager, createErrorHandler, buildViewUrl, createLoadDualMeshMessage } from "./utils/postMessage.js";

app.registerExtension({
    name: "geometrypack.meshpreview.dual",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackPreviewMeshDual") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Viewer state persisted via DOM widget serialization
                const viewerState = { show_edges: false, camera_state: "", selected_field: "", selected_channel: "magnitude", selected_colormap: "erdc_rainbow_bright" };

                // Create container for viewer + info panel
                const container = createContainer();

                // Create iframe for VTK.js viewer
                const iframe = createIframe(getViewerUrl("viewer_dual"), { minHeight: "550px" });

                // Create mesh info panel
                const infoPanel = createInfoPanel("Mesh info will appear here after execution");

                // Control bar: Layout + Mode (mirrors the node widgets).
                // side_by_side <-> slider share the SAME exported files, so that
                // switch is instant/client-side; overlay needs a combined export
                // and mode changes the export format, so those queue a re-run.
                const bar = document.createElement("div");
                bar.style.cssText = "background:#1a1a1a;border-bottom:1px solid #444;padding:4px 8px;display:flex;gap:8px;align-items:center;font:11px monospace;color:#ccc;flex-shrink:0;";
                const mkSel = (html, title) => {
                    const s = document.createElement("select");
                    s.style.cssText = "background:#333;color:#ccc;border:1px solid #555;border-radius:3px;font:11px monospace;padding:2px 6px;";
                    s.innerHTML = html; s.title = title;
                    return s;
                };
                const layoutSel = mkSel(
                    '<option value="side_by_side">Side by side</option><option value="overlay">Overlay</option><option value="slider">Slider</option>',
                    "Side by side <-> Slider switch instantly; Overlay re-runs (needs a combined export)");
                const modeSel = mkSel(
                    '<option value="fields">Fields</option><option value="texture">Texture</option>',
                    "Visualization mode (re-runs the node)");
                bar.appendChild(Object.assign(document.createElement("span"), { textContent: "Layout:" }));
                bar.appendChild(layoutSel);
                bar.appendChild(Object.assign(document.createElement("span"), { textContent: "Mode:" }));
                bar.appendChild(modeSel);
                bar.appendChild(createFullscreenButton(container));

                // Add bar, iframe and info panel to container
                container.appendChild(bar);
                container.appendChild(iframe);
                container.appendChild(infoPanel);

                // Add widget
                const widget = this.addDOMWidget("preview_dual", "MESH_PREVIEW_DUAL", container, {
                    getValue() { return JSON.stringify(viewerState); },
                    setValue(v) {
                        try { Object.assign(viewerState, JSON.parse(v)); } catch(e) {}
                    }
                });
                widget.computeSize = () => [768, 680];

                // Store references
                this.meshViewerIframeDual = iframe;
                this.meshInfoPanelDual = infoPanel;

                this.setSize(this.computeSize());

                // Bidirectional sync: viewer → node widgets (viewerState + real widgets like opacity)
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

                // Create viewer manager
                const viewerManager = createViewerManager(iframe, "[GeomPack Dual]");

                // Listen for error messages
                window.addEventListener('message', createErrorHandler(infoPanel, "[GeomPack Dual]", iframe));

                // Set initial node size
                this.setSize([768, 680]);

                // ---- widget <-> bar sync + opacity visibility ----
                const layoutWidget = this.widgets?.find(w => w.name === "layout");
                const modeWidget = this.widgets?.find(w => w.name === "mode");

                // opacity_1/opacity_2 only mean anything in the overlay layout --
                // hide the widgets otherwise (same hidden-type trick as
                // preview_mesh_batch_render.js).
                const setOpacityVisible = (visible) => {
                    let changed = false;
                    for (const nm of ["opacity_1", "opacity_2"]) {
                        const w = node.widgets?.find(x => x.name === nm);
                        if (!w) continue;
                        if (!visible && w.type !== "geometrypack_hidden") {
                            w._gpOrigType = w.type;
                            w._gpOrigComputeSize = w.computeSize;
                            w.type = "geometrypack_hidden";   // unknown type -> not drawn
                            w.computeSize = () => [0, -4];
                            changed = true;
                        } else if (visible && w.type === "geometrypack_hidden") {
                            w.type = w._gpOrigType;
                            w.computeSize = w._gpOrigComputeSize;
                            changed = true;
                        }
                    }
                    if (changed) node.setSize(node.computeSize());
                };

                let lastMsg = null;   // last execution message, for client-side switches

                const syncBar = () => {
                    if (layoutWidget) layoutSel.value = layoutWidget.value || "side_by_side";
                    if (modeWidget) modeSel.value = modeWidget.value || "fields";
                    setOpacityVisible(layoutSel.value === "overlay");
                };
                // Keep the bar honest when the node widgets are edited directly.
                for (const [w, after] of [[layoutWidget, syncBar], [modeWidget, syncBar]]) {
                    if (!w) continue;
                    const orig = w.callback;
                    w.callback = function(value) {
                        const res = orig?.apply(this, arguments);
                        after();
                        return res;
                    };
                }

                layoutSel.addEventListener("change", () => {
                    const newLayout = layoutSel.value;
                    if (layoutWidget) layoutWidget.value = newLayout;   // persists for next run
                    setOpacityVisible(newLayout === "overlay");
                    // side_by_side <-> slider reuse the SAME exported files ->
                    // re-render client-side from the stored message. Overlay (either
                    // direction) needs the combined export -> re-run.
                    if (newLayout !== "overlay" && lastMsg?.mesh_1_file && lastMsg?.mesh_2_file) {
                        render(lastMsg, newLayout);
                    } else {
                        app.queuePrompt();
                    }
                });
                modeSel.addEventListener("change", () => {
                    if (modeWidget) modeWidget.value = modeSel.value;
                    app.queuePrompt();   // different export format -> must re-run
                });
                syncBar();

                // Render one execution message; layoutOverride enables the
                // client-side side_by_side <-> slider switch without a re-run.
                const render = (message, layoutOverride) => {
                    const layout = layoutOverride || message.layout[0];
                    const mode = message.mode?.[0] || "fields";

                    // Determine viewer type and name
                    let viewerType, viewerName;
                    if (layout === 'slider') {
                        viewerType = "slider";
                        viewerName = "viewer_dual_slider";
                    } else if (mode === "texture") {
                        viewerType = "texture";
                        viewerName = "viewer_dual_textured";
                    } else {
                        viewerType = "fields";
                        viewerName = "viewer_dual";
                    }

                    let postMessageData;

                    if (layout === 'side_by_side' || layout === 'slider') {
                        // Side-by-side mode
                        if (!message?.mesh_1_file || !message?.mesh_2_file) {
                            return;
                        }

                        // Build info HTML using utility
                        const infoHTML = buildDualMeshInfoHTML({
                            mode: mode,
                            layout: layout,
                            mesh1: {
                                vertices: message.vertex_count_1?.[0] || 'N/A',
                                faces: message.face_count_1?.[0] || 'N/A',
                                extents: message.extents_1?.[0] || [],
                                isWatertight: message.is_watertight_1?.[0],
                                hasTexture: message.has_texture_1?.[0]
                            },
                            mesh2: {
                                vertices: message.vertex_count_2?.[0] || 'N/A',
                                faces: message.face_count_2?.[0] || 'N/A',
                                extents: message.extents_2?.[0] || [],
                                isWatertight: message.is_watertight_2?.[0],
                                hasTexture: message.has_texture_2?.[0]
                            },
                            commonFields: message.common_fields?.[0] || []
                        });

                        infoPanel.innerHTML = infoHTML;

                        postMessageData = createLoadDualMeshMessage({
                            layout: layout,
                            mesh1Filepath: buildViewUrl(message.mesh_1_file[0]),
                            mesh2Filepath: buildViewUrl(message.mesh_2_file[0]),
                            opacity1: message.opacity_1?.[0] || 1.0,
                            opacity2: message.opacity_2?.[0] || 1.0,
                            showEdges: viewerState.show_edges,
                            cameraState: viewerState.camera_state,
                            selectedField: viewerState.selected_field,
                            selectedChannel: viewerState.selected_channel,
                            selectedColormap: viewerState.selected_colormap,
                        });

                    } else {
                        // Overlay mode
                        if (!message?.mesh_file) {
                            return;
                        }

                        // Build info HTML using utility
                        const infoHTML = buildDualMeshInfoHTML({
                            mode: mode,
                            layout: "overlay",
                            mesh1: {
                                vertices: message.vertex_count_1?.[0] || 'N/A',
                                faces: message.face_count_1?.[0] || 'N/A',
                                hasTexture: message.has_texture_1?.[0]
                            },
                            mesh2: {
                                vertices: message.vertex_count_2?.[0] || 'N/A',
                                faces: message.face_count_2?.[0] || 'N/A',
                                hasTexture: message.has_texture_2?.[0]
                            },
                            commonFields: message.common_fields?.[0] || []
                        });

                        infoPanel.innerHTML = infoHTML;

                        postMessageData = createLoadDualMeshMessage({
                            layout: layout,
                            meshFilepath: buildViewUrl(message.mesh_file[0]),
                            opacity1: message.opacity_1?.[0] || 1.0,
                            opacity2: message.opacity_2?.[0] || 1.0,
                            showEdges: viewerState.show_edges,
                            cameraState: viewerState.camera_state,
                            selectedField: viewerState.selected_field,
                            selectedChannel: viewerState.selected_channel,
                            selectedColormap: viewerState.selected_colormap,
                        });
                    }

                    // Switch viewer if needed and send message
                    viewerManager.switchViewer(viewerType, getViewerUrl(viewerName), postMessageData);
                };

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);
                    if (!message?.layout) return;
                    lastMsg = message;
                    syncBar();   // reflect the layout/mode actually run + opacity visibility
                    render(message);
                };

                return r;
            };
        }
    }
});
