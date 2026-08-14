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

                // Viewer state persisted via DOM widget serialization.
                // layout + opacities live HERE (not as node inputs): Python exports
                // both view sets every run, so they are pure client-side choices.
                const viewerState = { layout: "side_by_side", opacity_1: 1.0, opacity_2: 1.0, color_1: "#ff4d4d", color_2: "#4d4dff", show_edges: false, camera_state: "", selected_field: "", selected_channel: "magnitude", selected_colormap: "erdc_rainbow_bright" };

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
                bar.style.cssText = "background:#1a1a1a;border-top:1px solid #444;padding:4px 8px;display:flex;gap:8px;align-items:center;font:11px monospace;color:#ccc;flex-shrink:0;";
                const mkSel = (html, title) => {
                    const s = document.createElement("select");
                    s.style.cssText = "background:#333;color:#ccc;border:1px solid #555;border-radius:3px;font:11px monospace;padding:2px 6px;";
                    s.innerHTML = html; s.title = title;
                    return s;
                };
                // Mode is controlled solely by the node's `mode` input widget --
                // no bar selector for it.
                const layoutSel = mkSel(
                    '<option value="side_by_side">Side by side</option><option value="overlay">Overlay</option><option value="slider">Slider</option>',
                    "Layout switches instantly (all views are exported every run)");
                bar.appendChild(Object.assign(document.createElement("span"), { textContent: "Layout:" }));
                bar.appendChild(layoutSel);
                // Opacity inputs (overlay only) -- applied client-side by re-posting
                const mkOp = () => {
                    const inp = document.createElement("input");
                    inp.type = "number"; inp.min = "0"; inp.max = "1"; inp.step = "0.1";
                    inp.style.cssText = "width:44px;background:#333;color:#ccc;border:1px solid #555;border-radius:3px;font:11px monospace;padding:2px 4px;";
                    return inp;
                };
                const op1Input = mkOp();
                const op2Input = mkOp();
                // Per-mesh color pickers (overlay, fields viewer)
                const mkColor = () => {
                    const inp = document.createElement("input");
                    inp.type = "color";
                    inp.style.cssText = "width:26px;height:20px;padding:0;border:1px solid #555;border-radius:3px;background:#333;cursor:pointer;";
                    return inp;
                };
                const col1Input = mkColor();
                const col2Input = mkColor();
                const opControls = document.createElement("span");
                opControls.style.cssText = "display:flex;gap:4px;align-items:center;";
                opControls.appendChild(Object.assign(document.createElement("span"), { textContent: "1:" }));
                opControls.appendChild(col1Input);
                opControls.appendChild(op1Input);
                opControls.appendChild(Object.assign(document.createElement("span"), { textContent: "2:" }));
                opControls.appendChild(col2Input);
                opControls.appendChild(op2Input);
                bar.appendChild(opControls);
                bar.appendChild(createFullscreenButton(container));

                // Order: canvas on top, controls below it, info panel last
                container.appendChild(iframe);
                container.appendChild(bar);
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

                // ---- bar wiring ----
                // mode is the node's own input widget (not in the bar); layout +
                // opacity + colors live in viewerState (persisted via the DOM
                // widget's serialization).
                const setOpacityVisible = (visible) => {
                    opControls.style.display = visible ? "flex" : "none";
                };

                let lastMsg = null;   // last execution message, for client-side switches

                const syncBar = () => {
                    layoutSel.value = viewerState.layout || "side_by_side";
                    op1Input.value = viewerState.opacity_1 ?? 1.0;
                    op2Input.value = viewerState.opacity_2 ?? 1.0;
                    col1Input.value = viewerState.color_1 || "#ff4d4d";
                    col2Input.value = viewerState.color_2 || "#4d4dff";
                    setOpacityVisible(layoutSel.value === "overlay");
                };

                // Layout: fully client-side -- every run exports both the separate
                // pair AND the combined overlay file, so any switch just re-renders.
                layoutSel.addEventListener("change", () => {
                    viewerState.layout = layoutSel.value;
                    setOpacityVisible(viewerState.layout === "overlay");
                    if (lastMsg) render(lastMsg);
                });
                // Opacity: client-side re-post; persisted in viewerState.
                const onOpacity = () => {
                    viewerState.opacity_1 = Math.max(0, Math.min(1, parseFloat(op1Input.value) || 0));
                    viewerState.opacity_2 = Math.max(0, Math.min(1, parseFloat(op2Input.value) || 0));
                    if (lastMsg) render(lastMsg);
                };
                op1Input.addEventListener("change", onOpacity);
                op2Input.addEventListener("change", onOpacity);
                // Colors: client-side re-post, persisted in viewerState.
                const onColor = () => {
                    viewerState.color_1 = col1Input.value;
                    viewerState.color_2 = col2Input.value;
                    if (lastMsg) render(lastMsg);
                };
                col1Input.addEventListener("change", onColor);
                col2Input.addEventListener("change", onColor);
                syncBar();

                // Render one execution message with the CURRENT client-side layout
                // and opacities (viewerState). Called on execution and on any bar
                // change -- both view sets are always present in the message.
                const render = (message) => {
                    const layout = viewerState.layout || "side_by_side";
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
                            opacity1: viewerState.opacity_1 ?? 1.0,
                            opacity2: viewerState.opacity_2 ?? 1.0,
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
                            // combined export injects a mesh_id field
                            commonFields: message.common_fields_overlay?.[0] || message.common_fields?.[0] || []
                        });

                        infoPanel.innerHTML = infoHTML;

                        postMessageData = createLoadDualMeshMessage({
                            layout: layout,
                            // Fields viewer renders overlay from the TWO per-mesh
                            // files (one actor each -> per-mesh color/opacity);
                            // the textured viewer still uses the combined GLB.
                            ...(mode === "texture"
                                ? { meshFilepath: buildViewUrl(message.mesh_file[0]) }
                                : { mesh1Filepath: buildViewUrl(message.mesh_1_file[0]),
                                    mesh2Filepath: buildViewUrl(message.mesh_2_file[0]),
                                    color1: viewerState.color_1 || "#ff4d4d",
                                    color2: viewerState.color_2 || "#4d4dff" }),
                            opacity1: viewerState.opacity_1 ?? 1.0,
                            opacity2: viewerState.opacity_2 ?? 1.0,
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
                    if (!message?.mesh_1_file) return;
                    lastMsg = message;
                    syncBar();   // reflect the layout/mode actually run + opacity visibility
                    render(message);
                };

                return r;
            };
        }
    }
});
