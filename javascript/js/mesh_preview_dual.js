/**
 * ComfyUI GeomPack - Dual Mesh Preview Widget
 * Unified viewer for side-by-side and overlay dual mesh visualization
 * with full field visualization support
 */

import { app } from "../../../scripts/app.js";
import { EXTENSION_FOLDER, getViewerUrl } from "./utils/extensionFolder.js";
import { createContainer, createIframe, createInfoPanel } from "./utils/uiComponents.js";
import { buildDualMeshInfoHTML, formatExtents } from "./utils/formatting.js";
import { createViewerManager, createErrorHandler, buildViewUrl, createLoadDualMeshMessage } from "./utils/postMessage.js";

app.registerExtension({
    name: "geompack.meshpreview.dual",

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

                // Add iframe and info panel to container
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
                window.addEventListener('message', createErrorHandler(infoPanel, "[GeomPack Dual]"));

                // Set initial node size
                this.setSize([768, 680]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    if (!message?.layout) {
                        return;
                    }

                    const layout = message.layout[0];
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

                return r;
            };
        }
    }
});
