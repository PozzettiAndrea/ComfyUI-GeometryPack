/**
 * ComfyUI GeomPack - VTK.js Mesh Preview Widget
 * Scientific visualization with VTK.js
 */

import { app } from "../../../scripts/app.js";
import { EXTENSION_FOLDER, getViewerUrl } from "./utils/extensionFolder.js";
import { createContainer, createIframe, createInfoPanel, showPanelError } from "./utils/uiComponents.js";
import { buildMeshInfoHTML } from "./utils/formatting.js";
import { createScreenshotHandler } from "./utils/screenshot.js";
import { createViewerManager, createErrorHandler, buildViewUrl } from "./utils/postMessage.js";

app.registerExtension({
    name: "geompack.meshpreview.vtk",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackPreviewMeshVTK") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Viewer state persisted via DOM widget serialization
                const viewerState = { show_edges: false, camera_state: "", selected_field: "", selected_channel: "magnitude", selected_colormap: "erdc_rainbow_bright" };

                // Create container for viewer + info panel
                const container = createContainer();

                // Create iframe for VTK.js viewer
                const iframe = createIframe(getViewerUrl("viewer_vtk"));

                // Create mesh info panel
                const infoPanel = createInfoPanel("Mesh info will appear here after execution");

                // Add iframe and info panel to container
                container.appendChild(iframe);
                container.appendChild(infoPanel);

                // Add widget
                const widget = this.addDOMWidget("preview_vtk", "MESH_PREVIEW_VTK", container, {
                    getValue() { return JSON.stringify(viewerState); },
                    setValue(v) {
                        try { Object.assign(viewerState, JSON.parse(v)); } catch(e) {}
                    },
                    // Low constant minimum: the DOM widget fills the node's height and
                    // the node resizes freely down to ~120px. A CONSTANT can't feed back
                    // the way a node.size-derived computeSize did (which grew forever),
                    // so do NOT re-add a computeSize override here.
                    getMinHeight: () => 120,
                });

                // Store references
                this.meshViewerIframeVTK = iframe;
                this.meshInfoPanelVTK = infoPanel;

                this.setSize([512, 640]);

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
                    } else if (event.data.type === 'TOGGLE_INFO') {
                        // viewer's minimise arrow also hides/shows this node's info panel
                        infoPanel.style.display = event.data.collapsed ? 'none' : '';
                    }
                });

                // Create viewer manager for handling viewer switching
                const viewerManager = createViewerManager(iframe, "[GeomPack VTK]");

                // Listen for screenshot and error messages
                window.addEventListener('message', createScreenshotHandler('vtk-screenshot'));
                window.addEventListener('message', createErrorHandler(infoPanel, "[GeomPack VTK]"));

                // Set initial node size
                this.setSize([512, 640]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    if (message?.mesh_file && message.mesh_file[0]) {
                        const filename = message.mesh_file[0];
                        const viewerType = message.viewer_type?.[0] || "fields";
                        const mode = message.mode?.[0] || "fields";

                        // Determine which viewer HTML to use
                        let viewerName;
                        if (viewerType === "pbr") {
                            viewerName = "viewer_pbr";
                        } else if (viewerType === "texture") {
                            viewerName = "viewer_vtk_textured";
                        } else {
                            viewerName = "viewer_vtk";
                        }

                        // Build info HTML using utility
                        const infoHTML = buildMeshInfoHTML({
                            mode: mode,
                            vertices: message.vertex_count?.[0] || 'N/A',
                            faces: message.face_count?.[0] || 'N/A',
                            boundsMin: message.bounds_min?.[0] || [],
                            boundsMax: message.bounds_max?.[0] || [],
                            extents: message.extents?.[0] || [],
                            avgEdge: message.avg_edge_length?.[0],
                            isWatertight: message.is_watertight?.[0],
                            fieldNames: message.field_names?.[0] || [],
                            fieldRanges: message.field_ranges?.[0] || [],
                            hasTexture: message.has_texture?.[0],
                            hasVertexColors: message.has_vertex_colors?.[0],
                            visualKind: message.visual_kind?.[0]
                        });

                        infoPanel.innerHTML = infoHTML;

                        // Click a field's min/max value to fly the camera to that element.
                        const minPos = message.field_min_pos?.[0] || [];
                        const maxPos = message.field_max_pos?.[0] || [];
                        infoPanel.querySelectorAll('.field-extreme').forEach((el) => {
                            el.addEventListener('click', () => {
                                const i = parseInt(el.dataset.fidx, 10);
                                const pos = (el.dataset.which === 'min' ? minPos : maxPos)[i];
                                if (pos && pos.length === 3) {
                                    viewerManager.sendMessage({ type: 'FOCUS_ON_POINT', point: pos, timestamp: Date.now() });
                                }
                            });
                        });

                        // Build file path and message
                        const filepath = buildViewUrl(filename);
                        const messageData = {
                            type: "LOAD_MESH",
                            filepath: filepath,
                            timestamp: Date.now(),
                            showEdges: viewerState.show_edges,
                            cameraState: viewerState.camera_state,
                            selectedField: viewerState.selected_field,
                            selectedChannel: viewerState.selected_channel,
                            selectedColormap: viewerState.selected_colormap,
                        };

                        // Switch viewer if needed and send message
                        viewerManager.switchViewer(viewerType, getViewerUrl(viewerName), messageData);
                    }
                };

                return r;
            };
        }
    }
});

