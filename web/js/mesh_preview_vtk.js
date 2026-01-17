/**
 * ComfyUI GeomPack - VTK.js Mesh Preview Widget
 * Scientific visualization with VTK.js
 */

import { app } from "../../../scripts/app.js";
import { EXTENSION_FOLDER, getViewerUrl } from "./utils/extensionFolder.js";
import { createContainer, createIframe, createInfoPanel, showPanelError, createWidgetOptions } from "./utils/uiComponents.js";
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
                const widget = this.addDOMWidget("preview_vtk", "MESH_PREVIEW_VTK", container, createWidgetOptions());
                widget.computeSize = () => [512, 640];

                // Store references
                this.meshViewerIframeVTK = iframe;
                this.meshInfoPanelVTK = infoPanel;

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
                            isWatertight: message.is_watertight?.[0],
                            fieldNames: message.field_names?.[0] || [],
                            hasTexture: message.has_texture?.[0],
                            hasVertexColors: message.has_vertex_colors?.[0],
                            visualKind: message.visual_kind?.[0]
                        });

                        infoPanel.innerHTML = infoHTML;

                        // Build file path and message
                        const filepath = buildViewUrl(filename);
                        const messageData = {
                            type: "LOAD_MESH",
                            filepath: filepath,
                            timestamp: Date.now()
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

