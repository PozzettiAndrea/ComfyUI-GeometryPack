/**
 * ComfyUI GeomPack - Connected Components Dynamic Display
 * Shows component details in the node after execution
 */

import { app } from "../../../scripts/app.js";
import { createAnalysisPanel, createWidgetOptions } from "./utils/uiComponents.js";
import { buildConnectedComponentsHTML, calculatePanelHeight } from "./utils/analysisPanel.js";

app.registerExtension({
    name: "geompack.connected_components",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackConnectedComponents") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create info panel using utility
                const infoPanel = createAnalysisPanel("Run workflow to see component details");

                // Add widget
                const widget = this.addDOMWidget("component_info", "COMPONENT_INFO", infoPanel, createWidgetOptions());
                widget.computeSize = () => [this.size[0] - 20, 120];

                this.componentInfoPanel = infoPanel;

                // Handle execution results
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    if (message?.component_data && message.component_data.length > 0) {
                        let html = '';

                        for (const meshData of message.component_data) {
                            html += buildConnectedComponentsHTML(meshData);
                        }

                        infoPanel.innerHTML = html;

                        // Resize widget based on content
                        const numRows = message.component_data[0]?.components?.length || 0;
                        const height = calculatePanelHeight(numRows);
                        widget.computeSize = () => [this.size[0] - 20, height];
                        this.setDirtyCanvas(true);
                    }
                };

                return r;
            };
        }
    }
});
