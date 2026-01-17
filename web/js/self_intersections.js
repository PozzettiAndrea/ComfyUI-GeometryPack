/**
 * ComfyUI GeomPack - Self Intersections Dynamic Display
 * Shows self-intersecting faces in the node after execution
 */

import { app } from "../../../scripts/app.js";
import { createAnalysisPanel, createWidgetOptions } from "./utils/uiComponents.js";
import { buildSelfIntersectionsHTML, calculatePanelHeight } from "./utils/analysisPanel.js";

app.registerExtension({
    name: "geompack.self_intersections",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackDetectSelfIntersections") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create info panel using utility
                const infoPanel = createAnalysisPanel("Run workflow to detect self-intersections");

                // Add widget
                const widget = this.addDOMWidget("intersection_info", "INTERSECTION_INFO", infoPanel, createWidgetOptions());
                widget.computeSize = () => [this.size[0] - 20, 120];

                this.intersectionInfoPanel = infoPanel;

                // Handle execution results
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    if (message?.intersection_data && message.intersection_data.length > 0) {
                        let html = '';

                        for (const data of message.intersection_data) {
                            html += buildSelfIntersectionsHTML(data);
                        }

                        infoPanel.innerHTML = html;

                        // Resize widget based on content
                        const numRows = message.intersection_data[0]?.faces?.length || 0;
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
