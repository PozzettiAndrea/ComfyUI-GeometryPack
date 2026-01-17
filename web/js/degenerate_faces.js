/**
 * ComfyUI GeomPack - Degenerate Faces Dynamic Display
 * Shows degenerate faces and smallest faces by area
 */

import { app } from "../../../scripts/app.js";
import { createAnalysisPanel, createWidgetOptions } from "./utils/uiComponents.js";

app.registerExtension({
    name: "geompack.degenerate_faces",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackDegenerateFaces") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create info panel using utility (with larger maxHeight for this panel)
                const infoPanel = createAnalysisPanel("Run workflow to detect degenerate faces", { maxHeight: "300px" });

                // Add widget
                const widget = this.addDOMWidget("degenerate_info", "DEGENERATE_INFO", infoPanel, createWidgetOptions());
                widget.computeSize = () => [this.size[0] - 20, 150];

                this.degenerateInfoPanel = infoPanel;

                // Handle execution results
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    if (message?.degenerate_data && message.degenerate_data.length > 0) {
                        let html = '';

                        for (const data of message.degenerate_data) {
                            const { mesh_name, num_degenerate, duplicate_count, zero_count, total_faces, degenerate_faces, smallest_faces } = data;

                            // Header
                            html += `<div style="margin-bottom: 8px;">`;

                            const statusColor = num_degenerate === 0 ? '#6f6' : '#f66';
                            const statusText = num_degenerate === 0 ? 'No degenerate faces' : `${num_degenerate} degenerate`;

                            html += `<div style="color: #fff; font-weight: bold; margin-bottom: 4px;">`;
                            html += `${mesh_name}: <span style="color: ${statusColor};">${statusText}</span>`;
                            html += `</div>`;

                            // Degenerate faces section (if any)
                            if (num_degenerate > 0) {
                                html += `<div style="color: #888; font-size: 9px; margin-bottom: 4px;">`;
                                const parts = [];
                                if (duplicate_count > 0) parts.push(`${duplicate_count} duplicate verts`);
                                if (zero_count > 0) parts.push(`${zero_count} zero area`);
                                html += parts.join(', ');
                                html += `</div>`;

                                html += `<table style="width: 100%; border-collapse: collapse; font-size: 9px; margin-bottom: 8px;">`;
                                html += `<tr style="color: #f66; border-bottom: 1px solid #333;">`;
                                html += `<th style="text-align: left; padding: 2px 4px;">Degenerate</th>`;
                                html += `<th style="text-align: right; padding: 2px 4px;">Area</th>`;
                                html += `<th style="text-align: left; padding: 2px 4px;">Issue</th>`;
                                html += `</tr>`;

                                // Limit to 30 degenerate faces
                                const displayDegenerate = degenerate_faces.slice(0, 30);
                                for (const face of displayDegenerate) {
                                    const areaStr = face.area.toExponential(2);
                                    const reasonStr = face.reason === 'duplicate_vertex' ? 'dup verts' : 'zero area';

                                    html += `<tr style="border-bottom: 1px solid #222;">`;
                                    html += `<td style="padding: 2px 4px; color: #f66;">${face.id}</td>`;
                                    html += `<td style="text-align: right; padding: 2px 4px; color: #f96;">${areaStr}</td>`;
                                    html += `<td style="padding: 2px 4px; color: #888;">${reasonStr}</td>`;
                                    html += `</tr>`;
                                }

                                if (num_degenerate > 30) {
                                    html += `<tr><td colspan="3" style="padding: 4px; color: #888; text-align: center;">`;
                                    html += `... and ${num_degenerate - 30} more degenerate`;
                                    html += `</td></tr>`;
                                }

                                html += `</table>`;
                            }

                            // Smallest faces section (limit based on how many degenerate we showed)
                            const maxSmallest = Math.min(30, 60 - Math.min(num_degenerate, 30));

                            html += `<div style="color: #6cf; font-size: 9px; margin-bottom: 2px; margin-top: 4px;">`;
                            html += `Smallest ${maxSmallest} faces by area:`;
                            html += `</div>`;

                            html += `<table style="width: 100%; border-collapse: collapse; font-size: 9px;">`;
                            html += `<tr style="color: #888; border-bottom: 1px solid #333;">`;
                            html += `<th style="text-align: left; padding: 2px 4px;">Face</th>`;
                            html += `<th style="text-align: right; padding: 2px 4px;">Area</th>`;
                            html += `</tr>`;

                            const displaySmallest = smallest_faces.slice(0, maxSmallest);
                            for (const face of displaySmallest) {
                                const areaStr = face.area.toExponential(2);

                                html += `<tr style="border-bottom: 1px solid #222;">`;
                                html += `<td style="padding: 2px 4px; color: #6cf;">${face.id}</td>`;
                                html += `<td style="text-align: right; padding: 2px 4px; color: #ccc;">${areaStr}</td>`;
                                html += `</tr>`;
                            }

                            html += `</table>`;
                            html += `</div>`;
                        }

                        infoPanel.innerHTML = html;

                        // Resize widget
                        const data = message.degenerate_data[0];
                        const numDegen = Math.min(data?.degenerate_faces?.length || 0, 30);
                        const numSmall = Math.min(30, 60 - numDegen);
                        const totalRows = numDegen + numSmall + 5;
                        const height = Math.min(Math.max(100, totalRows * 14 + 40), 350);
                        widget.computeSize = () => [this.size[0] - 20, height];
                        this.setDirtyCanvas(true);
                    }
                };

                return r;
            };
        }
    }
});
