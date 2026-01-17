/**
 * ComfyUI GeomPack - Open Edges Dynamic Display
 * Shows faces with open/boundary edges in the node after execution
 */

import { app } from "../../../scripts/app.js";
import { createAnalysisPanel, createWidgetOptions } from "./utils/uiComponents.js";
import { calculatePanelHeight, buildTableHeader, buildMoreRow } from "./utils/analysisPanel.js";

app.registerExtension({
    name: "geompack.open_edges",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeomPackOpenEdges") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create info panel using utility
                const infoPanel = createAnalysisPanel("Run workflow to see open edge details");

                // Add widget
                const widget = this.addDOMWidget("open_edges_info", "OPEN_EDGES_INFO", infoPanel, createWidgetOptions());

                widget.computeSize = () => [this.size[0] - 20, 120];

                this.openEdgesInfoPanel = infoPanel;

                // Handle execution results
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecuted?.apply(this, arguments);

                    if (message?.open_edges_data && message.open_edges_data.length > 0) {
                        let html = '';

                        for (const meshData of message.open_edges_data) {
                            const { mesh_name, num_open_edges, num_boundary_faces, num_boundary_vertices, total_faces, total_vertices, faces } = meshData;

                            // Header with mesh name and totals
                            html += `<div style="margin-bottom: 8px;">`;

                            // Color based on whether there are open edges
                            const statusColor = num_open_edges === 0 ? '#6f6' : '#f96';
                            const statusText = num_open_edges === 0 ? 'Watertight' : `${num_open_edges} open edge(s)`;

                            html += `<div style="color: #fff; font-weight: bold; margin-bottom: 4px;">`;
                            html += `${mesh_name}: <span style="color: ${statusColor};">${statusText}</span>`;
                            html += `</div>`;

                            if (num_open_edges > 0) {
                                html += `<div style="color: #888; font-size: 9px; margin-bottom: 4px;">`;
                                html += `${num_boundary_faces} face(s) with open edges, ${num_boundary_vertices} boundary vertices`;
                                html += `</div>`;

                                // Face table
                                html += `<table style="width: 100%; border-collapse: collapse; font-size: 9px;">`;
                                html += `<tr style="color: #888; border-bottom: 1px solid #333;">`;
                                html += `<th style="text-align: left; padding: 2px 4px;">Face</th>`;
                                html += `<th style="text-align: right; padding: 2px 4px;">Open Edges</th>`;
                                html += `<th style="text-align: left; padding: 2px 4px;">Vertices</th>`;
                                html += `</tr>`;

                                // Show faces (limit to 20 for UI performance)
                                const displayFaces = faces.slice(0, 20);
                                for (const face of displayFaces) {
                                    const edgeColor = face.open_edges >= 2 ? '#f66' : '#f96';

                                    html += `<tr style="border-bottom: 1px solid #222;">`;
                                    html += `<td style="padding: 2px 4px; color: #6cf;">${face.id}</td>`;
                                    html += `<td style="text-align: right; padding: 2px 4px; color: ${edgeColor};">${face.open_edges}</td>`;
                                    html += `<td style="padding: 2px 4px; color: #888;">[${face.vertices.join(', ')}]</td>`;
                                    html += `</tr>`;
                                }

                                if (faces.length > 20) {
                                    html += `<tr><td colspan="3" style="padding: 4px; color: #888; text-align: center;">`;
                                    html += `... and ${faces.length - 20} more faces`;
                                    html += `</td></tr>`;
                                }

                                html += `</table>`;
                            } else {
                                html += `<div style="color: #888; font-size: 9px;">`;
                                html += `Total: ${total_vertices.toLocaleString()} vertices, ${total_faces.toLocaleString()} faces`;
                                html += `</div>`;
                            }

                            html += `</div>`;
                        }

                        infoPanel.innerHTML = html;

                        // Resize widget based on content
                        const numRows = message.open_edges_data[0]?.faces?.length || 0;
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
