/**
 * ComfyUI GeomPack - Generic Text Report Renderer
 * Collapsible info panel. Arrow toggle at top of box.
 */

import { app } from "../../../scripts/app.js";
import { buildTextReportHTML } from "./utils/analysisPanel.js";

const TEXT_REPORT_NODES = [
    // CADabra
    "CADUltimateInspection",
    // Analysis
    "GeomPackMeshInfo",
    "GeomPackMeshQuality",
    "GeomPackDegenerateFaces",
    "GeomPackConnectedComponents",
    "GeomPackOpenEdges",
    "GeomPackComputeCurvature",
    "GeomPackSegmentByCurvature",
    "GeomPackComputeFilletField",
    "GeomPackSegmentPatches",
    "GeomPackClusterFaces",
    // Repair
    "GeomPackFillHoles",
    "GeomPackMeshFix",
    "GeomPackCheckNormals",
    "GeomPackFixNormals",
    "GeomPackMeshRepair",
    "GeomPackRemoveDegenerateFaces",
    "GeomPackMergeVertices",
    "GeomPackVisualizeNormals",
    "GeomPackAddNormalsToPointCloud",
    "GeomPackFixSelfIntersectionsByRemoval",
    "GeomPackDetectSelfIntersections",
    "GeomPackFixSelfIntersectionsByPerturbation",
    "GeomPackRemeshSelfIntersections",
    // Smoothing / Decimation (unified dispatcher front nodes — info forwarded
    // from the hidden backend via the executed event's display_node)
    "GeomPackSharpenMesh",
    "GeomPackDecimateMesh",
    // Remeshing
    "GeomPackRemesh",
    "GeomPackRemeshCGAL",
    "GeomPackRemeshBlender",
    "GeomPackRemeshGPU",
    "RefineMesh",
    "GeomPackEdgeFlip",
    // Reconstruction
    "GeomPackReconstructSurface",
    "GeomPackAlphaWrap",
    // Texture remeshing
    "GeomPackTextureToGeometry",
    "GeomPackDepthNormalsToMesh",
    "GeomPackRemeshWithTexture",
    // Boolean
    "GeomPackBooleanCGAL",
    "GeomPackBooleanBlender",
    // Combine
    "GeomPackCombineMeshes",
    "GeomPackCombineMeshesBatch",
    "GeomPackSplitByField",
    // Transforms
    "GeomPackTransformMesh",
    "GeomPackNormalizeMeshToBBox",
    // UV
    "GeomPackUVUnwrap",
    // Distance
    "GeomPackPointToMeshDistance",
    "GeomPackMeshToMeshDistance",
    // IO
    "GeomPackLoadMeshFBX",
    "GeomPackLoadMeshBlend",
    // CADabra
    "CADEdgeAnalysis",
    "CAD_Quality_Metrics",
    "CADCurvature",
    "CADSewFaces",
    "CADCheckOverlappingFaces",
    "CADSplitComponents",
    "CADHealShape",
    "CADFixDegenerateFaces",
    "CADPrimitiveReconstruction",
    "CADFaceAnalysis",
    // MeshSegmenter
    "MeshSegGenerateMasks",
    "MeshSegLift2DTo3DLabels",
    "MeshSegSmoothLabels",
    "MeshSegApplyLabelsToMesh",
];

const TOGGLE_HEIGHT = 28;
const DEFAULT_CONTENT_HEIGHT = 60;

app.registerExtension({
    name: "geompack.text_report",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!TEXT_REPORT_NODES.includes(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            let expanded = false;
            let contentHeight = DEFAULT_CONTENT_HEIGHT;
            let widgetHeight = TOGGLE_HEIGHT;
            const node = this;

            // --- Single container widget ---
            const container = document.createElement("div");
            container.style.cssText = "background:#1a1a2e; border:1px solid #333; border-radius:4px; overflow:hidden; display:flex; flex-direction:column; height:100%;";

            // Header bar (always visible, top of box)
            const header = document.createElement("div");
            header.style.cssText = "display:flex; align-items:center; gap:6px; padding:2px 8px; cursor:pointer; user-select:none; font-size:11px; color:#888; flex-shrink:0;";

            const arrow = document.createElement("span");
            arrow.style.cssText = "font-size:8px; display:inline-block; transition:transform 0.15s;";
            arrow.textContent = "\u25B6";

            const label = document.createElement("span");
            label.textContent = "Info";

            header.appendChild(arrow);
            header.appendChild(label);

            // Content area (hidden when collapsed)
            const content = document.createElement("div");
            content.style.cssText = "padding:6px 8px; border-top:1px solid #333; font-size:11px; color:#ccc; display:none; overflow-y:auto; flex:1; min-height:0;";

            container.appendChild(header);
            container.appendChild(content);

            // Use getMinHeight/getHeight — the API ComfyUI's DOMWidgetImpl actually reads
            const widget = this.addDOMWidget("report_panel", "REPORT_PANEL", container, {
                getValue() { return expanded ? "true" : "false"; },
                setValue(v) { if (v === "true") setExpanded(true); },
                getMinHeight: () => widgetHeight,
                getHeight: () => widgetHeight,
            });

            // Force node to grow to fit the widget
            requestAnimationFrame(() => {
                node.setSize([node.size[0], node.computeSize()[1]]);
                node.setDirtyCanvas(true, true);
            });

            function resize() {
                widgetHeight = expanded ? TOGGLE_HEIGHT + contentHeight : TOGGLE_HEIGHT;
                node.setSize([node.size[0], node.computeSize()[1]]);
                node.setDirtyCanvas(true, true);
            }

            function setExpanded(val) {
                expanded = val;
                content.style.display = expanded ? "block" : "none";
                arrow.style.transform = expanded ? "rotate(90deg)" : "";
                resize();
            }

            // Toggle click
            header.addEventListener("click", () => {
                setExpanded(!expanded);
            });

            // Execution results
            const onExecuted = this.onExecuted;
            this.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);

                if (message?.text && message.text.length > 0) {
                    const text = message.text[0];
                    content.innerHTML = buildTextReportHTML(text);

                    const firstLine = text.split("\n")[0].trim();
                    if (firstLine) label.textContent = firstLine;

                    resize();
                }
            };

            return r;
        };
    }
});
