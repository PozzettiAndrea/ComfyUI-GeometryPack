// Preview Mesh Batch Render: show the edge_opacity widget only when show_edges
// is enabled (collapse it otherwise).
import { app } from "../../../scripts/app.js";

const TAG = "[PreviewMeshBatchRender]";
console.log(`${TAG} script loaded`);

const findW = (node, name) => (node.widgets || []).find((w) => w.name === name);

function toggleWidget(w, show) {
    if (!w) return;
    if (!w.__gpOrig) w.__gpOrig = { type: w.type, computeSize: w.computeSize };
    if (show) {
        w.type = w.__gpOrig.type;
        w.computeSize = w.__gpOrig.computeSize;
    } else {
        w.type = "geompack_hidden";      // unknown type -> not drawn
        w.computeSize = () => [0, -4];    // collapse (cancels the 4px row gap)
    }
}

app.registerExtension({
    name: "geompack.previewmeshbatchrender.ui",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GeomPackPreviewMeshBatchRender") return;
        console.log(`${TAG} registering for GeomPackPreviewMeshBatchRender`);

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;
            const showEdges = findW(node, "show_edges");
            const edgeOpacity = findW(node, "edge_opacity");

            const update = () => {
                toggleWidget(edgeOpacity, !!(showEdges && showEdges.value));
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            };

            if (showEdges) {
                const cb = showEdges.callback;
                showEdges.callback = function () {
                    const ret = cb?.apply(this, arguments);
                    update();
                    return ret;
                };
            }
            setTimeout(update, 0);  // apply initial state (edges off -> hidden)
            return r;
        };
    },
});
