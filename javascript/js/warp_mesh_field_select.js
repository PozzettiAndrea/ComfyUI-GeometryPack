/**
 * ComfyUI GeomPack - Warp Mesh field selector.
 *
 * The Warp Mesh node (GeomPackWarpMesh) is an output node: running it reports the
 * candidate 3-component vertex fields. This extension renders them as clickable
 * chips in a box under the node; clicking one writes it into the `field_name`
 * widget. Re-run to apply the warp.
 */

import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "geompack.warp_mesh_field_select",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GeomPackWarpMesh") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            const container = document.createElement("div");
            container.style.cssText =
                "width:100%;box-sizing:border-box;padding:6px;background:#1c1c1c;" +
                "border:1px solid #444;border-radius:4px;font:11px monospace;color:#bbb;" +
                "display:flex;flex-direction:column;gap:6px;overflow:auto;";

            const status = document.createElement("div");
            status.style.cssText = "color:#888;flex:0 0 auto;";
            status.textContent = "Run to list 3-D vertex fields.";

            const chips = document.createElement("div");
            chips.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;align-content:flex-start;";

            container.appendChild(status);
            container.appendChild(chips);

            const widget = node.addDOMWidget("warp_fields", "WARP_FIELDS", container, {
                getValue() { return ""; }, setValue() {},
            });
            widget.computeSize = () => [Math.max(node.size?.[0] || 240, 240), 150];

            const fieldWidget = () => node.widgets?.find((w) => w.name === "field_name");

            const render = (fields, selected) => {
                chips.innerHTML = "";
                if (!fields || fields.length === 0) {
                    status.textContent = "No 3-D (n_vertices, 3) vertex fields on this mesh.";
                    return;
                }
                status.textContent = "Click a field to warp by it:";
                fields.forEach((name) => {
                    const chip = document.createElement("button");
                    chip.textContent = name;
                    const isSel = name === selected;
                    chip.style.cssText =
                        "font:11px monospace;padding:2px 8px;border-radius:10px;cursor:pointer;" +
                        "border:1px solid " + (isSel ? "#6c6" : "#555") + ";" +
                        "background:" + (isSel ? "#264" : "#2b2b2b") + ";" +
                        "color:" + (isSel ? "#cfc" : "#ddd") + ";";
                    chip.addEventListener("mouseenter", () => {
                        if (name !== selected) chip.style.background = "#383838";
                    });
                    chip.addEventListener("mouseleave", () => {
                        if (name !== selected) chip.style.background = "#2b2b2b";
                    });
                    chip.addEventListener("click", () => {
                        const w = fieldWidget();
                        if (w) {
                            w.value = name;
                            w.callback?.(name);
                        }
                        render(fields, name);
                        node.setDirtyCanvas(true, true);
                    });
                    chips.appendChild(chip);
                });
            };

            const onExecuted = node.onExecuted;
            node.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const fields = message?.fields?.[0] || [];
                const selected = message?.selected?.[0] || (fieldWidget()?.value ?? "");
                const warped = message?.warped?.[0];
                render(fields, selected);
                if (warped) status.textContent = "Warped by '" + selected + "'. Pick another to change.";
            };

            return r;
        };
    },
});
