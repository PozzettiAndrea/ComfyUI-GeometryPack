/**
 * ComfyUI GeomPack - Ultimate Mesh Inspection
 * Embedded VTK.js viewer + clickable statistics table. Every defect is baked as a
 * scalar field on the exported VTP; clicking a stat row reloads the viewer with that
 * field as the active scalar (selectedField), lighting it up.
 */

import { app } from "../../../scripts/app.js";

// Auto-detect extension folder name (same trick as the other viewers).
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-GeometryPack";
})();

app.registerExtension({
    name: "geompack.ultimate_mesh_inspection",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GeomPackUltimateMeshInspection") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            node._umiFilename = null;
            node._umiActiveField = null;

            const container = document.createElement("div");
            container.style.cssText =
                "width:100%;height:100%;display:flex;flex-direction:column;background:#222;overflow:hidden;";

            // --- title / status bar ---
            const bar = document.createElement("div");
            bar.style.cssText =
                "display:flex;align-items:center;gap:8px;padding:4px 8px;background:#1a1a1a;" +
                "border-bottom:1px solid #444;font:11px monospace;color:#bbb;flex:0 0 auto;";
            const title = document.createElement("div");
            title.textContent = "Ultimate Mesh Inspection";
            title.style.cssText = "font-weight:bold;color:#9df;";
            const status = document.createElement("div");
            status.style.cssText = "flex:1;color:#888;text-align:right;";
            status.textContent = "(run to inspect)";
            const clearBtn = document.createElement("button");
            clearBtn.textContent = "clear highlight";
            clearBtn.style.cssText =
                "font:10px monospace;padding:2px 6px;border:1px solid #555;border-radius:3px;" +
                "background:#333;color:#ccc;cursor:pointer;";
            bar.appendChild(title); bar.appendChild(status); bar.appendChild(clearBtn);

            // --- vtk.js iframe viewer ---
            const iframe = document.createElement("iframe");
            iframe.style.cssText = "width:100%;flex:1 1 0;min-height:0;border:none;background:#2a2a2a;";
            iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_vtk.html?v=` + Date.now();
            let iframeLoaded = false;
            iframe.addEventListener("load", () => { iframeLoaded = true; });

            // --- scrollable statistics table ---
            const tableWrap = document.createElement("div");
            tableWrap.style.cssText =
                "flex:0 0 auto;max-height:260px;overflow:auto;background:#1a1a1a;border-top:1px solid #444;" +
                "font:11px monospace;color:#ccc;";
            const table = document.createElement("table");
            table.style.cssText = "width:100%;border-collapse:collapse;";
            tableWrap.appendChild(table);

            container.appendChild(bar);
            container.appendChild(iframe);
            container.appendChild(tableWrap);

            const widget = this.addDOMWidget("ultimate_inspection", "UMI", container, {
                getValue() { return ""; }, setValue() {},
            });
            widget.computeSize = () => [520, 720];
            this.setSize([520, 720]);

            // --- helpers ---
            const loadMesh = (selectedField, focus) => {
                if (!node._umiFilename || !iframe.contentWindow) return;
                const filepath = `/view?filename=${encodeURIComponent(node._umiFilename)}&type=output&subfolder=`;
                iframe.contentWindow.postMessage({
                    type: "LOAD_MESH",
                    filepath,
                    selectedField: selectedField || null,
                    timestamp: Date.now(),
                }, "*");
                node._umiActiveField = selectedField || null;
                // for "worst" rows: zoom the camera onto that single face after it loads
                if (focus && focus.length === 3) {
                    setTimeout(() => {
                        iframe.contentWindow?.postMessage({
                            type: "FOCUS_ON_POINT", point: focus, timestamp: Date.now(),
                        }, "*");
                    }, 500);
                }
            };

            clearBtn.addEventListener("click", () => {
                loadMesh(null, null);
                table.querySelectorAll("tr.umi-active").forEach(tr => tr.classList.remove("umi-active"));
            });

            const buildTable = (rows) => {
                table.innerHTML = "";
                let curGroup = null;
                rows.forEach((row) => {
                    if (row.group !== curGroup) {
                        curGroup = row.group;
                        const gtr = document.createElement("tr");
                        const gtd = document.createElement("td");
                        gtd.colSpan = 2;
                        gtd.textContent = curGroup;
                        gtd.style.cssText =
                            "padding:4px 8px 2px;color:#9df;font-weight:bold;border-top:1px solid #333;";
                        gtr.appendChild(gtd); table.appendChild(gtr);
                    }
                    const tr = document.createElement("tr");
                    const tdL = document.createElement("td");
                    tdL.textContent = row.label;
                    tdL.style.cssText = "padding:2px 8px;color:#999;white-space:nowrap;";
                    const tdV = document.createElement("td");
                    tdV.textContent = row.value;
                    tdV.style.cssText = "padding:2px 8px;color:#eee;width:100%;";
                    tr.appendChild(tdL); tr.appendChild(tdV);
                    if (row.field) {
                        tr.title = `highlight: ${row.field}`;
                        tr.style.cursor = "pointer";
                        tdL.style.color = "#cfc";
                        tr.addEventListener("mouseenter", () => {
                            if (!tr.classList.contains("umi-active")) tr.style.background = "#2c2c2c";
                        });
                        tr.addEventListener("mouseleave", () => {
                            if (!tr.classList.contains("umi-active")) tr.style.background = "";
                        });
                        tr.addEventListener("click", () => {
                            table.querySelectorAll("tr.umi-active").forEach(x => {
                                x.classList.remove("umi-active"); x.style.background = "";
                            });
                            tr.classList.add("umi-active");
                            tr.style.background = "#264";
                            loadMesh(row.field, row.focus);
                        });
                    }
                    table.appendChild(tr);
                });
            };

            // --- on execute ---
            const onExecuted = this.onExecuted;
            this.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const rows = message?.report?.[0] || [];
                buildTable(rows);

                const vc = message?.vertex_count?.[0] ?? 0;
                const fc = message?.face_count?.[0] ?? 0;
                const wt = message?.is_watertight?.[0];
                status.textContent = `V ${vc.toLocaleString?.() ?? vc} | F ${fc.toLocaleString?.() ?? fc} | ` +
                    `watertight ${wt ? "yes" : "no"}  — click a green row to highlight`;

                const fn = message?.mesh_file?.[0];
                if (fn) {
                    node._umiFilename = fn;
                    const fire = () => loadMesh(null);
                    if (iframeLoaded) fire(); else setTimeout(fire, 600);
                }
            };

            return r;
        };
    },
});
