import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "MiniMaxH3AutoEditPrompt";
const ROUTE = "/h3_auto_edit/catalog";

function findWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

function upgradeToCombo(node, name, values) {
    const widget = findWidget(node, name);
    if (!widget) return;
    const unique = [];
    const seen = new Set();
    for (const v of ["", ...(values || [])]) {
        const s = typeof v === "string" ? v : String(v ?? "");
        if (!seen.has(s)) {
            seen.add(s);
            unique.push(s);
        }
    }
    const cur = typeof widget.value === "string" ? widget.value : "";
    if (cur && !seen.has(cur)) {
        unique.push(cur);
    }
    widget.type = "combo";
    widget.options = { ...(widget.options || {}), values: unique };
    node.setDirtyCanvas(true, true);
}

function updateConceptCombo(node, concepts) {
    const widget = findWidget(node, "concept");
    if (!widget || !Array.isArray(concepts) || concepts.length === 0) return;
    const values = [...concepts];
    const cur = typeof widget.value === "string" ? widget.value : "";
    if (cur && !values.includes(cur)) {
        values.push(cur);
    }
    widget.options = { ...(widget.options || {}), values };
    node.setDirtyCanvas(true, true);
}

async function refreshCatalog(node) {
    if (node._h3AutoEditBusy) return;
    node._h3AutoEditBusy = true;
    try {
        const baseWidget = findWidget(node, "base_url");
        const baseUrl = typeof baseWidget?.value === "string" ? baseWidget.value.trim() : "";
        const qs = baseUrl ? `?base_url=${encodeURIComponent(baseUrl)}` : "";
        const resp = await api.fetchApi(`${ROUTE}${qs}`);
        if (!resp.ok) return;
        const data = await resp.json();
        updateConceptCombo(node, data?.concepts);
        const hl = data?.heylook;
        if (!hl || !hl.reachable) return;
        const modelIds = (hl.models || []).map((m) => m.id).filter(Boolean);
        const presetNames = (hl.presets || []).map((p) => p.name || p.id).filter(Boolean);
        upgradeToCombo(node, "model", modelIds);
        upgradeToCombo(node, "preset", presetNames);
    } catch (_err) {
        // Keep blank defaults intact if heylook is offline; never inject host IPs into base_url.
    } finally {
        node._h3AutoEditBusy = false;
    }
}

app.registerExtension({
    name: "h3_explorations.auto_edit_heylook",
    nodeCreated(node) {
        if (node.comfyClass !== NODE_ID) return;

        const baseWidget = findWidget(node, "base_url");
        if (baseWidget && !baseWidget._h3AutoEditHooked) {
            baseWidget._h3AutoEditHooked = true;
            const origCb = baseWidget.callback;
            let timer = null;
            baseWidget.callback = function (...args) {
                const res = origCb ? origCb.apply(this, args) : undefined;
                if (timer) clearTimeout(timer);
                timer = setTimeout(() => {
                    refreshCatalog(node);
                }, 450);
                return res;
            };
        }

        if (!findWidget(node, "↻ Refresh heylook models & presets")) {
            const btn = node.addWidget("button", "↻ Refresh heylook models & presets", null, () => {
                refreshCatalog(node);
            });
            btn.serialize = false;
        }

        refreshCatalog(node);
    },
});
