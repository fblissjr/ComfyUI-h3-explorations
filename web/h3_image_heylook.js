import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Live catalog & autofill for MiniMaxH3ImagePromptExpander:
// - Fetches local system prompts plus live heylook /v1/models, /v1/system/metrics, and /v1/presets
//   through ComfyUI's same-origin /h3_image/expander_catalog route (heylook serves no CORS headers).
// - Upgrades 'model' and 'preset' to dropdowns with "" first so unpinned auto-selection stays default.
// - Populates the editable 'custom_system_prompt' box and sampler widgets whenever the user selects
//   a mode, preset, or model (or clicks the populate button), without clobbering saved workflow edits.

const NODE_ID = "MiniMaxH3ImagePromptExpander";
const ROUTE = "/h3_image/expander_catalog";
const DEBOUNCE_MS = 400;

const SAMPLER_NUMERIC_KEYS = [
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "repetition_penalty",
    "presence_penalty",
];

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function hasWiredImages(node) {
    if (!Array.isArray(node.inputs)) return false;
    return node.inputs.some(
        (inp) => inp && typeof inp.name === "string" && inp.name.startsWith("images.") && inp.link != null,
    );
}

function withSuppressedAutofill(node, fn) {
    const prev = Boolean(node.__h3SuppressAutofill);
    node.__h3SuppressAutofill = true;
    try {
        return fn();
    } finally {
        node.__h3SuppressAutofill = prev;
    }
}

function appendAspectRatioFooter(text, hasImages, footers) {
    const clean = (text || "").trim();
    if (clean.toLowerCase().includes("aspect_ratio:")) return clean;
    const footer = (hasImages ? footers?.edit : footers?.t2i) || "";
    if (!footer) return clean;
    return clean ? `${clean}\n\n${footer}` : footer;
}

function upgradeToCombo(node, widgetName, rawValues, onUserChange) {
    const existing = findWidget(node, widgetName);
    if (!existing) return null;
    const current = existing.value ?? "";
    const unique = [];
    const seen = new Set([""]);
    if (current && !seen.has(current)) {
        unique.push(current);
        seen.add(current);
    }
    for (const v of rawValues) {
        if (v && !seen.has(v)) {
            unique.push(v);
            seen.add(v);
        }
    }
    const values = ["", ...unique];

    if (existing.type === "combo") {
        existing.options.values = values;
        if (!values.includes(existing.value)) {
            withSuppressedAutofill(node, () => {
                existing.value = "";
            });
        }
        node.setDirtyCanvas(true, true);
        return existing;
    }

    const idx = node.widgets.indexOf(existing);
    const label = existing.label || widgetName;
    const combo = node.addWidget(
        "combo",
        widgetName,
        values.includes(current) ? current : "",
        function (val) {
            node.setDirtyCanvas(true, true);
            if (!node.__h3SuppressAutofill && onUserChange) {
                onUserChange(val);
            }
        },
        { values },
    );
    combo.label = label;
    node.widgets[idx] = combo;
    node.widgets.pop();
    node.setDirtyCanvas(true, true);
    return combo;
}

function resolveActiveModelRow(node, catalog) {
    const models = catalog?.heylook?.models || [];
    if (!models.length) return null;
    const wanted = (findWidget(node, "model")?.value || "").trim();
    if (wanted) {
        return models.find((m) => m.id === wanted) || null;
    }
    const needVision = hasWiredImages(node);
    const fit = needVision
        ? models.filter((m) => Array.isArray(m.capabilities) && m.capabilities.includes("vision"))
        : models;
    const pool = fit.length ? fit : models;
    const resident = new Set(catalog?.heylook?.resident_models || []);
    return pool.find((m) => resident.has(m.id)) || pool[0] || null;
}

function findPresetEntry(presetValue, catalog) {
    const presets = catalog?.heylook?.presets || [];
    const wanted = (presetValue || "").trim();
    if (!wanted) return null;
    const lower = wanted.toLowerCase();
    return (
        presets.find((p) => p.id === wanted || p.name === wanted) ||
        presets.find((p) => (p.name || "").toLowerCase() === lower) ||
        null
    );
}

function applySamplerFields(node, catalog, { flipToCustom = false } = {}) {
    const modelRow = resolveActiveModelRow(node, catalog);
    const presetVal = findWidget(node, "preset")?.value || "";
    const presetEntry = findPresetEntry(presetVal, catalog);

    const defaults = modelRow?.sampler_defaults || {};
    const params = presetEntry?.params || {};
    const merged = { ...defaults, ...params };

    withSuppressedAutofill(node, () => {
        for (const key of SAMPLER_NUMERIC_KEYS) {
            if (merged[key] === undefined || merged[key] === null) continue;
            const w = findWidget(node, key);
            if (w) w.value = merged[key];
        }
        const effortW = findWidget(node, "reasoning_effort");
        if (effortW) {
            effortW.value = typeof params.reasoning_effort === "string" ? params.reasoning_effort : "";
        }
        if (typeof params.enable_thinking === "boolean") {
            const thinkW = findWidget(node, "thinking");
            if (thinkW) thinkW.value = params.enable_thinking ? "on" : "off";
        }
        if (flipToCustom) {
            const smW = findWidget(node, "sampler_mode");
            if (smW) smW.value = "custom";
        }
    });
    node.setDirtyCanvas(true, true);
}

function applySystemPromptForSelection(node, catalog, { forceAuto = false } = {}) {
    if (!catalog) return;
    const modeW = findWidget(node, "mode");
    const sysW = findWidget(node, "custom_system_prompt");
    const presetW = findWidget(node, "preset");
    if (!modeW || !sysW) return;

    const mode = modeW.value;
    const hasImages = hasWiredImages(node);
    const localPrompts = catalog.local_prompts || {};

    let nextText = null;
    if (Object.prototype.hasOwnProperty.call(localPrompts, mode)) {
        nextText = localPrompts[mode];
    } else if (mode === "auto") {
        if (forceAuto) {
            nextText = localPrompts[hasImages ? "edit + aspect_ratio" : "t2i + aspect_ratio"] || "";
        } else if (sysW.value && sysW.value === node.__h3LastAutoPrompt) {
            nextText = "";
        }
    } else if (mode === "heylook preset" || mode === "heylook preset + aspect_ratio") {
        const entry = findPresetEntry(presetW?.value, catalog);
        if (entry) {
            nextText =
                mode === "heylook preset + aspect_ratio"
                    ? appendAspectRatioFooter(entry.system_prompt, hasImages, catalog.aspect_ratio_footers)
                    : (entry.system_prompt || "").trim();
        }
    } else if (mode === "custom" && forceAuto && presetW?.value) {
        const entry = findPresetEntry(presetW.value, catalog);
        if (entry?.system_prompt) {
            nextText = entry.system_prompt.trim();
        }
    }

    if (nextText !== null) {
        withSuppressedAutofill(node, () => {
            sysW.value = nextText;
            node.__h3LastAutoPrompt = nextText;
        });
        node.setDirtyCanvas(true, true);
    }
}

async function fetchCatalog(node) {
    if (node.__h3FetchAbort) {
        node.__h3FetchAbort.abort();
    }
    const controller = new AbortController();
    node.__h3FetchAbort = controller;

    const baseUrl = (findWidget(node, "base_url")?.value || "").trim();
    const url = `${ROUTE}?base_url=${encodeURIComponent(baseUrl)}`;
    try {
        const resp = await api.fetchApi(url, { signal: controller.signal });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        node.__h3Catalog = data;
        return data;
    } catch (err) {
        if (err?.name !== "AbortError") {
            console.warn(`[${NODE_ID}] catalog fetch failed:`, err);
        }
        return node.__h3Catalog || null;
    }
}

async function refreshDropdowns(node, handlers) {
    const catalog = await fetchCatalog(node);
    if (!catalog) return null;
    if (catalog.heylook?.reachable) {
        const modelIds = (catalog.heylook.models || []).map((m) => m.id);
        const presetNames = (catalog.heylook.presets || []).map((p) => p.name || p.id);
        upgradeToCombo(node, "model", modelIds, handlers.onModelChange);
        upgradeToCombo(node, "preset", presetNames, handlers.onPresetChange);
    }
    return catalog;
}

app.registerExtension({
    name: "h3_explorations.heylook_prompt_expander",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            return withSuppressedAutofill(this, () =>
                origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined,
            );
        };

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            const labels = {
                custom_system_prompt: "system_prompt  ·  editable",
                preset: "preset  ·  heylook /v1/presets",
            };
            for (const w of node.widgets || []) {
                if (labels[w.name]) w.label = labels[w.name];
            }

            const handlers = {
                onModeChange: async () => {
                    const catalog = node.__h3Catalog || (await refreshDropdowns(node, handlers));
                    if (!catalog) return;
                    applySystemPromptForSelection(node, catalog, { forceAuto: false });
                    const modeVal = findWidget(node, "mode")?.value;
                    if (modeVal === "heylook preset" || modeVal === "heylook preset + aspect_ratio") {
                        const entry = findPresetEntry(findWidget(node, "preset")?.value, catalog);
                        const hasSamplerParams = entry && Object.keys(entry.params || {}).length > 0;
                        applySamplerFields(node, catalog, { flipToCustom: Boolean(hasSamplerParams) });
                    }
                },
                onPresetChange: async (presetVal) => {
                    const catalog = node.__h3Catalog || (await refreshDropdowns(node, handlers));
                    if (!catalog) return;
                    const trimmed = (presetVal || "").trim();
                    if (!trimmed) {
                        applySamplerFields(node, catalog, { flipToCustom: false });
                        return;
                    }
                    const hasImages = hasWiredImages(node);
                    const entry = findPresetEntry(trimmed, catalog);
                    if (entry?.system_prompt) {
                        const modeW = findWidget(node, "mode");
                        if (modeW && modeW.value === "auto") {
                            withSuppressedAutofill(node, () => {
                                modeW.value = "heylook preset + aspect_ratio";
                            });
                        }
                        const activeMode = modeW?.value || "custom";
                        const sysW = findWidget(node, "custom_system_prompt");
                        if (sysW) {
                            const filled =
                                activeMode === "heylook preset + aspect_ratio"
                                    ? appendAspectRatioFooter(
                                          entry.system_prompt,
                                          hasImages,
                                          catalog.aspect_ratio_footers,
                                      )
                                    : entry.system_prompt.trim();
                            withSuppressedAutofill(node, () => {
                                sysW.value = filled;
                                node.__h3LastAutoPrompt = filled;
                            });
                        }
                    }
                    const hasSamplerParams = entry && Object.keys(entry.params || {}).length > 0;
                    applySamplerFields(node, catalog, { flipToCustom: Boolean(hasSamplerParams) });
                },
                onModelChange: async () => {
                    const catalog = node.__h3Catalog || (await refreshDropdowns(node, handlers));
                    if (!catalog) return;
                    if (findWidget(node, "sampler_mode")?.value !== "custom") {
                        applySamplerFields(node, catalog, { flipToCustom: false });
                    }
                },
            };

            const wrapWidget = (name, fn) => {
                const w = findWidget(node, name);
                if (!w) return;
                const orig = w.callback;
                w.callback = function (val) {
                    if (orig) orig.apply(this, arguments);
                    if (!node.__h3SuppressAutofill) fn(val);
                };
            };

            wrapWidget("mode", handlers.onModeChange);
            wrapWidget("preset", handlers.onPresetChange);
            wrapWidget("model", handlers.onModelChange);

            let urlTimer = null;
            wrapWidget("base_url", () => {
                clearTimeout(urlTimer);
                urlTimer = setTimeout(() => {
                    refreshDropdowns(node, handlers);
                }, DEBOUNCE_MS);
            });

            for (const key of [...SAMPLER_NUMERIC_KEYS, "reasoning_effort"]) {
                wrapWidget(key, () => {
                    const smW = findWidget(node, "sampler_mode");
                    if (smW && smW.value !== "custom") {
                        withSuppressedAutofill(node, () => {
                            smW.value = "custom";
                        });
                        node.setDirtyCanvas(true, true);
                    }
                });
            }

            const btn = node.addWidget(
                "button",
                "populate_from_source",
                "↻ Populate prompt & sampler from mode / heylook",
                async () => {
                    const catalog = await refreshDropdowns(node, handlers);
                    if (!catalog) return;
                    applySystemPromptForSelection(node, catalog, { forceAuto: true });
                    const presetVal = (findWidget(node, "preset")?.value || "").trim();
                    const entry = findPresetEntry(presetVal, catalog);
                    const hasSamplerParams = entry && Object.keys(entry.params || {}).length > 0;
                    applySamplerFields(node, catalog, { flipToCustom: Boolean(hasSamplerParams) });
                },
            );
            btn.serialize = false;

            setTimeout(() => {
                refreshDropdowns(node, handlers);
            }, 0);

            return ret;
        };
    },
});
