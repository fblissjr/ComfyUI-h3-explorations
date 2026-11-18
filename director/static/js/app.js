/**
 * Interactive Spatial Director — Main Controller (`app.js`).
 *
 * Hydrates all options from GET /api/schema & GET /api/system_stats, manages Beat 1
 * Anchor upload/generation, drives 24 fps loop + 3f FoleySynth playback, connects
 * live preview WS frames, runs accessible tablist navigation in the Engine drawer,
 * and coordinates A|B wipe comparison & Clean/HUD MP4 exports.
 *
 * Zero external URLs, zero inline handlers, zero element.style.* mutations,
 * zero hardcoded sound presets or scene suggestions.
 */

import { FoleySynth, snapFrameTo3f } from "./foley_synth.js";
import { DirectorWSClient } from "./ws_client.js";
import { ABCompareController, videoItemViewUrl } from "./ab_compare.js";
import { SequencerController, causalStepBoundaries } from "./sequencer.js";
import { createTakesController, videoItemRef } from "./takes_ui.js";
import {
  StageController,
  clamp,
  computeCouplingInfo,
  evalPortalAtFrame,
  evalSpriteAtFrame,
  evalTargetTemporalWeight,
} from "./stage.js";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const FPS = 24;
const INT32_TOKEN_LIMIT = 99864;

const state = {
  schema: null,
  lengths: [],
  soundPresetMap: {},
  soundPresetsList: [],
  canvases: [],
  workflows: [],
  anchorWorkflows: [],
  imageEditWorkflows: [],
  controlVectors: [],
  capabilities: {
    kj_preview_override: false,
    taeh3: false,
    taeh3_image: false,
    heylook_suggestions: false,
    sam3: false,
    sam3_engine: "grabcut",
    spatial_combined: true,
    audio_refine: true,
  },

  length: 141,
  frame: 0,
  playing: false,
  loop: true,
  showPatchGrid: false,
  spriteMode: false,
  settingSpritePath: false,
  snapToIncludeMask: false,
  selectedTargetIdx: 0,
  samPicking: false,
  samRole: "include",
  samSegmenting: false,
  manualDockSide: "",
  samEngine: "",
  spriteMaskRef: "",
  includeMaskRef: "",
  excludeMaskRef: "",
  includeMaskImg: null,
  excludeMaskImg: null,
  samPosPoints: [],
  samNegPoints: [],
  stageMuted: false,
  lastFrameRef: "",
  lastFrameName: "",
  shotHistory: [],
  parentTakeId: "",
  parentFrame: null,
  activeMaskPreviewTakeId: "",
  vlmSubjectLabel: "",
  residentBackbone: "",
  lanesOpen: false,
  streamsManualOverride: false,
  workflowManualOverride: false,
  unlinkPortalRy: false,
  seqSelection: null,
  portalTouched: false,

  workflowId: "i2v_portal",
  anchorWorkflowId: "h3_image",
  previewWorkflowId: "auto",
  seed: 730451892,
  seedLocked: true,

  hasAnchor: false,
  anchorRef: "",
  anchorName: "",
  anchorMedia: null,
  activeVideoEl: null,
  activeVideoUrl: "",
  activeStageTake: null,

  spriteCutout: null, // { ref, viewUrl, img }
  livePreview: null,  // { kind, media, step, total, fullFrame, fps, avgStepMs }
  maskPreview: null,  // { img, viewUrl, x, y, rx, ry, shape, feather, haloRadius, haloWeight, snapToInclude }
  activeRender: null, // { kind, pid, phase, step, total, startedAt, meta }

  targetPrompt: "",
  originalPrompt: "",
  samePrompt: false,
  audioRefine: false,
  controlVector: "none",
  controlVectorStrength: 1.0,
  couplings: [],
  activeOperator: "aperture",

  editGuidance: {
    scale: 2.0,
    audio_scale: 1.0,
    first_step: 0,
    last_step: 1000,
    streams: "independent",
  },

  doc: {
    mode: "portal_path",
    width: 1344,
    height: 768,
    feather: 0.08,
    invert_mask: false,
    portal_start: null,
    portal_end: null,
    portal_ramp: 0,
    portal: [
      { frame: 0, x: 0.44, y: 0.48, rx: 0.22, ry: 0.22, shape: "circle", label: "REALITY LENS", weight: 1.0, halo_radius: 0.10, halo_weight: 0.30 },
    ],
    targets: [],
    portal_cues: [],
    dpad: {
      waypoints: [{ frame: 0, x: 0.36, y: 0.74, face: "right", scale: 1.0 }],
      hops: [],
    },
  },

  takes: [],
  activeQueuePromptIds: null,
  activeRenderPromptId: "",
  activeBaselinePromptId: "",
  activeAnchorPromptId: "",
  activeStampPromptId: "",
  activeMaskPreviewPromptId: "",

  undoStack: [],
  redoStack: [],
  defaultSoundPreset: "",
};

const foley = new FoleySynth();
let stage = null;
let sequencer = null;
let abCompare = null;
let takesUI = null;
let wsClient = null;
let rafHandle = 0;
let lastRafTime = 0;
let toastTimer = 0;
let preflightTimer = 0;
let renderStatusTimer = 0;
let lastPollTakesAt = 0;

function updateRenderStatusOverlay() {
  const banner = $("#render-status-overlay");
  if (!banner) return;
  const ar = state.activeRender;
  if (!ar) {
    banner.classList.add("hidden");
    return;
  }
  banner.classList.remove("hidden");
  const elapsedMs = ar.startedEpochMs
    ? Math.max(0, Date.now() - ar.startedEpochMs)
    : Math.max(0, performance.now() - (ar.startedAt || performance.now()));
  const elapsedSec = (elapsedMs / 1000).toFixed(1);
  const total = Math.max(0, Number(ar.total || 0));
  const step = Math.max(0, Number(ar.step || 0));
  const phaseEl = $("#render-status-phase");
  const metaEl = $("#render-status-meta");
  const fillEl = $("#render-progress-fill");
  const paceMs = Number(ar.avgStepMs || ar.stepMs || 0);
  const paceStr = paceMs > 0 ? ` · ${(paceMs / 1000).toFixed(1)}s/st` : "";
  const hasDelta = ar.delta !== null && ar.delta !== undefined && Number.isFinite(Number(ar.delta));
  const deltaStr = hasDelta ? ` · Δ ${Number(ar.delta).toFixed(2)}` : "";
  if (phaseEl) phaseEl.textContent = ar.phase || "Running on GPU…";
  if (metaEl) metaEl.textContent = total > 0 ? `Step ${step}/${total}${paceStr}${deltaStr} · ${elapsedSec}s` : `${elapsedSec}s`;
  if (fillEl) {
    const pct = total > 0 ? clamp((step / total) * 100, 10, 100) : 22;
    fillEl.setAttribute("width", pct.toFixed(1));
  }
}

function setActiveRender(meta) {
  state.activeRender = {
    kind: meta.kind || "render",
    pid: meta.pid || "",
    phase: meta.phase || "Rendering…",
    step: Number(meta.step || 0),
    total: Number(meta.total || 0),
    stepMs: meta.stepMs ?? null,
    avgStepMs: meta.avgStepMs ?? null,
    delta: meta.delta ?? null,
    meta: meta.meta || null,
    startedAt: performance.now(),
    startedEpochMs: meta.startedEpochMs || null,
  };
  lastPollTakesAt = 0;
  if (renderStatusTimer) clearInterval(renderStatusTimer);
  renderStatusTimer = setInterval(async () => {
    if (!state.activeRender) {
      clearInterval(renderStatusTimer);
      renderStatusTimer = 0;
      return;
    }
    updateRenderStatusOverlay();
    const now = performance.now();
    if (now - lastPollTakesAt >= 2000 && state.activeRender?.pid) {
      lastPollTakesAt = now;
      await checkActivePromptCompletion(state.activeRender.pid);
    }
  }, 500);
  updateRenderStatusOverlay();
}

function clearActiveRender() {
  state.activeRender = null;
  if (renderStatusTimer) {
    clearInterval(renderStatusTimer);
    renderStatusTimer = 0;
  }
  updateRenderStatusOverlay();
}

async function handleFinishedTakeById(pid) {
  const take = state.takes.find((t) => t.prompt_id === pid || t.baseline_prompt_id === pid);
  if (!take) return false;
  const isBaselineEvent = Boolean(take.baseline_prompt_id && take.baseline_prompt_id === pid);
  if (isBaselineEvent) {
    if (take.baseline_video || take.baseline_error) {
      if (pid === state.activeBaselinePromptId) state.activeBaselinePromptId = "";
      if (state.activeRender?.pid === pid) clearActiveRender();
      abCompare?.setTakes({ directedTake: take, baselineTake: findBaselineTake(take) });
      syncAllUI();
      if (take.baseline_error) {
        showToast(`Same-seed baseline failed: ${take.baseline_error}`, "err", 5000);
      } else {
        showToast("Directed + Same-Seed Baseline A|B ready!", "ok");
      }
      return true;
    }
    return false;
  }
  if (take.status === "error") {
    if (pid === state.activeAnchorPromptId) state.activeAnchorPromptId = "";
    if (pid === state.activeMaskPreviewPromptId) state.activeMaskPreviewPromptId = "";
    if (pid === state.activeStampPromptId) state.activeStampPromptId = "";
    if (pid === state.activeRenderPromptId) {
      state.activeRenderPromptId = "";
      state.activeBaselinePromptId = "";
    }
    clearActiveRender();
    showToast(take.error || "Execution failed on ComfyUI", "err", 5000);
    return true;
  }
  const hasOutput = Boolean(take.still_image || take.clean_video || take.hud_video);
  const isDone = take.status === "completed" || take.status === "done" || hasOutput;
  if (!isDone) return false;
  if (pid && pid === state.activeAnchorPromptId) {
    state.activeAnchorPromptId = "";
    state.livePreview = null;
    wsClient?.clearPreviewMedia();
    state.maskPreview = null;
    clearActiveRender();
    const im = take.still_image;
    if (im) {
      await loadAnchorImageUrl(videoItemViewUrl(im), {
        ref: videoItemRef(im),
        name: im.filename,
        preserveOriginalPrompt: true,
      });
      showToast("Anchor still ready", "ok");
    }
    return true;
  }
  if (pid && pid === state.activeMaskPreviewPromptId) {
    const meta = state.activeRender?.meta || {};
    state.activeMaskPreviewPromptId = "";
    state.livePreview = null;
    wsClient?.clearPreviewMedia();
    clearActiveRender();
    const im = take.still_image;
    if (im) {
      await loadMaskStillPreviewUrl(videoItemViewUrl(im), {
        ...meta,
        ref: videoItemRef(im),
      });
      showToast("Masked image preview applied inside reticle", "ok");
    }
    return true;
  }
  if (pid && pid === state.activeStampPromptId) {
    state.activeStampPromptId = "";
    state.livePreview = null;
    wsClient?.clearPreviewMedia();
    clearActiveRender();
    const im = take.still_image;
    const stampLabel = String(take.params?.prompt || $("#stamp-prompt-input")?.value || "").trim();
    if (im) {
      try {
        const cut = await apiJson("/api/stamp_cutout", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_output: im }),
        });
        const url = cut.view_url || videoItemViewUrl(cut);
        if (url) {
          await loadCutoutUrl(url, cut.cutout_ref || cut.ref || "", {
            keepMaskRef: Boolean(cut.cutout_mask_ref),
            maskRef: cut.cutout_mask_ref || "",
            fromSceneLift: false,
            label: stampLabel,
            bbox: cut.bbox || null,
          });
        }
        showToast("Cutout actor ready (Click Set Path or drag)", "ok");
      } catch {
        await loadCutoutUrl(videoItemViewUrl(im), videoItemRef(im), {
          fromSceneLift: false,
          label: stampLabel,
        });
      }
    }
    return true;
  }
  if (pid && pid === state.activeRenderPromptId) {
    if (take.clean_video || take.hud_video) {
      state.activeRenderPromptId = "";
      state.livePreview = null;
      wsClient?.clearPreviewMedia();
      state.maskPreview = null;
      loadTakeIntoStage(take);
      const basePid = state.activeBaselinePromptId || (take.baseline_pending ? String(take.baseline_prompt_id || "") : "");
      if (basePid && !take.baseline_video && !take.baseline_error) {
        state.activeBaselinePromptId = basePid;
        setActiveRender({
          kind: "baseline",
          pid: basePid,
          phase: `Rendering Same-Seed Baseline (${take.length || state.length}f A|B)…`,
          step: 0,
          total: 6,
        });
        showToast("Directed clip loaded — rendering same-seed baseline (A|B)…", "info", 4200);
      } else {
        clearActiveRender();
        showToast("World clip render complete", "ok");
      }
      return true;
    }
  }
  return false;
}

async function checkActivePromptCompletion(pid) {
  if (!pid) return;
  try {
    await refreshTakes();
    if (await handleFinishedTakeById(pid)) {
      return;
    }
    if (state.activeQueuePromptIds instanceof Set && !state.activeQueuePromptIds.has(pid)) {
      const take = state.takes.find((t) => t.prompt_id === pid || t.baseline_prompt_id === pid);
      const isBaseline = Boolean(take && take.baseline_prompt_id === pid);
      const ageSec = Date.now() / 1000 - Number(take?.updated_at || take?.created_at || 0);
      if (take && ageSec > 5.0 && (take.status !== "completed" || (isBaseline && !take.baseline_video))) {
        if (pid === state.activeAnchorPromptId) state.activeAnchorPromptId = "";
        if (pid === state.activeMaskPreviewPromptId) state.activeMaskPreviewPromptId = "";
        if (pid === state.activeStampPromptId) state.activeStampPromptId = "";
        if (pid === state.activeRenderPromptId) state.activeRenderPromptId = "";
        if (pid === state.activeBaselinePromptId) state.activeBaselinePromptId = "";
        clearActiveRender();
      }
    }
  } catch {}
}

function loadMaskStillPreviewUrl(url, meta = {}) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const pf = evalPortalAtFrame(state.doc, state.frame, state.length);
      const reg = meta.region || {};
      state.maskPreview = {
        img,
        viewUrl: url,
        ref: meta.ref || "",
        x: Number(reg.x ?? pf.x ?? 0.5),
        y: Number(reg.y ?? pf.y ?? 0.5),
        rx: Number(reg.rx ?? pf.rx ?? 0.24),
        ry: Number(reg.ry ?? pf.ry ?? 0.24),
        shape: reg.shape || pf.shape || "circle",
        feather: Number(meta.feather ?? state.doc.feather ?? 0.08),
        haloRadius: Number(reg.halo_radius ?? pf.halo_radius ?? 0),
        haloWeight: Number(reg.halo_weight ?? pf.halo_weight ?? 0.30),
        snapToInclude: Boolean(meta.snapToInclude),
      };
      syncAllUI();
      resolve(img);
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

function showToast(msg, kind = "info", durationMs = 3600) {
  const el = $("#stage-toast");
  if (!el) return;
  el.textContent = msg;
  el.classList.remove("hidden", "err", "ok");
  if (kind === "err") el.classList.add("err");
  if (kind === "ok") el.classList.add("ok");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.add("hidden");
  }, durationMs);
}

function formatTimecode(frameIdx) {
  const sec = Math.max(0, Number(frameIdx || 0)) / FPS;
  const mins = Math.floor(sec / 60);
  const rem = (sec - mins * 60).toFixed(2).padStart(5, "0");
  return `${String(mins).padStart(2, "0")}:${rem}`;
}

function snapshotDoc() {
  return JSON.stringify({
    doc: state.doc,
    targetPrompt: state.targetPrompt,
    editGuidance: state.editGuidance,
  });
}

function pushUndo() {
  const snap = snapshotDoc();
  if (state.undoStack.length && state.undoStack[state.undoStack.length - 1] === snap) return;
  state.undoStack.push(snap);
  if (state.undoStack.length > 40) state.undoStack.shift();
  state.redoStack = [];
  syncUndoRedoButtons();
}

function restoreSnapshot(snap) {
  try {
    const parsed = JSON.parse(snap);
    if (parsed.doc) state.doc = parsed.doc;
    if (typeof parsed.targetPrompt === "string") state.targetPrompt = parsed.targetPrompt;
    if (parsed.editGuidance) state.editGuidance = parsed.editGuidance;
    syncAllUI();
    schedulePreflight();
  } catch {}
}

function syncUndoRedoButtons() {
  const u = $("#btn-undo");
  const r = $("#btn-redo");
  if (u) u.disabled = state.undoStack.length === 0;
  if (r) r.disabled = state.redoStack.length === 0;
}

async function apiJson(url, opts = {}) {
  const resp = await fetch(url, opts);
  const text = await resp.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { ok: false, error: text || `HTTP ${resp.status}` };
  }
  if (!resp.ok) {
    throw new Error(data.error || `HTTP ${resp.status}`);
  }
  return data;
}

// ---------------------------------------------------------------------------
// Schema & System Stats Hydration (Zero Hardcoded Options)
// ---------------------------------------------------------------------------

async function hydrateSchema() {
  const data = await apiJson("/api/schema");
  state.schema = data;
  state.lengths = Array.isArray(data.lengths) ? data.lengths.map(Number).filter((n) => n > 0) : [];
  state.canvases = Array.isArray(data.canvases) ? data.canvases : [];
  state.workflows = Array.isArray(data.workflows) ? data.workflows : [];
  state.anchorWorkflows = Array.isArray(data.anchor_workflows) ? data.anchor_workflows : [];
  state.imageEditWorkflows = Array.isArray(data.image_edit_workflows) ? data.image_edit_workflows : [];
  state.controlVectors = Array.isArray(data.control_vectors) ? data.control_vectors : [];
  state.capabilities = { ...state.capabilities, ...(data.capabilities || {}) };

  // Sound presets hydrated purely from backend schema
  const rawPresets = data.sound_presets;
  const presetMap = {};
  if (Array.isArray(rawPresets)) {
    for (const item of rawPresets) {
      if (typeof item === "string") presetMap[item] = 0.2;
      else if (item && item.name) presetMap[item.name] = Number(item.duration_sec || 0.2);
    }
  } else if (rawPresets && typeof rawPresets === "object") {
    for (const [k, v] of Object.entries(rawPresets)) {
      presetMap[k] = Number(v || 0.2);
    }
  }
  state.soundPresetMap = presetMap;
  state.soundPresetsList = Object.keys(presetMap);
  state.defaultSoundPreset = state.soundPresetsList[0] || "none";

  // Defaults from schema
  const defs = data.defaults || {};
  if (defs.length && state.lengths.includes(Number(defs.length))) {
    state.length = Number(defs.length);
  } else if (state.lengths.length) {
    state.length = state.lengths[Math.min(2, state.lengths.length - 1)];
  }
  if (defs.width && defs.height) {
    state.doc.width = Number(defs.width);
    state.doc.height = Number(defs.height);
  } else if (state.canvases.length) {
    state.doc.width = Number(state.canvases[0].width || 1344);
    state.doc.height = Number(state.canvases[0].height || 768);
  }
  if (defs.feather != null) state.doc.feather = Number(defs.feather);
  if (defs.scale != null) state.editGuidance.scale = Number(defs.scale);
  if (defs.audio_scale != null) state.editGuidance.audio_scale = Number(defs.audio_scale);
  if (defs.streams) state.editGuidance.streams = String(defs.streams);
  if (defs.anchor_workflow) state.anchorWorkflowId = String(defs.anchor_workflow);
  if (defs.preview_image_workflow) state.previewWorkflowId = String(defs.preview_image_workflow);
  if (defs.mode && state.workflows.some((w) => w.id === defs.mode)) {
    state.workflowId = String(defs.mode);
  }
  if (defs.seed != null) state.seed = Number(defs.seed);
  if (defs.events_json) {
    try {
      const parsedEv = typeof defs.events_json === "string" ? JSON.parse(defs.events_json) : defs.events_json;
      if (parsedEv && typeof parsedEv === "object") {
        state.doc = { ...state.doc, ...parsedEv };
      }
    } catch {}
  }
  const maxF = Math.max(0, state.length - 1);
  for (const p of state.doc.portal || []) p.frame = clamp(Number(p.frame || 0), 0, maxF);
  for (const w of state.doc.dpad?.waypoints || []) w.frame = clamp(Number(w.frame || 0), 0, maxF);

  renderHydratedControls();
  syncAllUI();
  schedulePreflight();
}

function renderHydratedControls() {
  // 1. Header Dual-Clock Pills from schema.lengths (default to >=124f universal 5..15s grid)
  const clockWrap = $("#clock-pills");
  if (clockWrap) {
    clockWrap.replaceChildren();
    const universal = state.lengths.filter((n) => n >= 124).slice(0, 2);
    const showList = universal.length ? universal : state.lengths.slice(0, 2);
    for (const lenVal of showList) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "clock-btn";
      btn.dataset.length = String(lenVal);
      btn.textContent = `${lenVal}f`;
      btn.title = `${lenVal} frames (${(lenVal / FPS).toFixed(2)}s on both clocks)`;
      btn.addEventListener("click", () => switchLength(lenVal));
      clockWrap.appendChild(btn);
    }
  }

  // 2. Engine Length Select (all on_both_clocks)
  const lenSel = $("#engine-length");
  if (lenSel) {
    lenSel.replaceChildren();
    for (const lenVal of state.lengths) {
      const opt = document.createElement("option");
      opt.value = String(lenVal);
      const r2vNote = lenVal < 124 ? " · R2V/T2V only" : "";
      opt.textContent = `${lenVal}f (${(lenVal / FPS).toFixed(2)}s${r2vNote})`;
      lenSel.appendChild(opt);
    }
    lenSel.value = String(state.length);
  }

  // 3. Canvases Select
  const canvasSel = $("#engine-canvas");
  if (canvasSel) {
    canvasSel.replaceChildren();
    for (const c of state.canvases) {
      const opt = document.createElement("option");
      const val = `${c.width}x${c.height}`;
      opt.value = val;
      opt.textContent = c.label || `${c.width}×${c.height} (${c.aspect || ""})`;
      canvasSel.appendChild(opt);
    }
    const curVal = `${state.doc.width}x${state.doc.height}`;
    if (Array.from(canvasSel.options).some((o) => o.value === curVal)) {
      canvasSel.value = curVal;
    }
  }

  // 4. Spatial Workflows Select
  const wfSel = $("#engine-workflow");
  if (wfSel) {
    wfSel.replaceChildren();
    for (const wf of state.workflows) {
      const opt = document.createElement("option");
      opt.value = wf.id;
      opt.textContent = wf.available === false ? `${wf.label || wf.id} (unavailable)` : (wf.label || wf.id);
      wfSel.appendChild(opt);
    }
    const preferred = state.workflows.find((w) => w.available !== false) || state.workflows[0];
    if (preferred && !state.workflowId) {
      state.workflowId = preferred.id;
    }
    if (state.workflowId) wfSel.value = state.workflowId;
  }

  // 5. Anchor Still Workflows Select & Image Edit Preview Select
  const anchorSel = $("#anchor-workflow-select");
  if (anchorSel) {
    anchorSel.replaceChildren();
    for (const aw of state.anchorWorkflows) {
      const opt = document.createElement("option");
      opt.value = aw.id;
      opt.textContent = aw.label || aw.id;
      anchorSel.appendChild(opt);
    }
    if (state.anchorWorkflows.length && !state.anchorWorkflowId) {
      state.anchorWorkflowId = state.anchorWorkflows[0].id;
    }
    if (state.anchorWorkflowId) anchorSel.value = state.anchorWorkflowId;
  }

  const prevSel = $("#preview-workflow-select");
  if (prevSel) {
    prevSel.replaceChildren();
    const list = state.imageEditWorkflows.length ? state.imageEditWorkflows : state.anchorWorkflows;
    for (const pw of list) {
      const opt = document.createElement("option");
      opt.value = pw.id;
      opt.textContent = pw.label || pw.id;
      prevSel.appendChild(opt);
    }
    if (list.length && !state.previewWorkflowId) {
      state.previewWorkflowId = list[0].id;
    }
    if (state.previewWorkflowId) prevSel.value = state.previewWorkflowId;
  }

  // 6. Sound Cue Presets (Card + Drawer)
  for (const selId of ["#card-cue-select", "#engine-cue-preset"]) {
    const sel = $(selId);
    if (!sel) continue;
    sel.replaceChildren();
    if (selId === "#card-cue-select") {
      const noneOpt = document.createElement("option");
      noneOpt.value = "none";
      noneOpt.textContent = "none (silent)";
      sel.appendChild(noneOpt);
    }
    for (const name of state.soundPresetsList) {
      const opt = document.createElement("option");
      opt.value = name;
      const ms = Math.round((state.soundPresetMap[name] || 0.2) * 1000);
      opt.textContent = `${name} (${ms}ms)`;
      sel.appendChild(opt);
    }
  }
  const hint = $("#cue-hotkeys-hint");
  if (hint) {
    const pairs = state.soundPresetsList.slice(0, 9).map((name, i) => `${i + 1}:${name}`);
    hint.textContent = pairs.length ? `Hotkeys: ${pairs.join(" · ")}` : "No cue presets reported by schema";
  }

  // 7. Control Vectors Select
  const cvSel = $("#engine-cv-select");
  if (cvSel) {
    cvSel.replaceChildren();
    const noneOpt = document.createElement("option");
    noneOpt.value = "none";
    noneOpt.textContent = "none";
    cvSel.appendChild(noneOpt);
    for (const cv of state.controlVectors) {
      const opt = document.createElement("option");
      opt.value = cv.name;
      opt.textContent = cv.arch ? `${cv.name} (${cv.arch})` : cv.name;
      cvSel.appendChild(opt);
    }
    cvSel.value = state.controlVector;
  }

  // 8. Capabilities Badges
  const capWrap = $("#capabilities-list");
  if (capWrap) {
    capWrap.replaceChildren();
    for (const [key, enabled] of Object.entries(state.capabilities)) {
      const badge = document.createElement("span");
      badge.className = enabled ? "cap-badge on" : "cap-badge";
      badge.textContent = `${key}: ${enabled ? "on" : "off"}`;
      capWrap.appendChild(badge);
    }
  }

  syncLiveTaeBadges();
}

async function pollSystemStats() {
  try {
    const stats = await apiJson("/api/system_stats");
    state.residentBackbone = stats.resident_backbone || "";
    const dev = Array.isArray(stats.devices) && stats.devices[0] ? stats.devices[0] : null;
    const vLabel = $("#vram-label");
    const vFill = $("#vram-bar-fill");
    const warmTag = state.residentBackbone ? ` · warm: ${state.residentBackbone}` : "";
    if (dev && dev.vram_total > 0) {
      const totalGb = dev.vram_total / (1024 ** 3);
      const freeGb = (dev.vram_free || 0) / (1024 ** 3);
      const usedGb = Math.max(0, totalGb - freeGb);
      const pct = clamp((usedGb / totalGb) * 100, 0, 100);
      if (vLabel) vLabel.textContent = `${usedGb.toFixed(1)} / ${totalGb.toFixed(1)} GB${warmTag}`;
      if (vFill) {
        vFill.setAttribute("width", pct.toFixed(1));
        vFill.setAttribute("fill", pct > 88 ? "#B45309" : "#047857");
      }
    } else if (vLabel) {
      vLabel.textContent = (stats.comfy_reachable ? "CPU / Ready" : "ComfyUI offline") + warmTag;
    }
  } catch {}
}

function schedulePreflight() {
  if (preflightTimer) clearTimeout(preflightTimer);
  preflightTimer = setTimeout(() => {
    runPreflight();
  }, 180);
}

async function runPreflight() {
  try {
    const payload = buildCompilePayload();
    const pf = await apiJson("/api/preflight", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const totalTok = Number(pf.total_tokens || 0);
    const limit = Number(pf.int32_limit || INT32_TOKEN_LIMIT);
    const pct = clamp((totalTok / limit) * 100, 0, 100);
    const label = $("#token-label");
    const fill = $("#token-bar-fill");
    const detail = $("#preflight-detail");
    if (label) {
      label.textContent = `${totalTok.toLocaleString()} / ${limit.toLocaleString()}`;
    }
    if (fill) {
      fill.setAttribute("width", pct.toFixed(1));
      fill.setAttribute("fill", pf.exceeds_int32 ? "#DC2626" : (pct > 82 ? "#B45309" : "#2563EB"));
    }
    if (detail) {
      const warnTxt = Array.isArray(pf.warnings) && pf.warnings.length ? ` · ${pf.warnings[0]}` : "";
      detail.textContent = `Video steps: ${pf.latent_steps ?? "—"} · Audio tok: ${pf.audio_tokens ?? "—"}${warnTxt}`;
    }
  } catch {}
}

// ---------------------------------------------------------------------------
// Sync UI & Scrubber SVG (Zero element.style.*)
// ---------------------------------------------------------------------------

function syncLiveTaeBadges(stepMeta = null) {
  const hdrPill = $("#live-tae-pill");
  const hdrText = $("#live-tae-text");
  const stageText = $("#stage-tae-text");
  const portalText = $("#portal-tae-text");

  const pf = evalPortalAtFrame(state.doc, state.frame, state.length);
  const radStr = state.doc.mode === "point_targets" && state.doc.targets?.[state.selectedTargetIdx]
    ? Number(state.doc.targets[state.selectedTargetIdx].radius || 0.16).toFixed(2)
    : Number(pf.rx || 0.22).toFixed(2);

  if (stepMeta && stepMeta.total > 0) {
    const fpsPart = stepMeta.fps ? ` · ${Number(stepMeta.fps).toFixed(1)}fps` : "";
    const hasDelta = stepMeta.delta !== null && stepMeta.delta !== undefined && Number.isFinite(Number(stepMeta.delta));
    const deltaPart = hasDelta ? ` · Δ ${Number(stepMeta.delta).toFixed(2)}` : "";
    const stepStr = `Step ${stepMeta.step}/${stepMeta.total}`;
    if (hdrText) hdrText.textContent = `Live Preview: ${stepMeta.step}/${stepMeta.total}${fpsPart}${deltaPart}`;
    if (stageText) stageText.textContent = `Live Latents · ${stepStr}`;
    if (portalText) portalText.textContent = `Live Preview · ${stepStr} · r=${radStr}`;
    hdrPill?.classList.remove("warn");
  } else if (state.capabilities.kj_preview_override && state.capabilities.taeh3) {
    if (hdrText) hdrText.textContent = "Live Preview: Ready · KV 1x";
    if (stageText) stageText.textContent = "Live Latents Ready";
    if (portalText) portalText.textContent = `Live Preview · Ready · r=${radStr}`;
    hdrPill?.classList.remove("warn");
  } else if (state.capabilities.kj_preview_override) {
    if (hdrText) hdrText.textContent = "Live Preview: Latent2RGB";
    if (stageText) stageText.textContent = "Latent2RGB Fallback";
    if (portalText) portalText.textContent = `Live Preview · Ready · r=${radStr}`;
    hdrPill?.classList.add("warn");
  } else {
    if (hdrText) hdrText.textContent = "Live Preview: Idle";
    if (stageText) stageText.textContent = "Sampler Telemetry Ready";
    if (portalText) portalText.textContent = `Portal · r=${radStr}`;
    hdrPill?.classList.add("warn");
  }
}

function syncScrubberSVG() {
  const maxF = Math.max(1, state.length - 1);
  $("#timecode-cur").textContent = formatTimecode(state.frame);
  $("#timecode-total").textContent = formatTimecode(state.length);
  $("#scrubber-frame-cur").textContent = String(state.frame);
  $("#scrubber-frame-total").textContent = `${state.length} f`;

  const canvas = $("#scrubber-canvas");
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const cw = Math.max(120, Math.round((rect.width || 600) * dpr));
  const ch = Math.max(24, Math.round((rect.height || 32) * dpr));
  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width = cw;
    canvas.height = ch;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, cw, ch);
  const padX = 10 * dpr;
  const trackW = Math.max(20, cw - padX * 2);
  const trackH = 8 * dpr;
  const trackY = (ch - trackH) * 0.5;
  const frac = state.frame / maxF;
  const thumbX = padX + frac * trackW;

  const drawPillBar = (x, y, w, h, r, fill) => {
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h);
    ctx.fill();
  };

  // Track base
  drawPillBar(padX, trackY, trackW, trackH, trackH * 0.5, "#E2E8F0");

  // Causal 17n+5 & [1, 4, 4, 4, 4] step boundaries
  const { clipStarts, stepStarts } = causalStepBoundaries(state.length);
  ctx.save();
  ctx.fillStyle = "rgba(100, 116, 139, 0.35)";
  for (const c of clipStarts) {
    if (c === 0) continue;
    const cx = padX + (c / maxF) * trackW;
    ctx.fillRect(cx - 0.5 * dpr, trackY - 3 * dpr, 1 * dpr, trackH + 6 * dpr);
  }
  if (state.showPatchGrid) {
    ctx.fillStyle = "rgba(37, 99, 235, 0.35)";
    for (const s of stepStarts) {
      const sx = padX + (s / maxF) * trackW;
      ctx.fillRect(sx - 0.5 * dpr, trackY + trackH + 1 * dpr, 1 * dpr, 3 * dpr);
    }
  }
  ctx.restore();

  // Causal active span highlights
  if (state.doc.mode === "portal_path" || state.doc.mode === "combined") {
    const pStart = state.doc.portal_start != null ? Number(state.doc.portal_start) : 0;
    const pEnd = state.doc.portal_end != null ? Number(state.doc.portal_end) : maxF;
    const x0 = padX + (pStart / maxF) * trackW;
    const x1 = padX + (pEnd / maxF) * trackW;
    drawPillBar(x0, trackY - 1 * dpr, Math.max(6 * dpr, x1 - x0), trackH + 2 * dpr, 4 * dpr, "rgba(37, 99, 235, 0.20)");
  }
  if (state.doc.mode === "point_targets" || state.doc.mode === "combined") {
    for (const t of state.doc.targets || []) {
      const x0 = padX + (clamp(Number(t.start || 0), 0, maxF) / maxF) * trackW;
      const x1 = padX + (clamp(Number(t.end || maxF), 0, maxF) / maxF) * trackW;
      drawPillBar(x0, trackY - 1 * dpr, Math.max(8 * dpr, x1 - x0), trackH + 2 * dpr, 4 * dpr, "rgba(180, 83, 9, 0.28)");
    }
  }

  // Played fill
  drawPillBar(padX, trackY, Math.max(0, thumbX - padX), trackH, trackH * 0.5, "#93C5FD");

  // Portal keyframe dots
  if (state.doc.mode === "portal_path" || state.doc.mode === "combined") {
    for (const kf of state.doc.portal || []) {
      const kx = padX + (clamp(Number(kf.frame || 0), 0, maxF) / maxF) * trackW;
      ctx.fillStyle = "#2563EB";
      ctx.beginPath();
      ctx.arc(kx, ch * 0.5, 3.5 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // 3f foley cue pins
  for (const c of allActiveCues()) {
    const cx = padX + (clamp(c.frame, 0, maxF) / maxF) * trackW;
    drawPillBar(cx - 1.5 * dpr, 6 * dpr, 3 * dpr, ch - 12 * dpr, 1.5 * dpr, "#047857");
  }

  // Playhead circle thumb (never clipped or stretched)
  ctx.fillStyle = "#FFFFFF";
  ctx.strokeStyle = "#2563EB";
  ctx.lineWidth = 2.5 * dpr;
  ctx.beginPath();
  ctx.arc(thumbX, ch * 0.5, 7 * dpr, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function allActiveCues() {
  const cues = [];
  if (state.doc.mode === "point_targets" || state.doc.mode === "combined") {
    for (const t of state.doc.targets || []) {
      if (t.cue && t.cue !== "none") {
        cues.push({
          frame: snapFrameTo3f(t.start, state.length),
          preset: t.cue,
          gain: t.gain || 1.15,
          x: Number(t.x ?? 0.5),
          source: "target",
        });
      }
    }
  }
  for (const c of state.doc.portal_cues || []) {
    const f = snapFrameTo3f(c.frame, state.length);
    const pf = evalPortalAtFrame(state.doc, f, state.length);
    cues.push({
      frame: f,
      preset: c.preset,
      gain: c.gain || 1.0,
      x: Number(pf.x ?? 0.5),
      source: "portal",
    });
  }
  cues.sort((a, b) => a.frame - b.frame);
  return cues;
}

function renderSynthesizedOperators(pf, sp, coupling) {
  const host = $("#synth-operators");
  const detailEl = $("#synth-op-detail");
  if (!host) return;
  const ops = [
    {
      id: "aperture",
      title: `Mask ${Number(pf.rx).toFixed(2)}`,
      sub: `${pf.shape} · EditGuider ${Number(state.editGuidance.scale).toFixed(1)}x · ${state.editGuidance.streams}`,
    },
    {
      id: "halo",
      title: `Halo ${Number(pf.halo_radius || 0).toFixed(2)}`,
      sub: `feather=${Number(state.doc.feather).toFixed(2)} · halo_weight=${Number(pf.halo_weight ?? 0.30).toFixed(2)}x`,
    },
    {
      id: "coupling",
      title: coupling.active ? `Couple ${Math.round(coupling.strength * 100)}%` : "Couple",
      sub: coupling.active
        ? `Sprite ${coupling.mode} Δ(${coupling.dx >= 0 ? "+" : ""}${coupling.dx.toFixed(2)}, ${coupling.dy >= 0 ? "+" : ""}${coupling.dy.toFixed(2)}) · Release 0.33`
        : "Shift+Drag Portal → Sprite or click to link push/pull",
    },
  ];
  host.replaceChildren();
  const activeOp = ops.find((o) => o.id === state.activeOperator) || ops[0];
  if (detailEl) detailEl.textContent = activeOp.sub;
  for (const op of ops) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.title = op.sub;
    btn.className = state.activeOperator === op.id ? "synth-op-btn active" : "synth-op-btn";
    const t = document.createElement("span");
    t.className = "synth-op-title";
    t.textContent = op.title;
    btn.append(t);
    btn.addEventListener("click", () => {
      state.activeOperator = op.id;
      if (op.id === "coupling") {
        pushUndo();
        if (state.couplings.length) {
          state.couplings[0].mode = state.couplings[0].mode === "push" ? "pull" : "push";
        } else {
          state.spriteMode = true;
          state.couplings = [{
            portal_index: 0,
            target_kind: "sprite",
            strength: 0.85,
            mode: "push",
            label: "Sprite Coupling",
          }];
        }
        syncAllUI();
      } else {
        syncContextualCard();
      }
    });
    host.appendChild(btn);
  }
}

function resolvedWorkflowMode() {
  if (state.workflowManualOverride && state.workflowId) {
    return state.workflowId;
  }
  if (state.spriteCutout?.ref) {
    return "r2v_portal";
  }
  if (state.anchorRef) {
    return "i2v_portal";
  }
  return "t2v_portal";
}

function syncPromptDiffPreview() {
  const origEl = $("#diff-orig-line");
  const editEl = $("#diff-edit-line");
  const badgeEl = $("#prompt-diff-badge");
  if (!origEl || !editEl) return;

  const wfMode = resolvedWorkflowMode();
  const modeLabelMap = {
    i2v_portal: "Frame-0 I2V",
    r2v_portal: "Actor R2V (Release 0.33)",
    world_targets: "World Targets R2V",
    t2v_portal: "Text-to-Video",
  };
  const customOrig = (state.originalPrompt || "").trim();
  const editClause = (state.targetPrompt || "").trim();
  const hasSpriteAction = Boolean(
    state.spriteCutout?.ref
    || (state.doc.dpad?.waypoints?.length || 0) > 1
    || (state.doc.dpad?.hops?.length || 0) > 0
  );
  origEl.textContent = customOrig
    ? `- ${customOrig.slice(0, 56)}${customOrig.length > 56 ? "…" : ""}`
    : (state.anchorRef ? "- Anchored <Picture 1> calm hold (auto)" : "- Clean unedited scene base (auto)");
  if (state.samePrompt) {
    if (badgeEl) badgeEl.textContent = "Same-Prompt Steer";
    editEl.textContent = "+ Control Vector delta (edited_only pass)";
  } else if (editClause) {
    const activeTargets = (state.doc.mode === "point_targets" || state.doc.mode === "combined")
      ? (state.doc.targets || []).filter((t) => (t.prompt || "").trim())
      : [];
    const portalBeats = (state.doc.portal || []).filter((p) => (p.prompt || "").trim());
    const beatFrames = Array.from(new Set([
      ...activeTargets.map((t) => `f${t.start}`),
      ...portalBeats.map((p) => `f${p.frame}`),
    ]));
    const beatCount = activeTargets.length + (portalBeats.length > 1 ? portalBeats.length : 0);
    const beatPrefix = beatCount >= 2 ? `[${beatFrames.length} beats: ${beatFrames.join(", ")}] ` : "";
    if (badgeEl) badgeEl.textContent = modeLabelMap[wfMode] || "Prompt Diff";
    editEl.textContent = `+ ${beatPrefix}${editClause}`;
  } else if (hasSpriteAction) {
    if (badgeEl) badgeEl.textContent = modeLabelMap[wfMode] || "Prompt Diff";
    editEl.textContent = "+ Auto-describe Actor Layer trajectory (or type action above)";
  } else {
    if (badgeEl) badgeEl.textContent = modeLabelMap[wfMode] || "Prompt Diff";
    editEl.textContent = "+ Enter action above to steer region";
  }
}

function syncContextualCard() {
  const card = $("#portal-card");
  if (!card) return;
  if (!state.hasAnchor || abCompare?.active) {
    card.classList.add("hidden");
    return;
  }
  card.classList.remove("hidden");

  const pf = evalPortalAtFrame(state.doc, state.frame, state.length);
  const sp = (state.spriteMode || state.spriteCutout?.img || (state.couplings || []).length > 0)
    ? evalSpriteAtFrame(state.doc, state.frame, state.length)
    : null;
  const coupling = computeCouplingInfo(state, pf, sp);
  let anchorX = pf.x;
  let normX = pf.x;
  let normY = pf.y;

  if (state.doc.mode === "point_targets" && state.doc.targets?.length) {
    const idx = clamp(state.selectedTargetIdx ?? 0, 0, state.doc.targets.length - 1);
    const t = state.doc.targets[idx];
    anchorX = t.x;
    normX = t.x;
    normY = t.y;
    $("#portal-card-heading").textContent = `Target Pin #${idx + 1}`;
    const nearCue = t.cue && t.cue !== "none" ? t.cue : "none";
    const cueSel = $("#card-cue-select");
    if (cueSel) cueSel.value = nearCue;
    $("#pill-cue-text").textContent = `3f @ f${snapFrameTo3f(t.start, state.length)}: ${nearCue}`;
  } else if (state.doc.mode === "combined") {
    $("#portal-card-heading").textContent = "Combined Portal";
    const nearCue = (state.doc.portal_cues || [])[0];
    const cueSel = $("#card-cue-select");
    if (nearCue) {
      if (cueSel) cueSel.value = nearCue.preset;
      $("#pill-cue-text").textContent = `3f @ f${snapFrameTo3f(nearCue.frame, state.length)}: ${nearCue.preset}`;
    } else {
      if (cueSel) cueSel.value = "none";
      $("#pill-cue-text").textContent = `3f @ f${snapFrameTo3f(state.frame, state.length)}: none`;
    }
  } else {
    $("#portal-card-heading").textContent = state.spriteMode ? "Spatial Portal · Actor" : "Spatial Portal";
    const nearCue = (state.doc.portal_cues || [])[0];
    const cueSel = $("#card-cue-select");
    if (nearCue) {
      if (cueSel) cueSel.value = nearCue.preset;
      $("#pill-cue-text").textContent = `3f @ f${snapFrameTo3f(nearCue.frame, state.length)}: ${nearCue.preset}`;
    } else {
      if (cueSel) cueSel.value = "none";
      $("#pill-cue-text").textContent = `3f @ f${snapFrameTo3f(state.frame, state.length)}: none`;
    }
  }

  // Interactive progress-bar dials (SVG rects, zero element.style)
  const rxVal = Number(pf.rx || 0.22);
  const haloVal = Number(pf.halo_radius || 0);
  const outerVal = rxVal + haloVal;
  $("#dial-mask-val").textContent = `${rxVal.toFixed(2)} → ${outerVal.toFixed(2)}`;
  $("#dial-mask-fill")?.setAttribute("width", clamp((rxVal / 0.45) * 60, 4, 65).toFixed(1));
  $("#dial-halo-fill")?.setAttribute("width", clamp((outerVal / 0.80) * 100, 6, 100).toFixed(1));

  const scaleVal = Number(state.editGuidance.scale || 1.5);
  const cPct = Math.round(coupling.strength * 100);
  $("#dial-coupling-val").textContent = `${scaleVal.toFixed(1)}x · ${cPct}%`;
  const cFillPct = coupling.active ? cPct : ((scaleVal - 0.5) / 2.0) * 100;
  $("#dial-coupling-fill")?.setAttribute("width", clamp(cFillPct, 6, 100).toFixed(1));

  // Compiled Nodes readout
  const cuesCount = allActiveCues().length;
  const spriteNote = sp
    ? ` · AppendSpriteTrajectory(51, Release 0.33, Δx=${coupling.dx >= 0 ? "+" : ""}${coupling.dx.toFixed(2)})`
    : "";
  const maskParts = [];
  if (sp && !state.doc.invert_mask) maskParts.push("SpriteEnv(dpad)");
  if (state.spriteMaskRef) maskParts.push("SpriteMask");
  if (state.excludeMaskRef) maskParts.push("Protect(exclude)");
  if (state.includeMaskRef) maskParts.push("Lock(include)");
  const maskNote = maskParts.length ? ` · ${maskParts.join("+")}` : "";
  $("#compiled-nodes-readout").textContent =
    `SpatialDirector(160) -> EditGuider(${scaleVal.toFixed(1)}x) · ${cuesCount}× 3f SoundCues${spriteNote}${maskNote}`;

  const pickBtn = $("#btn-sam-pick");
  if (pickBtn) {
    pickBtn.disabled = !state.hasAnchor || state.samSegmenting;
    pickBtn.classList.toggle("active", Boolean(state.samPicking));
    pickBtn.textContent = state.samSegmenting
      ? "Segmenting…"
      : (state.samPicking ? "Click Subject…" : "Pick Subject");
  }
  for (const rBtn of $$(".sam-role-btn")) {
    const activeRole = rBtn.dataset.role === state.samRole;
    rBtn.classList.toggle("active", activeRole);
    rBtn.setAttribute("aria-pressed", activeRole ? "true" : "false");
  }
  const hasAnySamMask = Boolean(
    state.samPicking || state.samPosPoints.length || state.samNegPoints.length || state.spriteMaskRef || state.excludeMaskRef || state.includeMaskRef
  );
  $("#btn-sam-clear")?.classList.toggle("hidden", !Boolean(state.spriteMaskRef || state.excludeMaskRef || state.includeMaskRef || state.samPosPoints.length || state.samNegPoints.length));
  $("#sam-refine-bar")?.classList.toggle("hidden", !hasAnySamMask);
  $("#sam-actor-actions")?.classList.toggle("hidden", !Boolean(state.spriteCutout?.ref || state.includeMaskRef || state.spriteMode));

  const setPathBtn = $("#btn-set-path");
  if (setPathBtn) {
    setPathBtn.classList.toggle("active", Boolean(state.settingSpritePath));
    setPathBtn.textContent = state.settingSpritePath ? "Click Destination…" : "Set Path →";
  }
  const snapChk = $("#chk-snap-mask");
  if (snapChk) {
    snapChk.checked = Boolean(state.snapToIncludeMask);
    snapChk.disabled = !state.includeMaskRef;
  }
  const actorHint = $("#sam-actor-hint");
  if (actorHint) {
    let hintText = "";
    let warn = false;
    if (state.spriteCutout?.ref && state.spriteCutout?.fromSceneLift) {
      hintText = "Frame-0 lift copies background behind actor — use + Stamp for a clean-plate actor.";
      warn = true;
    } else if (state.spriteCutout?.ref) {
      hintText = "Clean-plate cutout — click Set Path → or steer with WASD / Space.";
    } else if (state.spriteMode) {
      hintText = "Use SAM Movable or + Stamp to attach an actor cutout, then Set Path →.";
    }
    actorHint.textContent = hintText;
    actorHint.classList.toggle("is-warning", warn);
    actorHint.classList.toggle("hidden", !hintText);
  }

  const prevMaskBtn = $("#btn-preview-mask");
  if (prevMaskBtn) {
    prevMaskBtn.disabled = !state.hasAnchor || Boolean(state.activeRender);
  }
  $("#btn-clear-mask-preview")?.classList.toggle("hidden", !Boolean(state.maskPreview?.img));

  const coordBadge = $("#synth-latency-badge");
  if (coordBadge) {
    coordBadge.textContent = `[${normX.toFixed(2)}, ${normY.toFixed(2)}]`;
  }
  const modeTag = $("#synth-mode-tag");
  if (modeTag) {
    modeTag.textContent = resolvedWorkflowMode();
  }

  renderSynthesizedOperators(pf, sp, coupling);
  syncPromptDiffPreview();

  // Clamp card opposite the active portal/target (or honor manual Left/Right flip)
  const dockLeft = state.manualDockSide ? state.manualDockSide === "left" : anchorX > 0.56;
  card.classList.toggle("dock-left", dockLeft);
  card.classList.toggle("dock-right", !dockLeft);

  $("#card-time-badge").textContent = `f${String(state.frame).padStart(3, "0")} · ${(state.frame / FPS).toFixed(2)}s`;
  const tp = $("#target-prompt");
  if (tp && document.activeElement !== tp) {
    if (state.doc.mode === "point_targets" && state.doc.targets?.length) {
      const idx = clamp(state.selectedTargetIdx ?? 0, 0, state.doc.targets.length - 1);
      const tPrompt = state.doc.targets[idx]?.prompt;
      tp.value = typeof tPrompt === "string" && tPrompt.length ? tPrompt : state.targetPrompt;
    } else {
      tp.value = state.targetPrompt;
    }
  }
  const op = $("#original-prompt");
  if (op && document.activeElement !== op) {
    op.value = state.originalPrompt;
  }
}

function syncCueChips() {
  const wrap = $("#cue-chips");
  if (!wrap) return;
  wrap.replaceChildren();
  for (let i = 0; i < (state.doc.portal_cues || []).length; i++) {
    const c = state.doc.portal_cues[i];
    const chip = document.createElement("span");
    chip.className = "cue-chip";
    chip.textContent = `f${c.frame}: ${c.preset}`;
    const rm = document.createElement("button");
    rm.type = "button";
    rm.textContent = "×";
    rm.title = "Remove cue";
    rm.addEventListener("click", () => {
      pushUndo();
      state.doc.portal_cues.splice(i, 1);
      syncAllUI();
    });
    chip.appendChild(rm);
    wrap.appendChild(chip);
  }
}

function syncAllUI() {
  // Header clock pills
  for (const btn of $$(".clock-btn")) {
    btn.classList.toggle("active", Number(btn.dataset.length) === state.length);
  }
  const lenSel = $("#engine-length");
  if (lenSel && lenSel.value !== String(state.length)) {
    lenSel.value = String(state.length);
  }

  // Scene pill + footer anchor/layer pills
  const anchorTitle = state.anchorName ? state.anchorName.replace(/\.[^/.]+$/, "") : "No Anchor";
  $("#scene-pill-text").textContent = `${anchorTitle} (${state.doc.width}×${state.doc.height})`;
  $("#footer-anchor-text").textContent = state.hasAnchor ? `Anchor: ${anchorTitle}` : "Anchor: None";

  if (state.doc.mode === "point_targets") {
    $("#footer-layer-text").textContent = `Targets (${(state.doc.targets || []).length})`;
  } else if (state.doc.mode === "combined") {
    const ps = state.doc.portal_start != null ? state.doc.portal_start : 0;
    const pe = state.doc.portal_end != null ? state.doc.portal_end : state.length - 1;
    $("#footer-layer-text").textContent = `Portal(f${ps}–f${pe}) + ${(state.doc.targets || []).length} Targets`;
  } else {
    const ps = state.doc.portal_start != null ? state.doc.portal_start : 0;
    const pe = state.doc.portal_end != null ? state.doc.portal_end : state.length - 1;
    $("#footer-layer-text").textContent = `Portal+Halo (f${ps}–f${pe})`;
  }
  const pfFoot = evalPortalAtFrame(state.doc, state.frame, state.length);
  const spFoot = (state.spriteMode || state.spriteCutout?.img || (state.couplings || []).length > 0)
    ? evalSpriteAtFrame(state.doc, state.frame, state.length)
    : null;
  const coupFoot = computeCouplingInfo(state, pfFoot, spFoot);
  const cPill = $("#footer-coupling-pill");
  if (cPill) {
    cPill.classList.toggle("hidden", !coupFoot.active);
    if (coupFoot.active) {
      $("#footer-coupling-text").textContent = `Coupling ${Math.round(coupFoot.strength * 100)}%`;
    }
  }

  // Mode buttons (spatial mode switch + orthogonal Actor Layer toggle)
  $("#mode-btn-portal")?.classList.toggle("active", state.doc.mode === "portal_path");
  $("#mode-btn-targets")?.classList.toggle("active", state.doc.mode === "point_targets");
  $("#mode-btn-combined")?.classList.toggle("active", state.doc.mode === "combined");
  const spriteBtn = $("#mode-btn-sprite");
  if (spriteBtn) {
    spriteBtn.classList.toggle("active", Boolean(state.spriteMode));
    spriteBtn.setAttribute("aria-pressed", state.spriteMode ? "true" : "false");
  }

  $("#stage-helper-text").textContent = state.samPicking
    ? `Click subject on stage to segment (${state.samRole}) · Esc cancels`
    : (state.settingSpritePath
      ? "Click anywhere on stage to set Movable Actor destination"
      : (state.spriteMode
        ? "WASD steers actor · Set Path places arrow · Space hops"
        : (state.doc.mode === "combined"
          ? "Drag Region or Timed Events · Ctrl+Click adds Event · Scroll size"
          : (state.doc.mode === "point_targets"
            ? "Click to move timed event · Shift+Click adds event"
            : "Click/drag region · Scroll size · Shift Halo · Alt Scale"))));

  // Play / Pause icon
  $("#icon-play")?.classList.toggle("hidden", state.playing);
  $("#icon-pause")?.classList.toggle("hidden", !state.playing);

  // Sync drawer inputs without clobbering active typing
  const setVal = (sel, val) => {
    const el = $(sel);
    if (el && document.activeElement !== el) el.value = String(val);
  };
  setVal("#seed-input", state.seed);
  setVal("#engine-scale", state.editGuidance.scale);
  setVal("#engine-audio-scale", state.editGuidance.audio_scale);
  setVal("#engine-first-step", state.editGuidance.first_step);
  setVal("#engine-streams", state.editGuidance.streams);
  setVal("#engine-feather", state.doc.feather);
  const pf = evalPortalAtFrame(state.doc, state.frame, state.length);
  setVal("#engine-shape", pf.shape || "circle");
  const invChk = $("#chk-invert-mask");
  if (invChk) invChk.checked = Boolean(state.doc.invert_mask);
  const spChk = $("#chk-same-prompt");
  if (spChk) spChk.checked = Boolean(state.samePrompt);
  const arChk = $("#chk-audio-refine");
  if (arChk) arChk.checked = Boolean(state.audioRefine);

  syncLiveTaeBadges(
    state.livePreview
      ? {
          step: state.livePreview.step,
          total: state.livePreview.total,
          fps: state.livePreview.fps,
          delta: state.livePreview.delta,
        }
      : null
  );
  syncAudioGuardAndSequencer();
  syncEventChipsTrack();
  syncScrubberSVG();
  syncContextualCard();
  syncCueChips();
  syncUndoRedoButtons();
  updateRenderStatusOverlay();
  stage?.render();
}

function syncEventChipsTrack() {
  const track = $("#event-chips-track");
  if (!track) return;
  track.replaceChildren();
  const chips = [];
  if (state.doc.mode === "portal_path" || state.doc.mode === "combined") {
    const pf = evalPortalAtFrame(state.doc, state.frame, state.length);
    chips.push({
      label: `Region @ [${pf.x.toFixed(2)},${pf.y.toFixed(2)}]`,
      cls: "event-chip active",
      onClick: () => {
        state.selectedTargetIdx = 0;
        syncContextualCard();
      },
    });
  }
  if (state.doc.mode === "point_targets" || state.doc.mode === "combined") {
    (state.doc.targets || []).forEach((t, idx) => {
      chips.push({
        label: `E${idx + 1} f${t.start}..${t.end}`,
        title: (t.prompt || "").trim() || `Target ${idx + 1}`,
        cls: idx === state.selectedTargetIdx ? "event-chip active" : "event-chip",
        onClick: () => {
          state.selectedTargetIdx = idx;
          state.frame = clamp(Number(t.start || 0), 0, Math.max(0, state.length - 1));
          if (typeof t.prompt === "string" && t.prompt.trim()) {
            state.targetPrompt = t.prompt;
          }
          syncAllUI();
        },
      });
    });
  }
  if (state.spriteCutout?.ref) {
    const wps = state.doc.dpad?.waypoints || [];
    chips.push({
      label: `Actor (${wps.length} pts)`,
      cls: "event-chip emerald",
      onClick: () => {
        state.spriteMode = true;
        syncAllUI();
      },
    });
  }
  for (const c of allActiveCues()) {
    chips.push({
      label: `3f f${c.frame}: ${c.preset}`,
      cls: "event-chip",
      onClick: () => {
        state.frame = clamp(Number(c.frame || 0), 0, Math.max(0, state.length - 1));
        syncAllUI();
      },
    });
  }
  for (const ch of chips.slice(0, 6)) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = ch.cls;
    b.textContent = ch.label;
    if (ch.title) b.title = ch.title;
    b.addEventListener("click", ch.onClick);
    track.appendChild(b);
  }
}

function syncAudioGuardAndSequencer() {
  const streams = String(state.editGuidance?.streams || "independent");
  const guardBtn = $("#btn-audio-guard");
  if (guardBtn) {
    const labelMap = {
      independent: "Audio: Indep",
      video_only: "Audio: VideoOnly",
      both: "Audio: Gated",
      audio_only: "Audio: AudOnly",
    };
    guardBtn.textContent = labelMap[streams] || `Audio: ${streams}`;
    guardBtn.classList.toggle("active", streams === "independent");
    guardBtn.classList.toggle("emerald", streams === "video_only");
    guardBtn.classList.toggle("warn", streams === "both");
  }

  const chainBtn = $("#btn-chain-frame");
  if (chainBtn) {
    const hasVideo = Boolean(state.activeVideoEl && state.activeStageTake);
    const hasMaskPrev = Boolean(state.maskPreview?.img);
    chainBtn.classList.toggle("hidden", !(hasVideo || hasMaskPrev));
    chainBtn.textContent = hasVideo
      ? `Roll Forward f${Math.max(0, Math.round(state.frame))} ↪`
      : "Apply Preview ↪";
  }

  const muteBtn = $("#btn-stage-mute");
  if (muteBtn) {
    muteBtn.classList.toggle("active", !state.stageMuted);
    muteBtn.setAttribute("aria-pressed", state.stageMuted ? "true" : "false");
    muteBtn.textContent = state.stageMuted ? "Sound: Muted" : "Sound: On";
  }

  const abAudBtn = $("#btn-ab-audio-toggle");
  if (abAudBtn && abCompare) {
    abAudBtn.textContent = abCompare.getAudioLabel();
  }

  const bridgeBtn = $("#btn-shot-bridge");
  if (bridgeBtn) {
    const hasLast = Boolean(state.lastFrameRef);
    const shotNum = state.shotHistory.length + 1;
    const showBridge = hasLast || state.shotHistory.length > 0;
    bridgeBtn.classList.toggle("hidden", !showBridge);
    if (showBridge) {
      bridgeBtn.classList.toggle("active", hasLast);
      bridgeBtn.textContent = hasLast
        ? `Last: ${state.lastFrameName || "frame"} ×`
        : `Shot ${shotNum} · ↩ Shot ${shotNum - 1}`;
    }
  }

  const tray = $("#sequencer-tray");
  const stageEl = $("#stageContainer");
  const lanesBtn = $("#btn-lanes-toggle");
  tray?.classList.toggle("hidden", !state.lanesOpen);
  stageEl?.classList.toggle("lanes-open", Boolean(state.lanesOpen));
  if (lanesBtn) {
    lanesBtn.classList.toggle("active", Boolean(state.lanesOpen));
    lanesBtn.setAttribute("aria-expanded", state.lanesOpen ? "true" : "false");
    lanesBtn.textContent = state.lanesOpen ? "Advanced ▾" : "Advanced ▴";
  }

  const ryBtn = $("#btn-seq-ry-unlink");
  if (ryBtn) {
    const pf = evalPortalAtFrame(state.doc, state.frame, state.length);
    const unlinked = Boolean(state.unlinkPortalRy || pf.shape === "ellipse");
    ryBtn.classList.toggle("active", unlinked);
    ryBtn.setAttribute("aria-pressed", unlinked ? "true" : "false");
    ryBtn.textContent = unlinked ? "ry: Unlinked" : "ry: Linked";
  }

  const delBtn = $("#btn-seq-delete");
  if (delBtn) delBtn.disabled = !state.seqSelection;

  const selLbl = $("#seq-selection-label");
  if (selLbl && sequencer) {
    selLbl.textContent = sequencer.describeSelection(state);
  }
  if (state.lanesOpen) {
    sequencer?.render();
  }
}

// ---------------------------------------------------------------------------
// Actions: Anchor Upload, Anchor Generate, Context Suggest, Stamp Cutout
// ---------------------------------------------------------------------------

async function applyImportedRecipe(recipeDoc) {
  if (!recipeDoc || typeof recipeDoc !== "object") return;
  const st = recipeDoc.stage && typeof recipeDoc.stage === "object" ? recipeDoc.stage : recipeDoc;
  const urls = recipeDoc.ref_view_urls || {};
  pushUndo();

  if (st.length && state.lengths.includes(Number(st.length))) {
    state.length = Number(st.length);
  }
  if (st.seed != null) state.seed = Number(st.seed);
  if (st.mode && state.workflows.some((w) => w.id === st.mode)) {
    state.workflowId = String(st.mode);
    state.workflowManualOverride = true;
  }
  if (st.events_json && typeof st.events_json === "object") {
    state.doc = {
      ...state.doc,
      ...st.events_json,
      width: Number(st.width || st.events_json.width || state.doc.width || 1344),
      height: Number(st.height || st.events_json.height || state.doc.height || 768),
      feather: Number(st.feather ?? st.events_json.feather ?? state.doc.feather ?? 0.08),
      invert_mask: Boolean(st.invert_mask ?? st.events_json.invert_mask ?? false),
    };
    if (st.director_mode) state.doc.mode = String(st.director_mode);
  }
  state.originalPrompt = String(st.original_prompt || "");
  state.targetPrompt = String(st.target_prompt || "");
  state.portalTouched = true;
  state.samePrompt = Boolean(st.same_prompt);
  state.audioRefine = Boolean(st.audio_refine);
  state.snapToIncludeMask = Boolean(st.snap_to_include_mask);
  state.editGuidance = {
    scale: Number(st.scale ?? state.editGuidance.scale ?? 2.0),
    audio_scale: Number(st.audio_scale ?? state.editGuidance.audio_scale ?? 1.0),
    first_step: Number(st.first_step ?? state.editGuidance.first_step ?? 0),
    last_step: Number(st.last_step ?? state.editGuidance.last_step ?? 1000),
    streams: String(st.streams || state.editGuidance.streams || "independent"),
  };
  state.parentTakeId = String(recipeDoc.take_id || recipeDoc.parent_take_id || "");
  state.parentFrame = recipeDoc.parent_frame != null ? Number(recipeDoc.parent_frame) : null;
  state.lastFrameRef = String(st.last_frame || "");
  state.lastFrameName = state.lastFrameRef ? state.lastFrameRef.split("/").pop() : "";

  const cvs = Array.isArray(st.control_vectors) ? st.control_vectors : [];
  if (cvs.length && typeof cvs[0] === "object") {
    state.controlVector = String(cvs[0].vector || cvs[0].name || "none");
    state.controlVectorStrength = Number(cvs[0].strength ?? 1.0);
  } else {
    state.controlVector = "none";
  }

  state.spriteMaskRef = String(st.sprite_mask_ref || "");
  state.includeMaskRef = String(st.include_mask_ref || "");
  state.excludeMaskRef = String(st.exclude_mask_ref || "");
  state.includeMaskImg = null;
  state.excludeMaskImg = null;

  if (urls.source_ref) {
    try {
      await loadAnchorImageUrl(urls.source_ref, {
        ref: st.source_ref,
        name: (st.source_ref || "").split("/").pop() || "Restored Anchor",
        preserveOriginalPrompt: true,
      });
    } catch {}
  }
  if (urls.sprite_ref && st.sprite_ref) {
    await loadCutoutUrl(urls.sprite_ref, st.sprite_ref, {
      keepMaskRef: Boolean(st.sprite_mask_ref),
      maskRef: st.sprite_mask_ref || "",
      fromSceneLift: String(st.sprite_ref).includes("seg_cutout_"),
      preserveWaypoints: true,
      label: st.sprite_label || "",
    });
  } else if (!st.sprite_ref) {
    state.spriteCutout = null;
    state.spriteMode = false;
  }
  if (urls.include_mask_ref) await loadMaskOverlayImage(urls.include_mask_ref, "includeMaskImg");
  if (urls.exclude_mask_ref) await loadMaskOverlayImage(urls.exclude_mask_ref, "excludeMaskImg");

  state.frame = clamp(state.frame, 0, Math.max(0, state.length - 1));
  syncAllUI();
  schedulePreflight();
}

async function handleAnchorUpload(file, opts = {}) {
  if (!file) return;
  if (file.name.toLowerCase().endsWith(".json")) {
    try {
      const rawJson = JSON.parse(await file.text());
      const res = await apiJson("/api/import_recipe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ recipe: rawJson }),
      });
      await applyImportedRecipe(res.recipe);
      showToast(`Imported recipe ${file.name}`, "ok");
    } catch (err) {
      showToast(`Recipe import failed: ${err.message}`, "err");
    }
    return;
  }
  const fd = new FormData();
  fd.append("file", file);
  showToast(`Uploading ${file.name}…`);
  try {
    const data = await apiJson("/api/upload", { method: "POST", body: fd });
    const imgUrl = data.view_url || videoItemViewUrl(data);
    // Clear previous scene prompt, target prompt, actor cutout, & masks so new media never inherits stale context.
    state.originalPrompt = "";
    state.maskPreview = null;
    if (!opts.preserveLineage) {
      state.targetPrompt = "";
      state.portalTouched = false;
      state.parentTakeId = "";
      state.parentFrame = null;
      state.activeMaskPreviewTakeId = "";
      state.spriteCutout = null;
      state.spriteMode = false;
      state.spriteMaskRef = "";
      state.includeMaskRef = "";
      state.excludeMaskRef = "";
      state.includeMaskImg = null;
      state.excludeMaskImg = null;
      state.snapToIncludeMask = false;
      state.samPosPoints = [];
      state.samNegPoints = [];
      state.samPicking = false;
      state.settingSpritePath = false;
      state.couplings = [];
      state.doc.dpad = {
        waypoints: [{ frame: 0, x: 0.36, y: 0.74, face: "right", scale: 1.0 }],
        hops: [],
      };
      const targEl = $("#target-prompt");
      if (targEl) targEl.value = "";
      const samIn = $("#sam-prompt-input");
      if (samIn) samIn.value = "";
      const stampIn = $("#stamp-prompt-input");
      if (stampIn) stampIn.value = "";
    }
    const origEl = $("#original-prompt");
    if (origEl) origEl.value = "";
    await loadAnchorImageUrl(imgUrl, {
      ref: data.ref || `${data.subfolder ? data.subfolder + "/" : ""}${data.filename}`,
      name: file.name,
      width: data.width,
      height: data.height,
      preserveOriginalPrompt: Boolean(opts.preserveLineage),
    });
    if (data.imported_take_id) {
      const snappedLen = Number(data.snapped_length || 0);
      if (snappedLen > 0 && state.lengths.includes(snappedLen) && snappedLen !== state.length) {
        state.length = snappedLen;
        state.frame = clamp(state.frame, 0, state.length - 1);
      }
      await refreshTakes();
      const impTake = state.takes.find((t) => t.take_id === data.imported_take_id);
      if (impTake) {
        loadTakeIntoStage(impTake);
      }
      showToast(`Imported video + Frame 0 (${data.total_frames || "?"}f)`, "ok");
    } else {
      showToast(`Anchored ${file.name}`, "ok");
    }
  } catch (err) {
    showToast(`Upload failed: ${err.message}`, "err");
  }
}

function loadAnchorImageUrl(url, meta = {}) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const prevRef = state.anchorRef;
      if (state.activeVideoEl) {
        try {
          state.activeVideoEl.pause();
          state.activeVideoEl.muted = true;
        } catch {}
      }
      state.anchorMedia = img;
      state.anchorRef = meta.ref || state.anchorRef;
      state.anchorName = meta.name || state.anchorName || "Anchor Still";
      state.hasAnchor = true;
      state.activeVideoEl = null;
      state.activeStageTake = null;
      if (!meta.preserveOriginalPrompt) {
        state.maskPreview = null;
      }
      if (!meta.preserveOriginalPrompt && prevRef && meta.ref && prevRef !== meta.ref) {
        state.spriteMaskRef = "";
        state.includeMaskRef = "";
        state.excludeMaskRef = "";
        state.includeMaskImg = null;
        state.excludeMaskImg = null;
        state.snapToIncludeMask = false;
        state.samPosPoints = [];
        state.samNegPoints = [];
        const samIn = $("#sam-prompt-input");
        if (samIn) samIn.value = "";
        state.samPicking = false;
      }
      $("#empty-anchor-card")?.classList.add("hidden");
      if (state.capabilities.heylook_suggestions) {
        refreshContextSuggestions();
      }
      syncAllUI();
      schedulePreflight();
      resolve(img);
    };
    img.onerror = () => reject(new Error("Could not load anchor image"));
    img.src = url;
  });
}

async function generateAnchorStill() {
  const prompt = ($("#anchor-prompt")?.value || "").trim();
  if (!prompt) {
    showToast("Enter a short anchor scene description first.", "err");
    $("#anchor-prompt")?.focus();
    return;
  }
  const btn = $("#btn-gen-anchor");
  if (btn) btn.disabled = true;
  try {
    const wf = $("#anchor-workflow-select")?.value || state.anchorWorkflowId || "h3_image";
    const data = await apiJson("/api/anchor", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        workflow: wf,
        width: state.doc.width,
        height: state.doc.height,
        seed: state.seed,
        client_id: wsClient.clientId,
      }),
    });
    state.originalPrompt = prompt;
    state.parentTakeId = "";
    state.parentFrame = null;
    state.activeMaskPreviewTakeId = "";
    const origEl = $("#original-prompt");
    if (origEl) origEl.value = prompt;
    state.activeAnchorPromptId = data.prompt_id || "";
    setActiveRender({
      kind: "anchor",
      pid: data.prompt_id || "",
      phase: `Generating Still (${data.workflow || wf})…`,
      step: 0,
      total: (wf === "ph3_image" ? 8 : 6),
    });
    showToast("Generating 1-frame anchor still…", "info");
  } catch (err) {
    showToast(`Anchor generation failed: ${err.message}`, "err");
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function refreshContextSuggestions() {
  const wrap = $("#context-suggestions-wrap");
  const listEl = $("#context-suggestions-list");
  if (!wrap || !listEl) return;
  try {
    const pf = evalPortalAtFrame(state.doc, state.frame, state.length);
    const data = await apiJson("/api/context_suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_ref: state.anchorRef,
        anchor_ref: state.anchorRef,
        original_prompt: state.originalPrompt,
        scene_prompt: state.originalPrompt,
        prompt: state.targetPrompt,
        mode: state.doc.mode,
        x: pf.x,
        y: pf.y,
        rx: pf.rx,
        ry: pf.ry,
      }),
    });
    if (data.enabled) {
      state.vlmSubjectLabel = String(data.subject_label || "").trim();
      const sugLabel = $("#context-suggestions-wrap .field-label");
      if (sugLabel) {
        sugLabel.textContent = state.vlmSubjectLabel
          ? `Context Suggestions (${state.vlmSubjectLabel})`
          : "Context Suggestions (heylook)";
      }
      if (data.scene_caption && !state.originalPrompt.trim()) {
        const op = $("#original-prompt");
        if (op) op.placeholder = `Auto-detected: ${data.scene_caption}`;
      }
    }
    if (data.enabled && Array.isArray(data.suggestions) && data.suggestions.length > 0) {
      listEl.replaceChildren();
      for (const s of data.suggestions) {
        const promptText = typeof s === "string" ? s : (s.target_prompt || s.prompt || s.title || s.label || "");
        const titleText = typeof s === "object" && s.title ? s.title : promptText;
        if (!promptText) continue;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "suggestion-btn";
        btn.textContent = titleText;
        btn.addEventListener("click", () => {
          pushUndo();
          state.targetPrompt = promptText;
          const tp = $("#target-prompt");
          if (tp) tp.value = promptText;
          if (typeof s === "object" && s.preset && state.soundPresetsList.includes(s.preset)) {
            const f = snapFrameTo3f(state.frame, state.length);
            state.doc.portal_cues = (state.doc.portal_cues || []).filter((c) => c.frame !== f);
            state.doc.portal_cues.push({ frame: f, preset: s.preset, gain: 1.0 });
          }
          syncAllUI();
        });
        listEl.appendChild(btn);
      }
      wrap.classList.remove("hidden");
    } else {
      listEl.replaceChildren();
      wrap.classList.add("hidden");
    }
  } catch {
    listEl.replaceChildren();
    wrap.classList.add("hidden");
  }
}

async function generateOrExtractStamp() {
  const prompt = ($("#stamp-prompt-input")?.value || "").trim();
  if (!state.anchorRef && !prompt) {
    showToast("Anchor a scene or enter a spawn actor prompt in Engine -> Steering first.", "err");
    return;
  }
  try {
    const data = await apiJson("/api/stamp_cutout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_ref: state.anchorRef || undefined,
        prompt: prompt || undefined,
        workflow: state.anchorWorkflowId || "h3_image",
        width: state.doc.width,
        height: state.doc.height,
        seed: state.seed,
        client_id: wsClient.clientId,
      }),
    });
    if (data.prompt_id) {
      state.activeStampPromptId = data.prompt_id;
      setActiveRender({
        kind: "stamp",
        pid: data.prompt_id,
        phase: "Generating Actor Cutout…",
        step: 0,
        total: 6,
      });
      showToast("Generating RGBA actor cutout…", "info");
      return;
    }
    const url = data.view_url || videoItemViewUrl(data);
    if (url) {
      await loadCutoutUrl(url, data.cutout_ref || data.ref || "", {
        keepMaskRef: Boolean(data.cutout_mask_ref),
        maskRef: data.cutout_mask_ref || "",
        fromSceneLift: false,
        label: prompt,
        bbox: data.bbox || null,
      });
      showToast("RGBA cutout ready (use Set Path or WASD)", "ok");
    }
  } catch (err) {
    showToast(`Stamp failed: ${err.message}`, "err");
  }
}

async function previewInMask() {
  if (!state.anchorRef) {
    showToast("Load or generate an anchor scene first.", "err");
    return;
  }
  const cmd = (state.targetPrompt || "").trim();
  if (!cmd || cmd.includes("[target object]")) {
    showToast("Enter or replace [target object] with a concrete action/object first.", "err");
    $("#target-prompt")?.focus();
    return;
  }
  const btn = $("#btn-preview-mask");
  if (btn) btn.disabled = true;
  try {
    let region;
    if (state.doc.mode === "point_targets" && state.doc.targets?.length) {
      const idx = clamp(state.selectedTargetIdx ?? 0, 0, state.doc.targets.length - 1);
      const t = state.doc.targets[idx];
      region = {
        x: t.x,
        y: t.y,
        rx: t.radius,
        ry: t.radius,
        shape: "circle",
        weight: t.weight ?? 1.0,
        halo_radius: t.halo_radius ?? 0.0,
        halo_weight: t.halo_weight ?? 0.30,
      };
    } else {
      const pf = evalPortalAtFrame(state.doc, state.frame, state.length);
      region = {
        x: pf.x,
        y: pf.y,
        rx: pf.rx,
        ry: pf.ry,
        shape: pf.shape || "circle",
        weight: pf.weight ?? 1.0,
        halo_radius: pf.halo_radius ?? 0.0,
        halo_weight: pf.halo_weight ?? 0.30,
      };
    }
    const wf = state.previewWorkflowId || "auto";
    const res = await apiJson("/api/preview_in_mask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_ref: state.anchorRef,
        target_prompt: cmd,
        original_prompt: (state.originalPrompt || "").trim(),
        workflow: wf,
        parent_take_id: state.parentTakeId || undefined,
        parent_frame: state.parentFrame != null ? Number(state.parentFrame) : undefined,
        region,
        feather: Number(state.doc.feather),
        invert_mask: Boolean(state.doc.invert_mask),
        include_mask_ref: state.includeMaskRef || "",
        exclude_mask_ref: state.excludeMaskRef || "",
        snap_to_include_mask: Boolean(state.snapToIncludeMask),
        scale: Number(state.editGuidance.scale),
        first_step: Number(state.editGuidance.first_step),
        last_step: Number(state.editGuidance.last_step),
        seed: Number($("#seed-input")?.value || state.seed),
        client_id: wsClient.clientId,
      }),
    });
    state.activeMaskPreviewPromptId = res.prompt_id || "";
    state.activeMaskPreviewTakeId = res.take_id || "";
    if (res.resolved_source_ref) state.anchorRef = res.resolved_source_ref;
    if (res.resident_backbone) state.residentBackbone = res.resident_backbone;
    const resolvedWf = res.workflow || wf;
    setActiveRender({
      kind: "mask_preview",
      pid: res.prompt_id || "",
      phase: `Mask Preview (${resolvedWf})…`,
      step: 0,
      total: resolvedWf === "ph3_image" ? 8 : 6,
      meta: {
        region,
        feather: Number(state.doc.feather),
        snapToInclude: Boolean(state.snapToIncludeMask),
      },
    });
    showToast(`Running masked 1-frame preview (${resolvedWf} · taeh3_image)…`, "info");
  } catch (err) {
    showToast(`Mask preview failed: ${err.message}`, "err");
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function chainMaskPreviewToAnchor() {
  if (!state.anchorMedia || !state.maskPreview?.img) return;
  const mw = state.anchorMedia.naturalWidth || state.doc.width || 1344;
  const mh = state.anchorMedia.naturalHeight || state.doc.height || 768;
  const c = document.createElement("canvas");
  c.width = mw;
  c.height = mh;
  const cx = c.getContext("2d");
  cx.drawImage(state.anchorMedia, 0, 0, mw, mh);
  const compCanvas = document.createElement("canvas");
  compCanvas.width = mw;
  compCanvas.height = mh;
  const tCtx = compCanvas.getContext("2d");
  tCtx.drawImage(state.maskPreview.img, 0, 0, mw, mh);
  tCtx.globalCompositeOperation = "destination-in";
  if (state.maskPreview.snapToInclude && state.includeMaskImg) {
    tCtx.drawImage(state.includeMaskImg, 0, 0, mw, mh);
  } else if (stage) {
    stage._drawFeatheredMaskShape(tCtx, mw, mh, Math.min(mw, mh), {
      x: state.maskPreview.x,
      y: state.maskPreview.y,
      rx: state.maskPreview.rx,
      ry: state.maskPreview.ry,
      shape: state.maskPreview.shape,
      feather: state.maskPreview.feather,
      haloRadius: state.maskPreview.haloRadius,
      haloWeight: state.maskPreview.haloWeight,
      tw: 1.0,
    });
  }
  cx.drawImage(compCanvas, 0, 0, mw, mh);
  const blob = await new Promise((res) => c.toBlob(res, "image/png"));
  if (!blob) return;
  const file = new File([blob], "sandbox_roll_forward.png", { type: "image/png" });
  const prevMaskTid = state.activeMaskPreviewTakeId || state.parentTakeId || "";
  await handleAnchorUpload(file, { preserveLineage: true });
  state.parentTakeId = prevMaskTid;
  state.parentFrame = 0;
  showToast("Rolled preview forward as new World State", "ok");
}

function alignUntouchedPortalToSpritePath() {
  if (state.portalTouched || (state.targetPrompt || "").trim() || state.doc.mode !== "portal_path") {
    return;
  }
  const wps = [...(state.doc.dpad?.waypoints || [])]
    .filter((w) => w && typeof w === "object")
    .sort((a, b) => Number(a.frame || 0) - Number(b.frame || 0));
  if (wps.length < 2 || (state.doc.portal || []).length > 2) return;
  const tpl = (state.doc.portal || [])[0] || {
    rx: 0.22,
    ry: 0.22,
    shape: "circle",
    label: "REALITY LENS",
    weight: 1.0,
    halo_radius: 0.10,
    halo_weight: 0.30,
  };
  const w0 = wps[0];
  const w1 = wps[wps.length - 1];
  const f0 = clamp(Math.round(Number(w0.frame || 0)), 0, Math.max(0, state.length - 1));
  const f1 = clamp(Math.round(Number(w1.frame || (state.length - 1))), f0, Math.max(0, state.length - 1));
  const mk = (fr, wp) => ({
    frame: fr,
    x: Number(clamp(Number(wp.x ?? 0.5), 0.04, 0.96).toFixed(3)),
    y: Number(clamp(Number(wp.y ?? 0.5), 0.06, 0.94).toFixed(3)),
    rx: Number(tpl.rx ?? 0.22),
    ry: Number(tpl.ry ?? tpl.rx ?? 0.22),
    shape: tpl.shape || "circle",
    label: tpl.label || "REALITY LENS",
    weight: Number(tpl.weight ?? 1.0),
    halo_radius: Number(tpl.halo_radius ?? 0.10),
    halo_weight: Number(tpl.halo_weight ?? 0.30),
  });
  state.doc.portal = f1 > f0 ? [mk(f0, w0), mk(f1, w1)] : [mk(0, w1)];
}

function loadCutoutUrl(url, ref, opts = {}) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      if (opts.maskRef !== undefined) {
        state.spriteMaskRef = String(opts.maskRef || "");
      } else if (!opts.keepMaskRef) {
        state.spriteMaskRef = "";
      }
      if (opts.fromSceneLift === false && !opts.preserveWaypoints) {
        if (!state.doc.dpad) state.doc.dpad = { waypoints: [], hops: [] };
        if (!Array.isArray(state.doc.dpad.waypoints)) state.doc.dpad.waypoints = [];
        const wps = state.doc.dpad.waypoints;
        const pf = evalPortalAtFrame(state.doc, state.frame, state.length);
        const isDefaultSpot =
          wps.length === 1
          && Math.abs(Number(wps[0].x) - 0.36) < 0.005
          && Math.abs(Number(wps[0].y) - 0.74) < 0.005;
        if (wps.length === 0) {
          wps.push({
            frame: 0,
            x: Number(clamp(pf.x, 0.08, 0.92).toFixed(3)),
            y: Number(clamp(pf.y, 0.10, 0.90).toFixed(3)),
            face: "right",
            scale: 1.0,
          });
        } else if (isDefaultSpot) {
          wps[0].x = Number(clamp(pf.x, 0.08, 0.92).toFixed(3));
          wps[0].y = Number(clamp(pf.y, 0.10, 0.90).toFixed(3));
        }
      }
      state.spriteCutout = {
        ref,
        viewUrl: url,
        img,
        bboxHeight: Number(opts.bbox?.h || 0),
        fromSceneLift: Boolean(opts.fromSceneLift),
        label: String(opts.label || ""),
      };
      state.spriteMode = true;
      syncAllUI();
      resolve(img);
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

function loadMaskOverlayImage(url, field) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      state[field] = img;
      stage?.render();
      resolve(img);
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

function clearSamMasks() {
  state.samPicking = false;
  state.spriteMaskRef = "";
  state.includeMaskRef = "";
  state.excludeMaskRef = "";
  state.includeMaskImg = null;
  state.excludeMaskImg = null;
  state.samPosPoints = [];
  state.samNegPoints = [];
  const samIn = $("#sam-prompt-input");
  if (samIn) samIn.value = "";
  syncAllUI();
  schedulePreflight();
}

async function executeSamSegmentation() {
  if (!state.anchorRef || state.samSegmenting) return;
  const promptText = ($("#sam-prompt-input")?.value || "").trim();
  if (!state.samPosPoints.length && !state.samNegPoints.length && !promptText) {
    showToast("Click subject on stage or enter a SAM text prompt.", "info", 2000);
    return;
  }
  const role = state.samRole || "include";
  state.samSegmenting = true;
  syncAllUI();
  showToast(`Segmenting subject (${role})…`, "info");
  try {
    const data = await apiJson("/api/segment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_ref: state.anchorRef,
        positive_points: state.samPosPoints,
        negative_points: state.samNegPoints,
        prompt: promptText,
        combine_text_and_points: Boolean(promptText && state.samPosPoints.length),
        role,
      }),
    });
    state.samEngine = data.engine || "";
    if (data.resolved_source_ref) state.anchorRef = data.resolved_source_ref;
    if (data.role === "cutout") {
      state.spriteMaskRef = data.cutout_mask_ref || "";
      const wps = state.doc.dpad?.waypoints || [];
      if (
        data.bbox
        && wps.length === 1
        && Math.abs(Number(wps[0].x) - 0.36) < 0.005
        && Math.abs(Number(wps[0].y) - 0.74) < 0.005
      ) {
        wps[0].x = Number(Number(data.bbox.cx).toFixed(3));
        wps[0].y = Number(Number(data.bbox.cy).toFixed(3));
      }
      if (data.cutout_view_url) {
        await loadCutoutUrl(data.cutout_view_url, data.cutout_ref || "", {
          keepMaskRef: true,
          maskRef: data.cutout_mask_ref || "",
          fromSceneLift: true,
          label: promptText || state.vlmSubjectLabel || "",
          bbox: data.bbox || null,
        });
      }
    } else if (data.role === "exclude") {
      state.excludeMaskRef = data.mask_ref || "";
      if (data.mask_view_url) {
        await loadMaskOverlayImage(data.mask_view_url, "excludeMaskImg");
      }
    } else if (data.role === "include") {
      state.includeMaskRef = data.mask_ref || "";
      if (data.mask_view_url) {
        await loadMaskOverlayImage(data.mask_view_url, "includeMaskImg");
      }
    }
    const pct = Math.round(Number(data.coverage || 0) * 100);
    showToast(`Segmented ${data.role} via ${data.engine || "sam"} (${pct}%) · +${state.samPosPoints.length}/−${state.samNegPoints.length}`, "ok");
    schedulePreflight();
  } catch (err) {
    showToast(`Segmentation failed: ${err.message}`, "err");
  } finally {
    state.samSegmenting = false;
    state.samPicking = false;
    syncAllUI();
  }
}

async function runSamSegmentAt(nx, ny, opts = {}) {
  if (!state.anchorRef || state.samSegmenting) return;
  const pt = { nx: Number(nx.toFixed(4)), ny: Number(ny.toFixed(4)) };
  if (opts.negative) {
    state.samNegPoints = [...state.samNegPoints, pt].slice(-12);
  } else {
    state.samPosPoints = [...state.samPosPoints, pt].slice(-12);
  }
  stage?.render();
  await executeSamSegmentation();
}

// ---------------------------------------------------------------------------
// Compile, Render, Escalate, Takes, Forget
// ---------------------------------------------------------------------------

function buildCompilePayload() {
  const cuesLines = allActiveCues()
    .map((c) => `${c.frame}: ${c.preset} @${Number(c.gain || 1).toFixed(2)}`)
    .join("\n");
  const dpadLines = (state.doc.dpad?.waypoints || [])
    .map((w) => `${w.frame}: x=${Number(w.x).toFixed(3)}, y=${Number(w.y).toFixed(3)}, face=${w.face || "right"}, scale=${Number(w.scale || 1).toFixed(2)}`)
    .concat(
      (state.doc.dpad?.hops || []).map((h) => `hop ${h.start}..${h.end}: lift=${Number(h.lift || 0.25).toFixed(2)}`)
    )
    .join("\n");

  const mode = resolvedWorkflowMode();

  const pf = evalPortalAtFrame(state.doc, state.frame, state.length);
  const sp = (state.spriteMode || state.spriteCutout?.img || (state.couplings || []).length > 0)
    ? evalSpriteAtFrame(state.doc, state.frame, state.length)
    : null;
  const coupling = computeCouplingInfo(state, pf, sp);
  const activeCouplings = coupling.active
    ? [{
        portal_index: 0,
        target_kind: "sprite",
        strength: coupling.strength,
        mode: coupling.mode,
        label: "Sprite Coupling",
      }]
    : [];

  return {
    mode,
    director_mode: state.doc.mode,
    workflow: mode,
    width: state.doc.width,
    height: state.doc.height,
    length: state.length,
    seed: Number($("#seed-input")?.value || state.seed),
    source_ref: state.anchorRef,
    last_frame: state.lastFrameRef || "",
    sprite_ref: state.spriteCutout?.ref || "",
    sprite_mask_ref: state.spriteMaskRef || "",
    sprite_label: state.spriteCutout?.label || "",
    include_mask_ref: state.includeMaskRef || "",
    exclude_mask_ref: state.excludeMaskRef || "",
    snap_to_include_mask: Boolean(state.snapToIncludeMask),
    include_sprite_envelope: Boolean(
      (state.spriteCutout?.ref
        || (state.doc.dpad?.waypoints?.length || 0) > 1
        || (state.doc.dpad?.hops?.length || 0) > 0)
      && !state.doc.invert_mask
    ),
    sprite_pos: sp ? { x: sp.x, y: sp.y, face: sp.face } : null,
    halo_radius: Number(pf.halo_radius || 0),
    couplings: activeCouplings,
    original_prompt: (state.originalPrompt || "").trim(),
    target_prompt: state.targetPrompt,
    same_prompt: Boolean(state.samePrompt),
    audio_refine: Boolean(state.audioRefine),
    feather: Number(state.doc.feather),
    invert_mask: Boolean(state.doc.invert_mask),
    scale: Number(state.editGuidance.scale),
    audio_scale: Number(state.editGuidance.audio_scale),
    first_step: Number(state.editGuidance.first_step),
    last_step: Number(state.editGuidance.last_step),
    streams: state.editGuidance.streams,
    control_vectors: state.controlVector && state.controlVector !== "none"
      ? [{ vector: state.controlVector, strength: Number(state.controlVectorStrength), edit_pass: "edited_only" }]
      : [],
    queue_baseline: Boolean($("#chk-baseline")?.checked),
    events_json: JSON.stringify(state.doc),
    sound_cues: cuesLines,
    dpad_keyframes: dpadLines,
    parent_take_id: state.parentTakeId || undefined,
    parent_frame: state.parentFrame != null ? Number(state.parentFrame) : undefined,
    client_id: wsClient.clientId,
  };
}

async function renderPortalEdit() {
  const wfMode = resolvedWorkflowMode();
  if (!state.hasAnchor && wfMode !== "t2v_portal") {
    showToast("Upload/drop an image or generate a Still Scene first (or switch Engine → Workflow to Text-to-Video).", "err", 4200);
    $("#anchor-prompt")?.focus();
    return;
  }
  const hasSpritePath = Boolean(
    state.spriteCutout?.ref
    || (state.doc.dpad?.waypoints?.length || 0) > 1
    || (state.doc.dpad?.hops?.length || 0) > 0
  );
  const cmd = state.targetPrompt.trim();
  if ((!cmd || cmd.includes("[target object]")) && !state.samePrompt && state.controlVector === "none" && !hasSpritePath) {
    showToast("Enter a concrete action or target object in Action or Spawn Command first.", "err");
    $("#target-prompt")?.focus();
    return;
  }
  if (wfMode === "i2v_portal" && state.length !== 1 && state.length < 124) {
    showToast(`I2V requires >= 124 frames (use 141f+ in header, or switch to R2V for ${state.length}f).`, "err", 4500);
    return;
  }
  const btn = $("#btn-render");
  if (btn) btn.disabled = true;
  try {
    const payload = buildCompilePayload();
    const res = await apiJson("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.activeRenderPromptId = res.prompt_id || "";
    state.activeBaselinePromptId = res.baseline_prompt_id || "";
    setActiveRender({
      kind: "render",
      pid: res.prompt_id || "",
      phase: `Generating ${state.length}f World Clip (${payload.mode})…`,
      step: 0,
      total: 6,
    });
    showToast(`Queued ${state.length}f clip (${res.preview_override_injected ? "live taeh3 preview" : "standard"})`, "ok");
    await refreshTakes();
  } catch (err) {
    clearActiveRender();
    showToast(`Render error: ${err.message}`, "err");
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function switchLength(nextLength) {
  const targetLen = Number(nextLength);
  if (!targetLen || targetLen === state.length) return;
  try {
    const res = await apiJson("/api/escalate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        length: state.length,
        target_length: targetLen,
        events_json: JSON.stringify(state.doc),
        portal_cues: state.doc.portal_cues || [],
      }),
    });
    pushUndo();
    state.length = Number(res.target_length || targetLen);
    if (res.events_json) {
      const parsed = typeof res.events_json === "string" ? JSON.parse(res.events_json) : res.events_json;
      if (parsed && typeof parsed === "object") {
        state.doc = { ...state.doc, ...parsed };
      }
    }
    if (Array.isArray(res.portal_cues)) {
      state.doc.portal_cues = res.portal_cues;
    }
    state.frame = clamp(state.frame, 0, Math.max(0, state.length - 1));
    syncAllUI();
    schedulePreflight();
    showToast(`Scaled choreography to ${state.length}f`, "ok");
  } catch (err) {
    showToast(`Scale failed: ${err.message}`, "err");
  }
}

async function promoteSeedToHero() {
  const larger = state.lengths.find((n) => n > state.length);
  const heroTarget = larger || state.lengths[state.lengths.length - 1] || state.length;
  state.seedLocked = true;
  $("#btn-seed-lock")?.classList.add("active");
  $("#btn-seed-lock")?.setAttribute("aria-pressed", "true");
  await switchLength(heroTarget);
}

function rehydrateInFlightRender() {
  takesUI?.rehydrateInFlightRender();
}

async function refreshTakes() {
  await takesUI?.refreshTakes();
}

function renderTakesList() {
  takesUI?.renderTakesList();
}

function findBaselineTake(directedTake) {
  return takesUI ? takesUI.findBaselineTake(directedTake) : null;
}

function loadTakeIntoStage(take) {
  takesUI?.loadTakeIntoStage(take);
}

async function chainTakeFrameToAnchor(take, frameIndex = -1, btn = null, opts = {}) {
  await takesUI?.chainTakeFrameToAnchor(take, frameIndex, btn, opts);
}

async function redubTakeAudio(take, btn = null) {
  await takesUI?.redubTakeAudio(take, btn);
}

function syncMediaPlayback() {
  foley.setMuted(Boolean(state.stageMuted));
  if (abCompare?.active) {
    if (state.activeVideoEl && !state.activeVideoEl.paused) {
      try { state.activeVideoEl.pause(); } catch {}
    }
    if (state.activeVideoEl) state.activeVideoEl.muted = true;
    abCompare.syncPlayback({ playing: state.playing, stageMuted: state.stageMuted, loop: state.loop });
    return;
  }
  const v = state.activeVideoEl;
  if (!v) return;
  v.loop = Boolean(state.loop);
  v.muted = Boolean(state.stageMuted || !state.playing);
  if (state.playing && v.readyState >= 2) {
    if (v.paused) {
      v.play().catch(() => {
        v.muted = true;
        v.play().catch(() => {});
      });
    }
  } else if (!v.paused) {
    try { v.pause(); } catch {}
  }
}

function openABCompareForTake(take = null) {
  let target = take || state.activeStageTake || state.takes.find((t) => t.clean_video || t.hud_video) || null;
  if (target?.take_id) {
    target = state.takes.find((t) => t.take_id === target.take_id) || target;
    if (!take && state.activeStageTake?.take_id === target.take_id) {
      state.activeStageTake = target;
    }
  }
  if (!abCompare.active && !target) {
    showToast("Render or import a video take first to compare A|B.", "info", 2800);
    return;
  }
  if (target) {
    abCompare.setTakes({
      directedTake: target,
      baselineTake: findBaselineTake(target),
    });
  }
  const active = abCompare.toggle();
  $("#btn-compare-ab")?.classList.toggle("active", active);
  $("#ab-bar")?.classList.toggle("hidden", !active);
  syncMediaPlayback();
  if (active && target) {
    abCompare.fetchMetrics({
      events_json: JSON.stringify(state.doc),
      director_mode: state.doc.mode,
      feather: state.doc.feather,
      width: state.doc.width,
      height: state.doc.height,
    });
  }
  syncContextualCard();
  syncAudioGuardAndSequencer();
  stage?.render();
}

// ---------------------------------------------------------------------------
// Playback Loop (24 fps) & Scrubber Drag
// ---------------------------------------------------------------------------

function stepPlayback(ts) {
  if (!state.playing) {
    rafHandle = 0;
    return;
  }

  const abMaster = abCompare?.active ? abCompare.getMasterProgress() : null;
  const stageVideo = (!abCompare?.active && state.activeVideoEl && state.activeVideoEl.readyState >= 2 && Number.isFinite(state.activeVideoEl.duration) && state.activeVideoEl.duration > 0)
    ? state.activeVideoEl
    : null;

  if (abMaster && !abMaster.paused) {
    if (!state.loop && abMaster.ended) {
      state.frame = Math.max(0, state.length - 1);
      state.playing = false;
      syncMediaPlayback();
      syncAllUI();
      rafHandle = 0;
      return;
    }
    const nextFrame = clamp(Math.round(abMaster.progress * Math.max(1, state.length - 1)), 0, Math.max(0, state.length - 1));
    if (nextFrame !== state.frame) {
      state.frame = nextFrame;
      syncAllUI();
    } else {
      stage?.render();
    }
    abCompare.syncPlayback({ playing: state.playing, stageMuted: state.stageMuted, loop: state.loop });
    rafHandle = requestAnimationFrame(stepPlayback);
    return;
  }

  if (stageVideo && !stageVideo.paused) {
    if (!state.loop && stageVideo.ended) {
      state.frame = Math.max(0, state.length - 1);
      state.playing = false;
      syncMediaPlayback();
      syncAllUI();
      rafHandle = 0;
      return;
    }
    const frac = Math.max(0, Math.min(1, stageVideo.currentTime / stageVideo.duration));
    const nextFrame = clamp(Math.round(frac * Math.max(1, state.length - 1)), 0, Math.max(0, state.length - 1));
    if (nextFrame !== state.frame) {
      state.frame = nextFrame;
      syncAllUI();
    } else {
      stage?.render();
    }
    rafHandle = requestAnimationFrame(stepPlayback);
    return;
  }

  if (!lastRafTime) lastRafTime = ts;
  const dt = (ts - lastRafTime) / 1000;
  const frameStep = Math.floor(dt * FPS);
  if (frameStep >= 1) {
    lastRafTime = ts - ((dt * FPS - frameStep) / FPS) * 1000;
    const prevFrame = state.frame;
    let nextFrame = prevFrame + frameStep;
    if (nextFrame >= state.length) {
      if (state.loop) {
        nextFrame = nextFrame % state.length;
      } else {
        nextFrame = state.length - 1;
        state.playing = false;
        syncMediaPlayback();
      }
    }
    state.frame = nextFrame;

    // Trigger 3f foley cues only when not superseded by unmuted rendered video audio
    const audibleVideoPlaying = Boolean((abCompare?.active && abMaster?.audible) || (stageVideo && !state.stageMuted && !stageVideo.paused));
    if (!state.stageMuted && !audibleVideoPlaying) {
      for (const c of allActiveCues()) {
        const crossed = nextFrame >= prevFrame
          ? (c.frame > prevFrame && c.frame <= nextFrame)
          : (c.frame > prevFrame || c.frame <= nextFrame);
        if (crossed) {
          const pan = clamp((c.x ?? 0.5) * 2 - 1, -1, 1);
          foley.playCue(c.preset, c.gain, state.soundPresetMap[c.preset] || 0.2, pan);
        }
      }
    }
    syncAllUI();
  }
  rafHandle = requestAnimationFrame(stepPlayback);
}

function togglePlayback() {
  foley.unlock();
  state.playing = !state.playing;
  syncMediaPlayback();
  if (state.playing) {
    lastRafTime = 0;
    rafHandle = requestAnimationFrame(stepPlayback);
  } else if (rafHandle) {
    cancelAnimationFrame(rafHandle);
    rafHandle = 0;
  }
  syncAllUI();
}

function scrubToClientX(clientX) {
  const wrap = $("#scrubber-track-wrap");
  if (!wrap) return;
  const rect = wrap.getBoundingClientRect();
  const pad = 10;
  const u = clamp((clientX - rect.left - pad) / Math.max(1, rect.width - pad * 2), 0, 1);
  state.frame = clamp(Math.round(u * (state.length - 1)), 0, Math.max(0, state.length - 1));
  syncAllUI();
}

// ---------------------------------------------------------------------------
// Accessible Engine Drawer (`role="tablist"` with ArrowLeft/Right/Home/End)
// ---------------------------------------------------------------------------

function activateEngineTab(tabBtn) {
  const tabs = $$("#engine-tabs [role='tab']");
  const panels = $$("#engine-drawer [role='tabpanel']");
  const targetPanelId = tabBtn.getAttribute("aria-controls");
  for (const t of tabs) {
    const active = t === tabBtn;
    t.classList.toggle("active", active);
    t.setAttribute("aria-selected", active ? "true" : "false");
    t.setAttribute("tabindex", active ? "0" : "-1");
  }
  for (const p of panels) {
    const show = p.id === targetPanelId;
    p.classList.toggle("active", show);
    p.classList.toggle("hidden", !show);
  }
}

function toggleEngineDrawer(forceOpen) {
  const drawer = $("#engine-drawer");
  const btn = $("#btn-engine");
  if (!drawer || !btn) return;
  const open = typeof forceOpen === "boolean" ? forceOpen : drawer.classList.contains("hidden");
  drawer.classList.toggle("hidden", !open);
  btn.classList.toggle("active", open);
  btn.setAttribute("aria-expanded", open ? "true" : "false");
  if (open) {
    pollSystemStats();
    runPreflight();
    refreshTakes();
  }
}

// ---------------------------------------------------------------------------
// Initialization & Event Wiring
// ---------------------------------------------------------------------------

function init() {
  const canvas = $("#stage-canvas");
  const sink = $("#offscreen-media-sink");

  abCompare = new ABCompareController({
    onChange: () => {
      const m = abCompare.metrics;
      const mEl = $("#ab-metrics-text");
      if (mEl) {
        if (abCompare.metricsLoading) {
          mEl.textContent = "Computing mask MAE localization…";
        } else if (m && m.available) {
          mEl.textContent = `In-Mask MAE: ${Number(m.in_mask_mae).toFixed(3)} · Out-Mask MAE: ${Number(m.out_mask_mae).toFixed(3)} · Focus: ${Number(m.focus_ratio).toFixed(2)}x`;
        } else if (m && m.reason) {
          mEl.textContent = `A|B Split · ${m.reason}`;
        } else {
          mEl.textContent = "Split A|B · Drag divider across stage";
        }
      }
      const hudBtn = $("#btn-ab-hud-toggle");
      if (hudBtn) {
        hudBtn.textContent = abCompare.hudMode ? "B: HUD (Node 13)" : "B: Clean (Node 162)";
      }
      stage?.render();
    },
    onStatus: (msg, kind) => showToast(msg, kind),
  });
  abCompare.setVideoSink(sink);

  takesUI = createTakesController({
    state,
    $,
    apiJson,
    showToast,
    getWsClient: () => wsClient,
    getAbCompare: () => abCompare,
    getStage: () => stage,
    setActiveRender,
    syncMediaPlayback,
    syncAllUI,
    schedulePreflight,
    loadAnchorImageUrl,
    applyImportedRecipe,
    onOpenABCompare: (take) => openABCompareForTake(take),
  });

  stage = new StageController(canvas, {
    getState: () => state,
    abCompare,
    onMutate: (reason, opts = {}) => {
      if (!opts.live) pushUndo();
      if (typeof reason === "string" && (reason.startsWith("portal_") || reason.startsWith("target_"))) {
        state.portalTouched = true;
      } else if (reason === "set_sprite_path" || reason === "sprite_waypoint") {
        alignUntouchedPortalToSpritePath();
      }
      syncAllUI();
      if (!opts.live) schedulePreflight();
    },
    onCardInspect: () => {
      syncContextualCard();
    },
    onInteractHud: () => {
      syncAllUI();
    },
    onSamPick: (nx, ny, opts) => runSamSegmentAt(nx, ny, opts),
  });

  sequencer = new SequencerController($("#sequencer-canvas"), {
    getState: () => state,
    onMutate: (_reason, opts = {}) => {
      if (opts.beforeEdit) {
        pushUndo();
        return;
      }
      if (!opts.live) schedulePreflight();
      syncAllUI();
    },
    onSelect: () => {
      syncAllUI();
    },
    onScrub: () => {
      syncAllUI();
    },
  });

  wsClient = new DirectorWSClient({
    onConnection: (status) => {
      if (status === "disconnected") {
        $("#live-tae-text").textContent = "Live TAE: Reconnecting…";
      } else if (status === "connected") {
        syncLiveTaeBadges();
      }
    },
    onProgress: (d) => {
      if (d && d.max > 0) {
        syncLiveTaeBadges({ step: d.value, total: d.max, fps: null });
        if (state.activeRender) {
          state.activeRender.step = Number(d.value || 0);
          state.activeRender.total = Number(d.max || 0);
          updateRenderStatusOverlay();
        }
      }
    },
    onPreviewTelemetry: (d) => {
      if (d && d.total > 0) {
        syncLiveTaeBadges({ step: d.step || 0, total: d.total, fps: d.fps, delta: d.delta });
        if (state.activeRender) {
          state.activeRender.step = Number(d.step || 0);
          state.activeRender.total = Number(d.total || 0);
          if (d.step_ms !== undefined && d.step_ms !== null) state.activeRender.stepMs = d.step_ms;
          if (d.avg_step_ms !== undefined && d.avg_step_ms !== null) state.activeRender.avgStepMs = d.avg_step_ms;
          if (d.delta !== undefined && d.delta !== null) state.activeRender.delta = d.delta;
          updateRenderStatusOverlay();
        }
      }
    },
    onPreviewMedia: (item) => {
      state.livePreview = {
        ...item,
        fullFrame: state.livePreview?.fullFrame || Boolean(state.activeAnchorPromptId || state.activeStampPromptId),
      };
      syncLiveTaeBadges({ step: item.step || 0, total: item.total || 0, fps: item.fps, delta: item.delta });
      if (state.activeRender && item.total > 0) {
        state.activeRender.step = Number(item.step || 0);
        state.activeRender.total = Number(item.total || 0);
        if (item.stepMs !== undefined && item.stepMs !== null) state.activeRender.stepMs = item.stepMs;
        if (item.avgStepMs !== undefined && item.avgStepMs !== null) state.activeRender.avgStepMs = item.avgStepMs;
        if (item.delta !== undefined && item.delta !== null) state.activeRender.delta = item.delta;
        updateRenderStatusOverlay();
      }
      stage?.render();
    },
    onExecuted: async (d) => {
      const pid = d?.prompt_id;
      await refreshTakes();
      if (pid && await handleFinishedTakeById(pid)) {
        return;
      }
      if (!pid && state.activeRender?.pid) {
        await handleFinishedTakeById(state.activeRender.pid);
        return;
      }
      const fallback = state.takes.find((t) => t.clean_video || t.hud_video);
      if (!pid && fallback && state.activeRenderPromptId) {
        state.activeRenderPromptId = "";
        clearActiveRender();
        loadTakeIntoStage(fallback);
        showToast("World clip render complete", "ok");
      }
    },
    onExecutionError: (type, d) => {
      clearActiveRender();
      showToast(`${type}: ${d?.exception_message || "Execution stopped"}`, "err", 6000);
    },
  });
  wsClient.setVideoSink(sink);
  wsClient.connect();

  // Header controls
  $("#btn-undo")?.addEventListener("click", () => {
    if (!state.undoStack.length) return;
    state.redoStack.push(snapshotDoc());
    restoreSnapshot(state.undoStack.pop());
  });
  $("#btn-redo")?.addEventListener("click", () => {
    if (!state.redoStack.length) return;
    state.undoStack.push(snapshotDoc());
    restoreSnapshot(state.redoStack.pop());
  });
  $("#btn-engine")?.addEventListener("click", () => toggleEngineDrawer());
  $("#btn-engine-close")?.addEventListener("click", () => toggleEngineDrawer(false));
  $("#btn-render")?.addEventListener("click", () => renderPortalEdit());

  // Accessible tablist switching (click + ArrowLeft/ArrowRight/Home/End)
  const tabBtns = $$("#engine-tabs [role='tab']");
  for (let i = 0; i < tabBtns.length; i++) {
    const btn = tabBtns[i];
    btn.addEventListener("click", () => activateEngineTab(btn));
    btn.addEventListener("keydown", (ev) => {
      let nextIdx = -1;
      if (ev.key === "ArrowRight") nextIdx = (i + 1) % tabBtns.length;
      else if (ev.key === "ArrowLeft") nextIdx = (i - 1 + tabBtns.length) % tabBtns.length;
      else if (ev.key === "Home") nextIdx = 0;
      else if (ev.key === "End") nextIdx = tabBtns.length - 1;
      if (nextIdx >= 0) {
        ev.preventDefault();
        tabBtns[nextIdx].focus();
        activateEngineTab(tabBtns[nextIdx]);
      }
    });
  }

  // Stage top controls (spatial mode switch is orthogonal to Actor Layer)
  $("#mode-btn-portal")?.addEventListener("click", () => {
    pushUndo();
    state.doc.mode = "portal_path";
    if (!state.streamsManualOverride) {
      state.editGuidance.streams = "independent";
    }
    syncAllUI();
    schedulePreflight();
  });
  $("#mode-btn-targets")?.addEventListener("click", () => {
    pushUndo();
    state.doc.mode = "point_targets";
    if (!state.streamsManualOverride) {
      state.editGuidance.streams = "video_only";
    }
    if (!state.doc.targets?.length) {
      state.doc.targets = [{
        id: 1,
        label: "TARGET 1",
        start: snapFrameTo3f(Math.round(state.length * 0.2), state.length),
        end: Math.min(state.length - 1, Math.round(state.length * 0.7)),
        ramp: 4,
        x: 0.45,
        y: 0.52,
        radius: 0.18,
        cue: state.defaultSoundPreset,
        gain: 1.15,
        weight: 1.0,
        halo_radius: 0.08,
        halo_weight: 0.30,
      }];
    }
    syncAllUI();
    schedulePreflight();
  });
  $("#mode-btn-combined")?.addEventListener("click", () => {
    pushUndo();
    state.doc.mode = "combined";
    if (!state.streamsManualOverride) {
      state.editGuidance.streams = "independent";
    }
    if (!state.doc.targets?.length) {
      state.doc.targets = [{
        id: 1,
        label: "TARGET 1",
        start: snapFrameTo3f(Math.round(state.length * 0.2), state.length),
        end: Math.min(state.length - 1, Math.round(state.length * 0.7)),
        ramp: 4,
        x: 0.45,
        y: 0.52,
        radius: 0.18,
        cue: state.defaultSoundPreset,
        gain: 1.15,
        weight: 1.0,
        halo_radius: 0.08,
        halo_weight: 0.30,
      }];
    }
    syncAllUI();
    schedulePreflight();
  });
  $("#mode-btn-sprite")?.addEventListener("click", () => {
    state.spriteMode = !state.spriteMode;
    syncAllUI();
  });
  $("#btn-sam-pick")?.addEventListener("click", () => {
    if (!state.hasAnchor || state.samSegmenting) return;
    state.samPicking = !state.samPicking;
    syncAllUI();
    if (state.samPicking) {
      showToast(`Click a subject on stage (${state.samRole})`, "info", 2400);
    }
  });
  for (const roleBtn of $$(".sam-role-btn")) {
    roleBtn.addEventListener("click", () => {
      const nextRole = roleBtn.dataset.role || "include";
      state.samRole = nextRole;
      syncAllUI();
    });
  }
  $("#btn-sam-clear")?.addEventListener("click", () => clearSamMasks());
  $("#btn-sam-run")?.addEventListener("click", () => executeSamSegmentation());
  $("#sam-prompt-input")?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      executeSamSegmentation();
    }
  });
  $("#btn-shot-bridge")?.addEventListener("click", async () => {
    if (state.lastFrameRef) {
      state.lastFrameRef = "";
      state.lastFrameName = "";
      syncAllUI();
      schedulePreflight();
      showToast("Cleared pinned Last Frame", "info", 1500);
      return;
    }
    const prev = state.shotHistory.pop();
    if (!prev) return;
    state.lastFrameRef = prev.lastFrameRef || "";
    state.lastFrameName = prev.lastFrameName || "";
    if (prev.workflowId) {
      state.workflowId = prev.workflowId;
      const wfSel = $("#engine-workflow");
      if (wfSel) wfSel.value = prev.workflowId;
    }
    if (prev.anchorUrl) {
      try {
        await loadAnchorImageUrl(prev.anchorUrl, { ref: prev.anchorRef, name: prev.anchorName });
      } catch {}
    } else {
      syncAllUI();
    }
    showToast(`Restored Shot ${state.shotHistory.length + 1}`, "ok", 1600);
  });
  window.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && state.samPicking) {
      state.samPicking = false;
      syncAllUI();
    }
  });
  $("#btn-fullframe-tae")?.addEventListener("click", () => {
    if (!state.livePreview) state.livePreview = { fullFrame: true };
    else state.livePreview.fullFrame = !state.livePreview.fullFrame;
    $("#btn-fullframe-tae")?.classList.toggle("active", Boolean(state.livePreview.fullFrame));
    $("#btn-fullframe-tae").textContent = state.livePreview.fullFrame ? "Full-Frame TAE" : "Masked TAE";
    stage?.render();
  });

  // Beat 1 Anchor Upload + Drag & Drop + Still Gen
  const uploadInput = $("#upload-anchor-input");
  uploadInput?.addEventListener("change", (ev) => {
    const file = ev.target.files?.[0];
    if (file) handleAnchorUpload(file);
    uploadInput.value = "";
  });
  const dropzone = $("#anchor-dropzone");
  if (dropzone) {
    dropzone.addEventListener("dragover", (ev) => {
      ev.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (ev) => {
      ev.preventDefault();
      dropzone.classList.remove("dragover");
      const file = ev.dataTransfer?.files?.[0];
      if (file) handleAnchorUpload(file);
    });
  }
  $("#btn-gen-anchor")?.addEventListener("click", () => generateAnchorStill());
  $("#anchor-workflow-select")?.addEventListener("change", (ev) => {
    state.anchorWorkflowId = ev.target.value;
  });

  // Contextual Portal Card & World Sandbox controls
  $("#btn-dock-card")?.addEventListener("click", () => {
    const card = $("#portal-card");
    const currentlyLeft = state.manualDockSide ? state.manualDockSide === "left" : Boolean(card?.classList.contains("dock-left"));
    state.manualDockSide = currentlyLeft ? "right" : "left";
    syncContextualCard();
  });
  $("#btn-collapse-card")?.addEventListener("click", () => {
    $("#portal-card")?.classList.toggle("collapsed");
  });
  $("#target-prompt")?.addEventListener("input", (ev) => {
    state.targetPrompt = ev.target.value;
    if (String(ev.target.value || "").trim()) state.portalTouched = true;
    if ((state.doc.mode === "point_targets" || state.doc.mode === "combined") && state.doc.targets?.[state.selectedTargetIdx]) {
      state.doc.targets[state.selectedTargetIdx].prompt = ev.target.value;
    }
    syncPromptDiffPreview();
  });
  $("#original-prompt")?.addEventListener("input", (ev) => {
    state.originalPrompt = ev.target.value;
    syncPromptDiffPreview();
  });
  for (const qc of $$("#quick-action-chips .quick-chip")) {
    qc.addEventListener("click", () => {
      const tpl = qc.dataset.template || "";
      const act = tpl || qc.dataset.action || qc.textContent || "";
      state.targetPrompt = act;
      state.portalTouched = true;
      if ((state.doc.mode === "point_targets" || state.doc.mode === "combined") && state.doc.targets?.[state.selectedTargetIdx]) {
        state.doc.targets[state.selectedTargetIdx].prompt = act;
      }
      const tp = $("#target-prompt");
      if (tp) {
        tp.value = act;
        if (tpl) {
          tp.focus();
          const phStart = act.indexOf("[");
          const phEnd = act.indexOf("]", phStart);
          if (phStart >= 0 && phEnd > phStart && tp.setSelectionRange) {
            tp.setSelectionRange(phStart, phEnd + 1);
          }
        }
      }
      syncPromptDiffPreview();
    });
  }
  $("#btn-preview-mask")?.addEventListener("click", () => previewInMask());
  $("#btn-clear-mask-preview")?.addEventListener("click", () => {
    state.maskPreview = null;
    syncAllUI();
  });
  $("#btn-cancel-render")?.addEventListener("click", async () => {
    await apiJson("/api/interrupt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ client_id: wsClient.clientId }) }).catch(() => {});
    clearActiveRender();
    state.livePreview = null;
    showToast("Cancelled active render", "info", 2000);
    stage?.render();
  });
  $("#btn-set-path")?.addEventListener("click", () => {
    state.settingSpritePath = !state.settingSpritePath;
    if (state.settingSpritePath) state.samPicking = false;
    syncAllUI();
    if (state.settingSpritePath) {
      showToast("Click destination point on stage for Movable Actor path", "info", 2400);
    }
  });
  $("#chk-snap-mask")?.addEventListener("change", (ev) => {
    state.snapToIncludeMask = Boolean(ev.target.checked);
    syncAllUI();
    schedulePreflight();
  });
  $("#btn-refresh-suggest")?.addEventListener("click", () => refreshContextSuggestions());
  $("#card-cue-select")?.addEventListener("change", (ev) => {
    const preset = ev.target.value;
    pushUndo();
    if (state.doc.mode === "point_targets" && state.doc.targets?.[state.selectedTargetIdx]) {
      state.doc.targets[state.selectedTargetIdx].cue = preset;
    } else if (preset !== "none") {
      const f = snapFrameTo3f(state.frame, state.length);
      state.doc.portal_cues = (state.doc.portal_cues || []).filter((c) => c.frame !== f);
      state.doc.portal_cues.push({ frame: f, preset, gain: 1.0 });
    }
    if (preset !== "none") {
      const pfPan = evalPortalAtFrame(state.doc, state.frame, state.length);
      foley.playCue(preset, 1.0, state.soundPresetMap[preset] || 0.2, clamp(pfPan.x * 2 - 1, -1, 1));
    }
    syncAllUI();
  });
  $("#pill-cue-audition")?.addEventListener("click", () => {
    const sel = $("#card-cue-select")?.value;
    if (sel && sel !== "none") {
      const pfPan = evalPortalAtFrame(state.doc, state.frame, state.length);
      foley.playCue(sel, 1.0, state.soundPresetMap[sel] || 0.2, clamp(pfPan.x * 2 - 1, -1, 1));
    }
  });
  $("#btn-apply-event")?.addEventListener("click", () => {
    pushUndo();
    state.portalTouched = true;
    const curAction = (state.targetPrompt || "").trim();
    if ((state.doc.mode === "point_targets" || state.doc.mode === "combined") && state.doc.targets?.[state.selectedTargetIdx]) {
      state.doc.targets[state.selectedTargetIdx].prompt = curAction;
    }
    const pf = evalPortalAtFrame(state.doc, state.frame, state.length);
    state.doc.portal = (state.doc.portal || []).filter((p) => p.frame !== state.frame);
    state.doc.portal.push({
      frame: state.frame,
      x: Number(pf.x.toFixed(3)),
      y: Number(pf.y.toFixed(3)),
      rx: Number(pf.rx.toFixed(3)),
      ry: Number(pf.ry.toFixed(3)),
      shape: pf.shape,
      label: pf.label,
      prompt: curAction || pf.prompt || "",
      weight: Number((pf.weight ?? 1.0).toFixed(2)),
      halo_radius: Number((pf.halo_radius ?? 0.10).toFixed(3)),
      halo_weight: Number((pf.halo_weight ?? 0.30).toFixed(2)),
    });
    state.doc.portal.sort((a, b) => a.frame - b.frame);
    syncAllUI();
    schedulePreflight();
    showToast(`Applied choreography at f${state.frame} (0ms local)`, "ok", 1800);
  });
  $("#dial-mask-halo")?.addEventListener("pointerdown", (ev) => {
    const el = $("#dial-mask-halo");
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const u = clamp((ev.clientX - rect.left) / Math.max(1, rect.width), 0.05, 1.0);
    pushUndo();
    stage?.setHaloRadius(u * 0.32);
    syncAllUI();
    schedulePreflight();
  });
  $("#dial-coupling")?.addEventListener("pointerdown", (ev) => {
    const el = $("#dial-coupling");
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const u = clamp((ev.clientX - rect.left) / Math.max(1, rect.width), 0.0, 1.0);
    pushUndo();
    state.editGuidance.scale = Number(clamp(0.5 + u * 2.0, 0.5, 2.5).toFixed(1));
    if (state.couplings.length) {
      state.couplings[0].strength = Number(clamp(u, 0.1, 1.0).toFixed(2));
    }
    syncAllUI();
    schedulePreflight();
  });
  $("#btn-gen-stamp")?.addEventListener("click", () => generateOrExtractStamp());
  $("#footer-add-stamp-btn")?.addEventListener("click", () => generateOrExtractStamp());
  $("#btn-drawer-stamp")?.addEventListener("click", () => generateOrExtractStamp());
  $("#btn-hop-arc")?.addEventListener("click", () => stage?.triggerSpriteHop());

  // Scrubber & Transport
  $("#btn-play")?.addEventListener("click", () => togglePlayback());
  $("#btn-loop")?.addEventListener("click", () => {
    state.loop = !state.loop;
    $("#btn-loop")?.classList.toggle("active", state.loop);
    $("#btn-loop")?.setAttribute("aria-pressed", state.loop ? "true" : "false");
    syncMediaPlayback();
  });
  $("#btn-grid-toggle")?.addEventListener("click", () => {
    state.showPatchGrid = !state.showPatchGrid;
    $("#btn-grid-toggle")?.classList.toggle("active", state.showPatchGrid);
    syncScrubberSVG();
    stage?.render();
  });
  $("#btn-audio-guard")?.addEventListener("click", () => {
    const order = ["independent", "video_only", "both"];
    const idx = order.indexOf(String(state.editGuidance.streams || "independent"));
    state.editGuidance.streams = order[(idx + 1) % order.length];
    state.streamsManualOverride = true;
    syncAllUI();
    schedulePreflight();
  });
  $("#btn-stage-mute")?.addEventListener("click", () => {
    state.stageMuted = !state.stageMuted;
    syncMediaPlayback();
    syncAudioGuardAndSequencer();
  });
  $("#btn-chain-frame")?.addEventListener("click", async () => {
    if (state.activeStageTake) {
      await chainTakeFrameToAnchor(state.activeStageTake, Math.max(0, Math.round(state.frame)), $("#btn-chain-frame"));
    } else if (state.maskPreview?.img) {
      await chainMaskPreviewToAnchor();
    }
  });

  const toggleLanes = (forceOpen = null) => {
    state.lanesOpen = forceOpen === null ? !state.lanesOpen : Boolean(forceOpen);
    syncAudioGuardAndSequencer();
    if (state.lanesOpen) sequencer?.resize();
  };
  const deleteSelectedSeqItem = () => {
    if (!sequencer || !state.seqSelection) return;
    const snap = snapshotDoc();
    const res = sequencer.deleteSelected(state);
    if (!res.ok) {
      showToast(res.reason || "Cannot remove sole anchor", "err", 2200);
      return;
    }
    state.undoStack.push(snap);
    if (state.undoStack.length > 40) state.undoStack.shift();
    state.redoStack = [];
    syncAllUI();
    schedulePreflight();
  };

  $("#btn-lanes-toggle")?.addEventListener("click", () => toggleLanes());
  $("#btn-seq-close")?.addEventListener("click", () => toggleLanes(false));
  $("#btn-seq-add-kf")?.addEventListener("click", () => {
    pushUndo();
    const pf = evalPortalAtFrame(state.doc, state.frame, state.length);
    if (!Array.isArray(state.doc.portal)) state.doc.portal = [];
    let kf = state.doc.portal.find((p) => Number(p.frame) === state.frame);
    if (!kf) {
      kf = {
        frame: state.frame,
        x: Number(pf.x.toFixed(3)),
        y: Number(pf.y.toFixed(3)),
        rx: Number(pf.rx.toFixed(3)),
        ry: Number(pf.ry.toFixed(3)),
        shape: pf.shape || "circle",
        label: pf.label || "REALITY LENS",
        weight: Number((pf.weight ?? 1.0).toFixed(2)),
        halo_radius: Number((pf.halo_radius ?? 0.10).toFixed(3)),
        halo_weight: Number((pf.halo_weight ?? 0.30).toFixed(2)),
        unlink_ry: Boolean(state.unlinkPortalRy),
      };
      state.doc.portal.push(kf);
      state.doc.portal.sort((a, b) => a.frame - b.frame);
    }
    state.seqSelection = { kind: "portal_kf", idx: state.doc.portal.indexOf(kf) };
    syncAllUI();
    schedulePreflight();
  });
  $("#btn-seq-add-target")?.addEventListener("click", () => {
    pushUndo();
    if (!Array.isArray(state.doc.targets)) state.doc.targets = [];
    const maxF = Math.max(3, state.length - 1);
    let start = snapFrameTo3f(state.frame, state.length);
    if (start > maxF - 3) start = Math.max(0, maxF - 3);
    const end = Math.min(maxF, Math.max(start + 3, start + 24));
    const nextId = state.doc.targets.reduce((m, t) => Math.max(m, Number(t.id || 0)), 0) + 1;
    state.doc.targets.push({
      id: nextId,
      label: `TARGET ${nextId}`,
      start,
      end,
      ramp: 4,
      x: Number(clamp(0.25 + ((nextId * 0.19) % 0.5), 0.15, 0.85).toFixed(3)),
      y: 0.50,
      radius: 0.16,
      cue: state.defaultSoundPreset || "none",
      gain: 1.15,
      weight: 1.0,
      halo_radius: 0.08,
      halo_weight: 0.30,
    });
    if (state.doc.mode === "portal_path") {
      state.doc.mode = "combined";
    }
    state.selectedTargetIdx = state.doc.targets.length - 1;
    state.seqSelection = { kind: "target", idx: state.selectedTargetIdx };
    syncAllUI();
    schedulePreflight();
  });
  $("#btn-seq-add-hop")?.addEventListener("click", () => {
    pushUndo();
    if (!state.doc.dpad) state.doc.dpad = { waypoints: [], hops: [] };
    if (!Array.isArray(state.doc.dpad.hops)) state.doc.dpad.hops = [];
    const hs = clamp(state.frame, 0, Math.max(0, state.length - 7));
    const he = clamp(hs + 15, hs + 4, state.length - 1);
    state.doc.dpad.hops.push({ start: hs, end: he, lift: 0.25 });
    state.seqSelection = { kind: "hop_arc", idx: state.doc.dpad.hops.length - 1 };
    syncAllUI();
    schedulePreflight();
  });
  $("#btn-seq-ry-unlink")?.addEventListener("click", () => {
    pushUndo();
    state.unlinkPortalRy = !state.unlinkPortalRy;
    const kf = stage?._ensurePortalKeyframeAtFrame(state);
    if (kf && state.unlinkPortalRy) {
      if (kf.shape !== "ellipse") {
        const w = Number(state.doc.width || 1344);
        const h = Number(state.doc.height || 768);
        const s = Math.min(w, h);
        kf.rx = Number(clamp((kf.rx * s) / w, 0.04, 0.45).toFixed(3));
        kf.ry = Number(clamp((kf.ry * s) / h, 0.04, 0.48).toFixed(3));
        kf.shape = "ellipse";
      }
      kf.unlink_ry = true;
    } else if (kf) {
      kf.unlink_ry = false;
    }
    syncAllUI();
    schedulePreflight();
  });
  $("#btn-seq-delete")?.addEventListener("click", () => deleteSelectedSeqItem());

  const scrubWrap = $("#scrubber-track-wrap");
  if (scrubWrap) {
    let scrubbing = false;
    scrubWrap.addEventListener("pointerdown", (ev) => {
      scrubbing = true;
      scrubWrap.setPointerCapture?.(ev.pointerId);
      scrubToClientX(ev.clientX);
    });
    window.addEventListener("pointermove", (ev) => {
      if (scrubbing) scrubToClientX(ev.clientX);
    });
    window.addEventListener("pointerup", () => {
      scrubbing = false;
    });
  }

  // Engine Drawer fields
  $("#engine-workflow")?.addEventListener("change", (ev) => {
    state.workflowId = ev.target.value;
    state.workflowManualOverride = true;
    syncPromptDiffPreview();
    syncContextualCard();
    schedulePreflight();
  });
  $("#preview-workflow-select")?.addEventListener("change", (ev) => {
    state.previewWorkflowId = ev.target.value;
  });
  $("#engine-canvas")?.addEventListener("change", (ev) => {
    const [w, h] = String(ev.target.value).split("x").map(Number);
    if (w > 0 && h > 0) {
      state.doc.width = w;
      state.doc.height = h;
      syncAllUI();
      schedulePreflight();
    }
  });
  $("#engine-length")?.addEventListener("change", (ev) => switchLength(Number(ev.target.value)));
  $("#btn-promote-hero")?.addEventListener("click", () => promoteSeedToHero());
  $("#seed-input")?.addEventListener("change", (ev) => {
    state.seed = Number(ev.target.value || 0);
  });
  $("#btn-seed-lock")?.addEventListener("click", () => {
    state.seedLocked = !state.seedLocked;
    $("#btn-seed-lock")?.classList.toggle("active", state.seedLocked);
    $("#btn-seed-lock")?.setAttribute("aria-pressed", state.seedLocked ? "true" : "false");
    $("#btn-seed-lock").textContent = state.seedLocked ? "Locked" : "Unlocked";
  });
  $("#btn-seed-rand")?.addEventListener("click", () => {
    state.seed = Math.floor(Math.random() * 2147483640) + 1;
    syncAllUI();
  });
  $("#engine-shape")?.addEventListener("change", (ev) => {
    pushUndo();
    for (const p of state.doc.portal || []) p.shape = ev.target.value;
    syncAllUI();
  });
  $("#engine-streams")?.addEventListener("change", (ev) => {
    state.editGuidance.streams = ev.target.value;
    state.streamsManualOverride = true;
    syncAudioGuardAndSequencer();
    schedulePreflight();
  });
  $("#engine-scale")?.addEventListener("change", (ev) => {
    state.editGuidance.scale = clamp(Number(ev.target.value || 1.5), 0.5, 2.5);
    syncAllUI();
  });
  $("#engine-audio-scale")?.addEventListener("change", (ev) => {
    state.editGuidance.audio_scale = clamp(Number(ev.target.value || 1.0), 0.0, 2.5);
    syncAllUI();
  });
  $("#engine-first-step")?.addEventListener("change", (ev) => {
    state.editGuidance.first_step = clamp(Math.round(Number(ev.target.value || 0)), 0, 5);
  });
  $("#engine-feather")?.addEventListener("change", (ev) => {
    state.doc.feather = clamp(Number(ev.target.value || 0.08), 0.0, 0.5);
    stage?.render();
  });
  $("#chk-invert-mask")?.addEventListener("change", (ev) => {
    state.doc.invert_mask = Boolean(ev.target.checked);
    stage?.render();
  });
  $("#chk-same-prompt")?.addEventListener("change", (ev) => {
    state.samePrompt = Boolean(ev.target.checked);
  });
  $("#chk-audio-refine")?.addEventListener("change", (ev) => {
    state.audioRefine = Boolean(ev.target.checked);
    syncAllUI();
  });
  $("#engine-cv-select")?.addEventListener("change", (ev) => {
    state.controlVector = ev.target.value;
    schedulePreflight();
  });
  $("#engine-cv-strength")?.addEventListener("change", (ev) => {
    state.controlVectorStrength = Number(ev.target.value || 1.0);
  });
  $("#btn-drop-cue")?.addEventListener("click", () => {
    const preset = $("#engine-cue-preset")?.value;
    if (!preset) return;
    pushUndo();
    const f = snapFrameTo3f(state.frame, state.length);
    state.doc.portal_cues = (state.doc.portal_cues || []).filter((c) => c.frame !== f);
    state.doc.portal_cues.push({ frame: f, preset, gain: 1.0 });
    foley.playCue(preset, 1.0, state.soundPresetMap[preset] || 0.2);
    syncAllUI();
  });

  $("#btn-refresh-takes")?.addEventListener("click", () => refreshTakes());
  $("#btn-compile-inspect")?.addEventListener("click", async () => {
    try {
      const res = await apiJson("/api/compile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildCompilePayload()),
      });
      const nodeCount = Object.keys(res.graph || {}).length;
      showToast(`Compiled ${res.mode} (${nodeCount} nodes, preview=${Boolean(res.preview_override_injected)})`, "ok");
    } catch (err) {
      showToast(`Compile check failed: ${err.message}`, "err");
    }
  });
  $("#btn-interrupt")?.addEventListener("click", async () => {
    await apiJson("/api/interrupt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ client_id: wsClient.clientId }) }).catch(() => {});
    showToast("Interrupted active job", "info");
  });
  $("#btn-forget")?.addEventListener("click", async () => {
    try {
      const keepUploads = [
        state.anchorRef,
        state.spriteCutout?.ref,
        state.spriteMaskRef,
        state.includeMaskRef,
        state.excludeMaskRef,
      ].filter(Boolean);
      const res = await apiJson("/api/forget", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ client_id: wsClient.clientId, keep_uploads: keepUploads }),
      });
      state.takes = [];
      renderTakesList();
      showToast(`Cleared temp files (${res.deleted_temp ?? 0} outputs)`, "ok");
    } catch (err) {
      showToast(`Forget failed: ${err.message}`, "err");
    }
  });

  // A|B Compare & Export
  $("#btn-compare-ab")?.addEventListener("click", () => openABCompareForTake());
  $("#btn-ab-close")?.addEventListener("click", () => openABCompareForTake());
  $("#btn-ab-audio-toggle")?.addEventListener("click", () => {
    abCompare.cycleAudioSource();
    syncMediaPlayback();
    syncAudioGuardAndSequencer();
  });
  $("#btn-ab-hud-toggle")?.addEventListener("click", () => abCompare.setHudMode(!abCompare.hudMode));
  $("#btn-export-clean")?.addEventListener("click", () => abCompare.exportTake("clean"));
  $("#btn-export-hud")?.addEventListener("click", () => abCompare.exportTake("hud"));

  // Global keyboard shortcuts (ignored inside text inputs)
  window.addEventListener("keydown", (ev) => {
    const tag = (ev.target?.tagName || "").toUpperCase();
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

    if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "z") {
      ev.preventDefault();
      if (ev.shiftKey) $("#btn-redo")?.click();
      else $("#btn-undo")?.click();
      return;
    }
    if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "y") {
      ev.preventDefault();
      $("#btn-redo")?.click();
      return;
    }

    if ((ev.key === "Delete" || ev.key === "Backspace") && state.lanesOpen && state.seqSelection) {
      ev.preventDefault();
      $("#btn-seq-delete")?.click();
      return;
    }

    if (!ev.metaKey && !ev.ctrlKey && ev.key.toLowerCase() === "l") {
      ev.preventDefault();
      $("#btn-lanes-toggle")?.click();
      return;
    }

    if (!ev.metaKey && !ev.ctrlKey && ev.key.toLowerCase() === "m") {
      ev.preventDefault();
      $("#btn-stage-mute")?.click();
      return;
    }

    if (ev.key === " ") {
      ev.preventDefault();
      if (state.spriteMode && state.playing) {
        stage?.triggerSpriteHop();
      } else {
        togglePlayback();
      }
      return;
    }

    if (state.spriteMode) {
      const k = ev.key.toLowerCase();
      if (k === "w" || ev.key === "ArrowUp") { ev.preventDefault(); stage?.nudgeSprite(0, -0.035); return; }
      if (k === "s" || ev.key === "ArrowDown") { ev.preventDefault(); stage?.nudgeSprite(0, 0.035); return; }
      if (k === "a" || ev.key === "ArrowLeft") { ev.preventDefault(); stage?.nudgeSprite(-0.04, 0, ev.shiftKey ? null : "left"); return; }
      if (k === "d" || ev.key === "ArrowRight") { ev.preventDefault(); stage?.nudgeSprite(0.04, 0, ev.shiftKey ? null : "right"); return; }
    } else {
      if (ev.key === "ArrowLeft" || ev.key === "[") {
        ev.preventDefault();
        state.frame = clamp(state.frame - (ev.shiftKey ? 3 : 1), 0, state.length - 1);
        syncAllUI();
        return;
      }
      if (ev.key === "ArrowRight" || ev.key === "]") {
        ev.preventDefault();
        state.frame = clamp(state.frame + (ev.shiftKey ? 3 : 1), 0, state.length - 1);
        syncAllUI();
        return;
      }
      if (ev.key === "ArrowUp" || ev.key === "ArrowDown") {
        ev.preventDefault();
        const dir = ev.key === "ArrowUp" ? 1 : -1;
        state.editGuidance.scale = Number(clamp(state.editGuidance.scale + dir * 0.1, 0.5, 2.5).toFixed(1));
        syncAllUI();
        return;
      }
    }

    // Keys 1..9 drop & audition hydrated 3f sound presets
    if (/^[1-9]$/.test(ev.key) && state.soundPresetsList.length) {
      const idx = Number(ev.key) - 1;
      const preset = state.soundPresetsList[idx];
      if (preset) {
        ev.preventDefault();
        pushUndo();
        const f = snapFrameTo3f(state.frame, state.length);
        state.doc.portal_cues = (state.doc.portal_cues || []).filter((c) => c.frame !== f);
        state.doc.portal_cues.push({ frame: f, preset, gain: 1.0 });
        foley.playCue(preset, 1.0, state.soundPresetMap[preset] || 0.2);
        syncAllUI();
        showToast(`Dropped 3f cue f${f}: ${preset}`, "ok", 1600);
      }
    }
  });

  window.addEventListener("resize", () => {
    stage?.resize();
    syncScrubberSVG();
    sequencer?.resize();
  });

  // Initial hydrate
  hydrateSchema()
    .then(() => Promise.all([pollSystemStats(), refreshTakes()]))
    .catch((err) => {
      showToast(`Schema load failed: ${err.message}`, "err", 6000);
    });

  // Expose debug handle for inspection without touching DOM styles
  window.__SPATIAL_DIRECTOR__ = { state, stage, sequencer, abCompare, takesUI, wsClient, foley, syncAllUI };
}

document.addEventListener("DOMContentLoaded", init);
