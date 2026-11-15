/**
 * MiniMax H3 Spatial World Director — Interactive ComfyUI Stage Extension
 *
 * Replaces blind JSON coordinate editing with an Obsidian-Glass 2D+Time Stage:
 * - Reality Portal (`portal_path`): drag portal keyframes `(x, y)`, resize rim
 *   radius `(rx, ry)`, switch shapes (`circle`, `ellipse`, `vertical_split`).
 * - Genie + Point-and-Click World Targets (`point_targets`): click/drag world
 *   target zones `(x, y, radius)` + audio cue badges, drag D-Pad sprite
 *   waypoints, or steer with the embedded Genie D-Pad buttons.
 * - Live 24 fps transport scrubber synced bidirectionally with ComfyUI widgets.
 */

import { app } from "../../scripts/app.js";

const NODE_NAME = "MiniMaxH3SpatialDirector";
const VIDEO_FPS = 24.0;
const SOUND_PRESETS = [
  "slam",
  "hop",
  "thud",
  "crash",
  "pop",
  "snap",
  "whoosh",
  "bell",
  "gong",
  "click",
  "tone",
  "none",
];

const PRESET_PORTAL_CIRCLE = {
  portal: [
    { frame: 0, x: 0.18, y: 0.52, rx: 0.24, ry: 0.30, shape: "circle", label: "REALITY LENS" },
    { frame: 70, x: 0.50, y: 0.50, rx: 0.26, ry: 0.32, shape: "circle", label: "REALITY LENS" },
    { frame: 140, x: 0.82, y: 0.52, rx: 0.24, ry: 0.30, shape: "circle", label: "REALITY LENS" },
  ],
  targets: [
    { id: 1, label: "TARGET A", start: 24, end: 66, ramp: 5, x: 0.24, y: 0.54, radius: 0.19, cue: "slam", gain: 1.20 },
    { id: 2, label: "TARGET B", start: 78, end: 126, ramp: 5, x: 0.76, y: 0.54, radius: 0.19, cue: "hop", gain: 1.20 },
  ],
  dpad: {
    waypoints: [
      { frame: 0, x: 0.24, y: 0.76, face: "right", scale: 1.0 },
      { frame: 54, x: 0.50, y: 0.76, face: "right", scale: 1.0 },
      { frame: 96, x: 0.68, y: 0.76, face: "right", scale: 1.0 },
      { frame: 140, x: 0.54, y: 0.76, face: "right", scale: 1.0 },
    ],
    hops: [{ start: 63, end: 78, lift: 0.25 }],
  },
};

const PRESET_PORTAL_SPLIT = {
  portal: [
    { frame: 0, x: 0.15, y: 0.50, rx: 0.22, ry: 0.30, shape: "vertical_split", label: "SPLIT WIPE" },
    { frame: 70, x: 0.52, y: 0.50, rx: 0.22, ry: 0.30, shape: "vertical_split", label: "SPLIT WIPE" },
    { frame: 140, x: 0.85, y: 0.50, rx: 0.22, ry: 0.30, shape: "vertical_split", label: "SPLIT WIPE" },
  ],
  targets: PRESET_PORTAL_CIRCLE.targets,
  dpad: PRESET_PORTAL_CIRCLE.dpad,
};

const PRESET_GENIE_TARGETS = {
  portal: PRESET_PORTAL_CIRCLE.portal,
  targets: [
    { id: 1, label: "TARGET A · CRYSTAL", start: 24, end: 66, ramp: 5, x: 0.72, y: 0.48, radius: 0.19, cue: "bell", gain: 1.25 },
    { id: 2, label: "TARGET B · GEYSER", start: 78, end: 126, ramp: 5, x: 0.30, y: 0.56, radius: 0.19, cue: "slam", gain: 1.20 },
  ],
  dpad: {
    waypoints: [
      { frame: 0, x: 0.28, y: 0.74, face: "right", scale: 1.0 },
      { frame: 54, x: 0.68, y: 0.72, face: "left", scale: 1.0 },
      { frame: 96, x: 0.42, y: 0.72, face: "left", scale: 1.0 },
      { frame: 140, x: 0.28, y: 0.76, face: "right", scale: 1.0 },
    ],
    hops: [{ start: 63, end: 78, lift: 0.26 }],
  },
};

function rescalePresetForLength(preset, length) {
  const cloned = deepClone(preset);
  const maxF = Math.max(4, length - 1);
  if (maxF === 140) return cloned;
  const s = (f) => clamp(Math.round((Number(f || 0) / 140) * maxF), 0, maxF);
  cloned.portal.forEach((p) => { p.frame = s(p.frame); });
  cloned.targets.forEach((t) => {
    t.start = s(t.start);
    t.end = Math.max(Math.min(maxF, t.start + 3), s(t.end));
  });
  cloned.dpad.waypoints.forEach((w) => { w.frame = s(w.frame); });
  cloned.dpad.hops.forEach((h) => {
    h.start = s(h.start);
    h.end = Math.max(Math.min(maxF, h.start + 3), s(h.end));
  });
  return cloned;
}

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function cosineEase(u) {
  const c = clamp(u, 0.0, 1.0);
  return 0.5 - 0.5 * Math.cos(Math.PI * c);
}

function snapTo3f(frame, length) {
  const maxMult = Math.floor(Math.max(0, length - 1) / 3) * 3;
  return clamp(Math.round(frame / 3) * 3, 0, maxMult);
}

function resolveFrame(val, length) {
  if (typeof val === "number" && Number.isFinite(val)) {
    const iv = Math.round(val);
    return clamp(iv >= 0 ? iv : length + iv, 0, Math.max(0, length - 1));
  }
  if (typeof val === "string") {
    const tok = val.trim();
    if (tok.endsWith("s")) {
      const sec = parseFloat(tok.slice(0, -1));
      if (Number.isFinite(sec)) {
        return clamp(Math.round(sec * VIDEO_FPS), 0, Math.max(0, length - 1));
      }
    } else {
      const iv = parseInt(tok, 10);
      if (Number.isFinite(iv)) {
        return clamp(iv >= 0 ? iv : length + iv, 0, Math.max(0, length - 1));
      }
    }
  }
  return 0;
}

function ensureDocShape(rawDoc) {
  const doc = rawDoc && typeof rawDoc === "object" && !Array.isArray(rawDoc) ? rawDoc : {};
  if (!Array.isArray(doc.portal) || doc.portal.length === 0) {
    doc.portal = deepClone(PRESET_PORTAL_CIRCLE.portal);
  }
  if (!Array.isArray(doc.targets) || doc.targets.length === 0) {
    doc.targets = deepClone(PRESET_PORTAL_CIRCLE.targets);
  }
  if (!doc.dpad || typeof doc.dpad !== "object") {
    doc.dpad = deepClone(PRESET_PORTAL_CIRCLE.dpad);
  }
  if (!Array.isArray(doc.dpad.waypoints)) {
    doc.dpad.waypoints = [];
  }
  if (!Array.isArray(doc.dpad.hops)) {
    doc.dpad.hops = [];
  }
  return doc;
}

function evalPortalAtFrame(doc, frame, length) {
  const keyframes = [...doc.portal]
    .map((p, idx) => ({
      idx,
      frame: resolveFrame(p.frame ?? 0, length),
      x: Number(p.x ?? 0.5),
      y: Number(p.y ?? 0.5),
      rx: Number(p.rx ?? p.radius ?? 0.24),
      ry: Number(p.ry ?? p.rx ?? p.radius ?? 0.24),
      shape: String(p.shape || "circle"),
      label: String(p.label || "REALITY LENS"),
    }))
    .sort((a, b) => a.frame - b.frame);

  if (keyframes.length === 0) {
    return { x: 0.5, y: 0.5, rx: 0.24, ry: 0.24, shape: "circle", label: "REALITY LENS", keyframes };
  }
  if (frame <= keyframes[0].frame) {
    return { ...keyframes[0], keyframes };
  }
  if (frame >= keyframes[keyframes.length - 1].frame) {
    return { ...keyframes[keyframes.length - 1], keyframes };
  }
  for (let i = 0; i < keyframes.length - 1; i++) {
    const p0 = keyframes[i];
    const p1 = keyframes[i + 1];
    if (p0.frame <= frame && frame <= p1.frame) {
      const u = cosineEase((frame - p0.frame) / Math.max(1, p1.frame - p0.frame));
      const src = u < 0.5 ? p0 : p1;
      return {
        x: p0.x + (p1.x - p0.x) * u,
        y: p0.y + (p1.y - p0.y) * u,
        rx: p0.rx + (p1.rx - p0.rx) * u,
        ry: p0.ry + (p1.ry - p0.ry) * u,
        shape: src.shape,
        label: src.label,
        keyframes,
      };
    }
  }
  return { ...keyframes[keyframes.length - 1], keyframes };
}

function targetTemporalWeight(t, frame, length) {
  const s = resolveFrame(t.start ?? 0, length);
  const e = resolveFrame(t.end ?? length - 1, length);
  if (frame < s || frame > e) return 0.0;
  const span = e - s;
  const ramp = Math.max(0, Number(t.ramp ?? 4));
  if (ramp < 1 || span < 2) return 1.0;
  const rEff = Math.min(ramp, Math.max(1, Math.floor(span / 2)));
  const relIn = frame - s;
  const relOut = e - frame;
  if (relIn < rEff) return cosineEase((relIn + 1) / (rEff + 1));
  if (relOut < rEff) return cosineEase((relOut + 1) / (rEff + 1));
  return 1.0;
}

function evalSpriteAtFrame(doc, frame, length) {
  const wps = [...(doc.dpad?.waypoints || [])]
    .map((w, idx) => ({
      idx,
      frame: resolveFrame(w.frame ?? 0, length),
      x: Number(w.x ?? 0.5),
      y: Number(w.y ?? 0.72),
      face: String(w.face || "right"),
    }))
    .sort((a, b) => a.frame - b.frame);
  if (wps.length === 0) return null;
  let pose = { x: wps[0].x, y: wps[0].y, face: wps[0].face, hopping: false, lift: 0 };
  if (frame >= wps[wps.length - 1].frame) {
    const last = wps[wps.length - 1];
    pose = { x: last.x, y: last.y, face: last.face, hopping: false, lift: 0 };
  } else {
    for (let i = 0; i < wps.length - 1; i++) {
      const a = wps[i];
      const b = wps[i + 1];
      if (a.frame <= frame && frame <= b.frame) {
        const u = cosineEase((frame - a.frame) / Math.max(1, b.frame - a.frame));
        pose = {
          x: a.x + (b.x - a.x) * u,
          y: a.y + (b.y - a.y) * u,
          face: u < 0.5 ? a.face : b.face,
          hopping: false,
          lift: 0,
        };
        break;
      }
    }
  }
  for (const hp of doc.dpad?.hops || []) {
    const hs = resolveFrame(hp.start ?? 0, length);
    const he = resolveFrame(hp.end ?? hs + 6, length);
    if (hs <= frame && frame <= he && he > hs) {
      const u = (frame - hs) / (he - hs);
      const lift = Number(hp.lift ?? 0.22) * Math.sin(Math.PI * clamp(u, 0, 1));
      pose.y -= lift;
      pose.hopping = true;
      pose.lift = lift;
    }
  }
  return pose;
}

function injectStyleOnce() {
  if (document.getElementById("h3-spatial-director-style")) return;
  const style = document.createElement("style");
  style.id = "h3-spatial-director-style";
  style.textContent = `
    .h3sd-root {
      box-sizing: border-box;
      width: 100%;
      padding: 8px;
      border-radius: 10px;
      background: linear-gradient(160deg, rgba(14, 18, 30, 0.96), rgba(9, 12, 20, 0.98));
      border: 1px solid rgba(92, 225, 230, 0.25);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
      color: #e2e8f0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      font-size: 11px;
      display: flex;
      flex-direction: column;
      gap: 7px;
      -webkit-user-select: none;
      user-select: none;
    }
    .h3sd-bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
    }
    .h3sd-seg {
      display: inline-flex;
      border-radius: 7px;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.25);
      overflow: hidden;
    }
    .h3sd-btn {
      appearance: none;
      border: 0;
      background: transparent;
      color: #94a3b8;
      padding: 4px 8px;
      font-size: 10.5px;
      font-weight: 600;
      cursor: pointer;
      transition: background 120ms ease, color 120ms ease;
    }
    .h3sd-btn:hover { color: #f8fafc; background: rgba(51, 65, 85, 0.45); }
    .h3sd-btn.active {
      background: rgba(34, 211, 238, 0.2);
      color: #7df9ff;
      box-shadow: inset 0 0 0 1px rgba(34, 211, 238, 0.45);
    }
    .h3sd-btn.gold.active, .h3sd-btn.gold:hover {
      color: #fde047;
      background: rgba(250, 204, 21, 0.16);
    }
    .h3sd-pill {
      appearance: none;
      border: 1px solid rgba(148, 163, 184, 0.28);
      background: rgba(15, 23, 42, 0.85);
      color: #cbd5e1;
      border-radius: 6px;
      padding: 3px 7px;
      font-size: 10px;
      font-weight: 600;
      cursor: pointer;
    }
    .h3sd-pill:hover {
      border-color: rgba(125, 249, 255, 0.55);
      color: #f8fafc;
    }
    .h3sd-select {
      background: rgba(15, 23, 42, 0.92);
      color: #e2e8f0;
      border: 1px solid rgba(148, 163, 184, 0.3);
      border-radius: 6px;
      padding: 3px 6px;
      font-size: 10.5px;
    }
    .h3sd-stage-wrap {
      position: relative;
      width: 100%;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid rgba(148, 163, 184, 0.25);
      background: #060912;
    }
    .h3sd-canvas {
      display: block;
      width: 100%;
      height: auto;
      touch-action: none;
      cursor: crosshair;
      outline: none;
    }
    .h3sd-canvas:focus-visible {
      box-shadow: inset 0 0 0 2px #7df9ff;
    }
    .h3sd-transport {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .h3sd-slider {
      flex: 1 1 auto;
      accent-color: #22d3ee;
      height: 18px;
      cursor: pointer;
      margin: 0;
    }
    .h3sd-mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 10px;
      color: #7df9ff;
      white-space: nowrap;
    }
    .h3sd-status {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 10px;
      color: #94a3b8;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }
    .h3sd-sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip-path: inset(50%);
      white-space: nowrap;
    }
  `;
  document.head.appendChild(style);
}

function buildSpatialDirectorUI(node) {
  injectStyleOnce();

  const findWidget = (name) => (node.widgets || []).find((w) => w.name === name);
  const wMode = findWidget("mode");
  const wWidth = findWidget("width");
  const wHeight = findWidget("height");
  const wLength = findWidget("length");
  const wFeather = findWidget("feather");
  const wInvert = findWidget("invert_mask");
  const wEvents = findWidget("events_json");

  const state = {
    doc: ensureDocShape(null),
    frame: 0,
    playing: false,
    rafId: 0,
    lastTickMs: 0,
    selected: { kind: "portal", index: 1 },
    dragging: null,
    suppressWidgetSync: false,
  };

  function getParams() {
    const mode = wMode?.value === "point_targets" ? "point_targets" : "portal_path";
    const width = Math.max(64, Number(wWidth?.value ?? 1344));
    const height = Math.max(64, Number(wHeight?.value ?? 768));
    const length = Math.max(5, Math.round(Number(wLength?.value ?? 141)));
    const feather = clamp(Number(wFeather?.value ?? 0.08), 0.0, 0.5);
    const invert = Boolean(wInvert?.value);
    return { mode, width, height, length, feather, invert };
  }

  function loadDocFromWidget() {
    if (!wEvents) return;
    try {
      const parsed = JSON.parse(String(wEvents.value || "{}"));
      state.doc = ensureDocShape(parsed);
    } catch {
      state.doc = ensureDocShape(null);
    }
  }

  function pushDocToWidget() {
    if (!wEvents) return;
    state.suppressWidgetSync = true;
    try {
      const serialized = JSON.stringify(state.doc, null, 2);
      wEvents.value = serialized;
      wEvents.callback?.(serialized);
      app.graph?.setDirtyCanvas?.(true, true);
    } finally {
      state.suppressWidgetSync = false;
    }
  }

  // Root container
  const root = document.createElement("div");
  root.className = "h3sd-root";

  const srLive = document.createElement("div");
  srLive.className = "h3sd-sr-only";
  srLive.setAttribute("aria-live", "polite");
  root.appendChild(srLive);

  function announce(msg) {
    srLive.textContent = msg;
  }

  // Top bar: Mode toggle + Presets
  const topBar = document.createElement("div");
  topBar.className = "h3sd-bar";

  const modeSeg = document.createElement("div");
  modeSeg.className = "h3sd-seg";
  const btnPortalMode = document.createElement("button");
  btnPortalMode.type = "button";
  btnPortalMode.className = "h3sd-btn";
  btnPortalMode.textContent = "Reality Portal";
  const btnTargetMode = document.createElement("button");
  btnTargetMode.type = "button";
  btnTargetMode.className = "h3sd-btn";
  btnTargetMode.textContent = "Genie + Targets";
  modeSeg.append(btnPortalMode, btnTargetMode);

  const presetWrap = document.createElement("div");
  presetWrap.className = "h3sd-bar";
  presetWrap.style.gap = "4px";

  const btnSplitPreset = document.createElement("button");
  btnSplitPreset.type = "button";
  btnSplitPreset.className = "h3sd-pill";
  btnSplitPreset.textContent = "Split Wipe";

  const btnGeniePreset = document.createElement("button");
  btnGeniePreset.type = "button";
  btnGeniePreset.className = "h3sd-pill";
  btnGeniePreset.textContent = "2-Target Preset";

  const btnResetPreset = document.createElement("button");
  btnResetPreset.type = "button";
  btnResetPreset.className = "h3sd-pill";
  btnResetPreset.textContent = "Reset";

  presetWrap.append(btnSplitPreset, btnGeniePreset, btnResetPreset);
  topBar.append(modeSeg, presetWrap);
  root.appendChild(topBar);

  // Secondary bar: Context controls (shape / cue / D-pad nudger)
  const subBar = document.createElement("div");
  subBar.className = "h3sd-bar";

  const shapeSelect = document.createElement("select");
  shapeSelect.className = "h3sd-select";
  shapeSelect.title = "Portal geometry shape";
  for (const sh of ["circle", "ellipse", "vertical_split"]) {
    const opt = document.createElement("option");
    opt.value = sh;
    opt.textContent = `shape: ${sh}`;
    shapeSelect.appendChild(opt);
  }

  const cueSelect = document.createElement("select");
  cueSelect.className = "h3sd-select";
  cueSelect.title = "Selected target SoundCue preset (snaps 3f)";
  for (const pr of SOUND_PRESETS) {
    const opt = document.createElement("option");
    opt.value = pr;
    opt.textContent = `cue: ${pr}`;
    cueSelect.appendChild(opt);
  }

  const dpadCluster = document.createElement("div");
  dpadCluster.className = "h3sd-seg";
  const makeDpadBtn = (lbl, dx, dy, face) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "h3sd-btn gold";
    b.textContent = lbl;
    b.title = "Nudge or add Genie D-Pad waypoint at current frame";
    b.addEventListener("click", (ev) => {
      ev.stopPropagation();
      nudgeOrInsertDpad(dx, dy, face);
    });
    return b;
  };
  const btnDpadLeft = makeDpadBtn("◀", -0.06, 0, "left");
  const btnDpadHop = makeDpadBtn("▲HOP", 0, -0.05, null);
  const btnDpadDown = makeDpadBtn("▼", 0, 0.05, null);
  const btnDpadRight = makeDpadBtn("▶", 0.06, 0, "right");
  dpadCluster.append(btnDpadLeft, btnDpadHop, btnDpadDown, btnDpadRight);

  const btnAddTarget = document.createElement("button");
  btnAddTarget.type = "button";
  btnAddTarget.className = "h3sd-pill";
  btnAddTarget.textContent = "+ Target";

  subBar.append(shapeSelect, cueSelect, btnAddTarget, dpadCluster);
  root.appendChild(subBar);

  // Interactive Stage Canvas
  const stageWrap = document.createElement("div");
  stageWrap.className = "h3sd-stage-wrap";

  const canvas = document.createElement("canvas");
  canvas.className = "h3sd-canvas";
  canvas.width = 480;
  canvas.height = 274;
  canvas.tabIndex = 0;
  canvas.setAttribute(
    "aria-label",
    "Spatial World Director interactive canvas. Drag handles or use arrow keys to move, [ and ] to switch selection."
  );
  stageWrap.appendChild(canvas);
  root.appendChild(stageWrap);

  // Transport Bar
  const transport = document.createElement("div");
  transport.className = "h3sd-transport";

  const btnPlay = document.createElement("button");
  btnPlay.type = "button";
  btnPlay.className = "h3sd-pill";
  btnPlay.textContent = "▶ Play";

  const slider = document.createElement("input");
  slider.type = "range";
  slider.className = "h3sd-slider";
  slider.min = "0";
  slider.max = "140";
  slider.step = "1";
  slider.value = "0";
  slider.setAttribute("aria-label", "Preview frame scrubber");

  const frameLabel = document.createElement("span");
  frameLabel.className = "h3sd-mono";
  frameLabel.textContent = "f000/140 · 0.00s";

  transport.append(btnPlay, slider, frameLabel);
  root.appendChild(transport);

  // Bottom Status Strip
  const statusStrip = document.createElement("div");
  statusStrip.className = "h3sd-status";
  const statusLeft = document.createElement("span");
  const statusRight = document.createElement("span");
  statusStrip.append(statusLeft, statusRight);
  root.appendChild(statusStrip);

  function nudgeOrInsertDpad(dx, dy, face) {
    const { length } = getParams();
    const wps = state.doc.dpad.waypoints;
    let wp = wps.find((w) => Math.abs(resolveFrame(w.frame ?? 0, length) - state.frame) <= 2);
    if (!wp) {
      const pose = evalSpriteAtFrame(state.doc, state.frame, length) || { x: 0.5, y: 0.72, face: "right" };
      wp = {
        frame: state.frame,
        x: Number(pose.x.toFixed(3)),
        y: Number(pose.y.toFixed(3)),
        face: face || pose.face || "right",
        scale: 1.0,
      };
      wps.push(wp);
      wps.sort((a, b) => resolveFrame(a.frame, length) - resolveFrame(b.frame, length));
    }
    wp.x = Number(clamp(Number(wp.x ?? 0.5) + dx, 0.05, 0.95).toFixed(3));
    wp.y = Number(clamp(Number(wp.y ?? 0.72) + dy, 0.15, 0.92).toFixed(3));
    if (face) wp.face = face;
    if (dy < 0) {
      const hopS = Math.max(0, state.frame);
      const hopE = Math.min(length - 1, hopS + 12);
      if (hopE > hopS) {
        state.doc.dpad.hops = [{ start: hopS, end: hopE, lift: 0.24 }];
      }
    }
    state.selected = { kind: "dpad", index: wps.indexOf(wp) };
    pushDocToWidget();
    drawStage();
    announce(`Genie D-Pad waypoint updated at frame ${state.frame}`);
  }

  function syncToolbarState() {
    const { mode, length } = getParams();
    btnPortalMode.classList.toggle("active", mode === "portal_path");
    btnTargetMode.classList.toggle("active", mode === "point_targets");
    shapeSelect.style.display = mode === "portal_path" ? "" : "none";
    cueSelect.style.display = mode === "point_targets" ? "" : "none";
    btnAddTarget.style.display = mode === "point_targets" ? "" : "none";
    dpadCluster.style.display = mode === "point_targets" ? "" : "none";

    slider.max = String(Math.max(1, length - 1));
    state.frame = clamp(state.frame, 0, Math.max(0, length - 1));
    slider.value = String(state.frame);
    const sec = (state.frame / VIDEO_FPS).toFixed(2);
    frameLabel.textContent = `f${String(state.frame).padStart(3, "0")}/${String(length - 1).padStart(3, "0")} · ${sec}s`;
    slider.setAttribute("aria-valuetext", `Frame ${state.frame} of ${length - 1} (${sec} seconds)`);

    if (mode === "portal_path") {
      const pIdx = clamp(state.selected.index, 0, Math.max(0, state.doc.portal.length - 1));
      const kf = state.doc.portal[pIdx];
      if (kf) shapeSelect.value = kf.shape || "circle";
    } else {
      const tIdx =
        state.selected.kind === "target"
          ? clamp(state.selected.index, 0, Math.max(0, state.doc.targets.length - 1))
          : 0;
      const tg = state.doc.targets[tIdx];
      if (tg) cueSelect.value = tg.cue || "slam";
    }
  }

  function drawStage() {
    syncToolbarState();
    const { mode, width, height, length, feather, invert } = getParams();
    const cw = canvas.width;
    const ch = Math.max(160, Math.round(cw * (height / width)));
    if (canvas.height !== ch) {
      canvas.height = ch;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, cw, ch);

    // Obsidian Glass Backdrop + Rule-of-Thirds Grid
    const grad = ctx.createLinearGradient(0, 0, 0, ch);
    grad.addColorStop(0, invert ? "#131c2e" : "#0b101d");
    grad.addColorStop(1, invert ? "#0c1220" : "#070a13");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, cw, ch);

    ctx.strokeStyle = "rgba(148, 163, 184, 0.14)";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 2; i++) {
      ctx.beginPath();
      ctx.moveTo((cw * i) / 3, 0);
      ctx.lineTo((cw * i) / 3, ch);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, (ch * i) / 3);
      ctx.lineTo(cw, (ch * i) / 3);
      ctx.stroke();
    }

    const shortPx = Math.min(cw, ch);
    const featherPx = feather * shortPx;

    if (mode === "portal_path") {
      const evalP = evalPortalAtFrame(state.doc, state.frame, length);
      // Draw full smooth spline trail
      ctx.beginPath();
      for (let f = 0; f < length; f++) {
        const pt = evalPortalAtFrame(state.doc, f, length);
        const px = pt.x * cw;
        const py = pt.y * ch;
        if (f === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = "rgba(125, 249, 255, 0.35)";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Current interpolated portal preview + feather halo
      const px = evalP.x * cw;
      const py = evalP.y * ch;
      if (evalP.shape === "vertical_split") {
        ctx.fillStyle = invert ? "rgba(250, 204, 21, 0.16)" : "rgba(34, 211, 238, 0.16)";
        if (invert) {
          ctx.fillRect(0, 0, px, ch);
        } else {
          ctx.fillRect(px, 0, cw - px, ch);
        }
        ctx.strokeStyle = "rgba(125, 249, 255, 0.92)";
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(px, 0);
        ctx.lineTo(px, ch);
        ctx.stroke();
      } else {
        const rxPx = evalP.shape === "ellipse" ? evalP.rx * cw : evalP.rx * shortPx;
        const ryPx = evalP.shape === "ellipse" ? evalP.ry * ch : evalP.rx * shortPx;
        ctx.save();
        ctx.beginPath();
        ctx.ellipse(px, py, Math.max(4, rxPx + featherPx * 0.5), Math.max(4, ryPx + featherPx * 0.5), 0, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(34, 211, 238, 0.25)";
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.beginPath();
        ctx.ellipse(px, py, Math.max(4, rxPx), Math.max(4, ryPx), 0, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(34, 211, 238, 0.18)";
        ctx.fill();
        ctx.strokeStyle = "#7df9ff";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();

        // Rim radius drag knob on active portal
        ctx.fillStyle = "#fde047";
        ctx.beginPath();
        ctx.arc(px + rxPx, py, 5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Draw each Portal Keyframe handle
      evalP.keyframes.forEach((kf, kIdx) => {
        const kx = kf.x * cw;
        const ky = kf.y * ch;
        const isSel = state.selected.kind === "portal" && state.selected.index === kf.idx;
        ctx.beginPath();
        ctx.arc(kx, ky, isSel ? 7 : 5, 0, Math.PI * 2);
        ctx.fillStyle = isSel ? "#fde047" : "#22d3ee";
        ctx.fill();
        ctx.strokeStyle = "#090d16";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = isSel ? "#fde047" : "#cbd5e1";
        ctx.font = "bold 10px ui-monospace, monospace";
        ctx.fillText(`K${kIdx + 1}·f${kf.frame}`, kx + 8, ky - 6);
      });

      statusLeft.textContent = `Portal: x=${evalP.x.toFixed(2)} y=${evalP.y.toFixed(2)} r=${evalP.rx.toFixed(2)} (${evalP.shape})`;
      statusRight.textContent = `Drag dot=move · Yellow rim=resize · f=${state.frame}`;
    } else {
      // point_targets + Genie D-pad visualization
      state.doc.targets.forEach((t, idx) => {
        const tx = Number(t.x ?? 0.5) * cw;
        const ty = Number(t.y ?? 0.5) * ch;
        const tr = Math.max(8, Number(t.radius ?? 0.18) * shortPx);
        const tw = targetTemporalWeight(t, state.frame, length);
        const isSel = state.selected.kind === "target" && state.selected.index === idx;

        ctx.save();
        ctx.beginPath();
        ctx.arc(tx, ty, tr, 0, Math.PI * 2);
        ctx.fillStyle = tw > 0 ? `rgba(244, 114, 182, ${(0.12 + 0.22 * tw).toFixed(2)})` : "rgba(148, 163, 184, 0.06)";
        ctx.fill();
        ctx.strokeStyle = isSel ? "#fde047" : tw > 0 ? "#f472b6" : "rgba(148, 163, 184, 0.45)";
        ctx.lineWidth = isSel ? 2.5 : 1.5;
        if (tw <= 0) ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();

        // Crosshair ticks
        ctx.strokeStyle = tw > 0 ? "#7df9ff" : "rgba(148,163,184,0.6)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(tx - 9, ty);
        ctx.lineTo(tx + 9, ty);
        ctx.moveTo(tx, ty - 9);
        ctx.lineTo(tx, ty + 9);
        ctx.stroke();

        // Rim resize dot
        ctx.fillStyle = isSel ? "#fde047" : "#f472b6";
        ctx.beginPath();
        ctx.arc(tx + tr, ty, 4.5, 0, Math.PI * 2);
        ctx.fill();

        const snapCue = snapTo3f(resolveFrame(t.start ?? 0, length), length);
        ctx.fillStyle = tw > 0 ? "#f9a8d4" : "#94a3b8";
        ctx.font = "bold 10px ui-monospace, monospace";
        ctx.fillText(
          `${t.label || "TARGET"} [${t.start}-${t.end}] ♪${t.cue}@f${snapCue}`,
          clamp(tx - tr, 6, cw - 150),
          Math.max(12, ty - tr - 6)
        );
      });

      // Genie D-Pad Sprite Trail & Waypoints
      const wps = state.doc.dpad?.waypoints || [];
      if (wps.length > 0) {
        ctx.beginPath();
        for (let f = 0; f < length; f++) {
          const sp = evalSpriteAtFrame(state.doc, f, length);
          if (!sp) continue;
          const sx = sp.x * cw;
          const sy = sp.y * ch;
          if (f === 0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }
        ctx.strokeStyle = "rgba(250, 204, 21, 0.6)";
        ctx.lineWidth = 2;
        ctx.stroke();

        wps.forEach((wp, idx) => {
          const wx = Number(wp.x ?? 0.5) * cw;
          const wy = Number(wp.y ?? 0.72) * ch;
          const isSel = state.selected.kind === "dpad" && state.selected.index === idx;
          ctx.fillStyle = isSel ? "#ffffff" : "#fde047";
          ctx.beginPath();
          ctx.arc(wx, wy, isSel ? 6 : 4.5, 0, Math.PI * 2);
          ctx.fill();
        });

        const liveSp = evalSpriteAtFrame(state.doc, state.frame, length);
        if (liveSp) {
          const sx = liveSp.x * cw;
          const sy = liveSp.y * ch;
          ctx.strokeStyle = "#fde047";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(sx, sy, 8, 0, Math.PI * 2);
          ctx.stroke();
          ctx.fillStyle = "#fde047";
          ctx.font = "bold 10px ui-monospace, monospace";
          ctx.fillText(liveSp.hopping ? "HOP!" : `SPRITE(${liveSp.face})`, sx + 10, sy + 4);
        }
      }

      statusLeft.textContent = `Targets: ${state.doc.targets.length} · D-Pad wps: ${wps.length} (3f audio snap)`;
      statusRight.textContent = `Drag target/sprite · Rim dot=radius · f=${state.frame}`;
    }

    // Bottom Mini-Sequencer Bar inside canvas bottom 14px
    const barY = ch - 14;
    ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
    ctx.fillRect(0, barY, cw, 14);
    if (mode === "point_targets") {
      state.doc.targets.forEach((t) => {
        const s = resolveFrame(t.start ?? 0, length);
        const e = resolveFrame(t.end ?? length - 1, length);
        const x0 = (s / Math.max(1, length - 1)) * cw;
        const x1 = (e / Math.max(1, length - 1)) * cw;
        ctx.fillStyle = "rgba(244, 114, 182, 0.7)";
        ctx.fillRect(x0, barY + 3, Math.max(4, x1 - x0), 4);
        const cs = snapTo3f(s, length);
        const cx = (cs / Math.max(1, length - 1)) * cw;
        ctx.fillStyle = "#34d399";
        ctx.fillRect(cx - 1.5, barY + 8, 3, 4);
      });
    } else {
      ctx.fillStyle = "rgba(34, 211, 238, 0.55)";
      ctx.fillRect(0, barY + 4, cw, 5);
    }
    const playX = (state.frame / Math.max(1, length - 1)) * cw;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(playX - 1, barY, 2, 14);
  }

  function pointerToNorm(ev) {
    const rect = canvas.getBoundingClientRect();
    const nx = clamp((ev.clientX - rect.left) / Math.max(1, rect.width), 0.02, 0.98);
    const ny = clamp((ev.clientY - rect.top) / Math.max(1, rect.height), 0.02, 0.95);
    return { nx, ny, rect };
  }

  function hitTest(nx, ny) {
    const { mode, width, height, length } = getParams();
    const cw = canvas.width;
    const ch = canvas.height;
    const shortPx = Math.min(cw, ch);
    const px = nx * cw;
    const py = ny * ch;

    if (mode === "portal_path") {
      const evalP = evalPortalAtFrame(state.doc, state.frame, length);
      if (evalP.shape !== "vertical_split") {
        const rxPx = evalP.shape === "ellipse" ? evalP.rx * cw : evalP.rx * shortPx;
        const rimX = evalP.x * cw + rxPx;
        const rimY = evalP.y * ch;
        if (Math.hypot(px - rimX, py - rimY) <= 14) {
          // Match nearest portal keyframe to current frame for radius edit
          let bestIdx = 0;
          let bestDf = 1e9;
          state.doc.portal.forEach((p, idx) => {
            const df = Math.abs(resolveFrame(p.frame ?? 0, length) - state.frame);
            if (df < bestDf) {
              bestDf = df;
              bestIdx = idx;
            }
          });
          return { kind: "portal_rim", index: bestIdx };
        }
      }
      let bestIdx = -1;
      let bestD = 18;
      state.doc.portal.forEach((p, idx) => {
        const d = Math.hypot(px - Number(p.x ?? 0.5) * cw, py - Number(p.y ?? 0.5) * ch);
        if (d < bestD) {
          bestD = d;
          bestIdx = idx;
        }
      });
      if (bestIdx >= 0) return { kind: "portal", index: bestIdx };

      // Fallback: nearest keyframe in time
      let timeBest = 0;
      let timeDf = 1e9;
      state.doc.portal.forEach((p, idx) => {
        const df = Math.abs(resolveFrame(p.frame ?? 0, length) - state.frame);
        if (df < timeDf) {
          timeDf = df;
          timeBest = idx;
        }
      });
      return { kind: "portal", index: timeBest };
    }

    // point_targets: check target rim dots -> D-pad waypoints -> target centers
    for (let idx = 0; idx < state.doc.targets.length; idx++) {
      const t = state.doc.targets[idx];
      const tr = Math.max(8, Number(t.radius ?? 0.18) * shortPx);
      const rimX = Number(t.x ?? 0.5) * cw + tr;
      const rimY = Number(t.y ?? 0.5) * ch;
      if (Math.hypot(px - rimX, py - rimY) <= 13) {
        return { kind: "target_rim", index: idx };
      }
    }
    for (let idx = 0; idx < (state.doc.dpad?.waypoints || []).length; idx++) {
      const wp = state.doc.dpad.waypoints[idx];
      if (Math.hypot(px - Number(wp.x ?? 0.5) * cw, py - Number(wp.y ?? 0.72) * ch) <= 14) {
        return { kind: "dpad", index: idx };
      }
    }
    let bestT = 0;
    let bestD = 1e9;
    state.doc.targets.forEach((t, idx) => {
      const d = Math.hypot(px - Number(t.x ?? 0.5) * cw, py - Number(t.y ?? 0.5) * ch);
      if (d < bestD) {
        bestD = d;
        bestT = idx;
      }
    });
    return { kind: "target", index: bestT };
  }

  canvas.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    ev.preventDefault();
    ev.stopPropagation();
    canvas.focus();
    const { nx, ny } = pointerToNorm(ev);
    const hit = hitTest(nx, ny);
    state.selected = { kind: hit.kind.replace("_rim", ""), index: hit.index };
    state.dragging = hit;
    try {
      canvas.setPointerCapture(ev.pointerId);
    } catch {
      // headless / synthetic events may skip capture
    }
    applyDrag(nx, ny);
  });

  function applyDrag(nx, ny) {
    if (!state.dragging) return;
    const { width, height } = getParams();
    const shortRatio = Math.min(width, height);
    const drag = state.dragging;

    if (drag.kind === "portal") {
      const p = state.doc.portal[drag.index];
      if (p) {
        p.x = Number(nx.toFixed(3));
        p.y = Number(ny.toFixed(3));
      }
    } else if (drag.kind === "portal_rim") {
      const p = state.doc.portal[drag.index];
      if (p) {
        const dxPx = (nx - Number(p.x ?? 0.5)) * width;
        const dyPx = (ny - Number(p.y ?? 0.5)) * height;
        const r = clamp(Math.hypot(dxPx, dyPx) / Math.max(1, shortRatio), 0.08, 0.55);
        p.rx = Number(r.toFixed(3));
        p.ry = Number((r * 1.15).toFixed(3));
      }
    } else if (drag.kind === "target") {
      const t = state.doc.targets[drag.index];
      if (t) {
        t.x = Number(nx.toFixed(3));
        t.y = Number(ny.toFixed(3));
      }
    } else if (drag.kind === "target_rim") {
      const t = state.doc.targets[drag.index];
      if (t) {
        const dxPx = (nx - Number(t.x ?? 0.5)) * width;
        const dyPx = (ny - Number(t.y ?? 0.5)) * height;
        t.radius = Number(clamp(Math.hypot(dxPx, dyPx) / Math.max(1, shortRatio), 0.06, 0.45).toFixed(3));
      }
    } else if (drag.kind === "dpad") {
      const wp = state.doc.dpad?.waypoints?.[drag.index];
      if (wp) {
        wp.x = Number(nx.toFixed(3));
        wp.y = Number(ny.toFixed(3));
      }
    }
    pushDocToWidget();
    drawStage();
  }

  canvas.addEventListener("pointermove", (ev) => {
    if (!state.dragging) return;
    ev.preventDefault();
    ev.stopPropagation();
    const { nx, ny } = pointerToNorm(ev);
    applyDrag(nx, ny);
  });

  function stopDrag(ev) {
    if (!state.dragging) return;
    state.dragging = null;
    try {
      if (ev?.pointerId != null && canvas.hasPointerCapture(ev.pointerId)) {
        canvas.releasePointerCapture(ev.pointerId);
      }
    } catch {
      // ignore
    }
  }
  canvas.addEventListener("pointerup", stopDrag);
  canvas.addEventListener("pointercancel", stopDrag);
  canvas.addEventListener("lostpointercapture", () => {
    state.dragging = null;
  });

  canvas.addEventListener("keydown", (ev) => {
    const step = ev.shiftKey ? 0.005 : 0.025;
    let handled = true;
    const { mode } = getParams();

    const activeObj =
      mode === "portal_path"
        ? state.doc.portal[clamp(state.selected.index, 0, state.doc.portal.length - 1)]
        : state.selected.kind === "dpad"
        ? state.doc.dpad?.waypoints?.[clamp(state.selected.index, 0, Math.max(0, (state.doc.dpad?.waypoints?.length || 1) - 1))]
        : state.doc.targets[clamp(state.selected.index, 0, state.doc.targets.length - 1)];

    if (ev.key === "ArrowLeft" && activeObj) {
      activeObj.x = Number(clamp(Number(activeObj.x ?? 0.5) - step, 0.02, 0.98).toFixed(3));
    } else if (ev.key === "ArrowRight" && activeObj) {
      activeObj.x = Number(clamp(Number(activeObj.x ?? 0.5) + step, 0.02, 0.98).toFixed(3));
    } else if (ev.key === "ArrowUp" && activeObj) {
      activeObj.y = Number(clamp(Number(activeObj.y ?? 0.5) - step, 0.02, 0.95).toFixed(3));
    } else if (ev.key === "ArrowDown" && activeObj) {
      activeObj.y = Number(clamp(Number(activeObj.y ?? 0.5) + step, 0.02, 0.95).toFixed(3));
    } else if (ev.key === "[") {
      state.selected.index = Math.max(0, state.selected.index - 1);
    } else if (ev.key === "]") {
      state.selected.index = state.selected.index + 1;
    } else if (ev.key === " ") {
      togglePlayback();
    } else {
      handled = false;
    }
    if (handled) {
      ev.preventDefault();
      ev.stopPropagation();
      pushDocToWidget();
      drawStage();
    }
  });

  // Button & Select Wiring
  function setMode(nextMode) {
    if (wMode) {
      wMode.value = nextMode;
      wMode.callback?.(nextMode);
    }
    state.selected = { kind: nextMode === "portal_path" ? "portal" : "target", index: 0 };
    drawStage();
    announce(`Mode switched to ${nextMode}`);
  }

  btnPortalMode.addEventListener("click", (e) => {
    e.stopPropagation();
    setMode("portal_path");
  });
  btnTargetMode.addEventListener("click", (e) => {
    e.stopPropagation();
    setMode("point_targets");
  });

  btnSplitPreset.addEventListener("click", (e) => {
    e.stopPropagation();
    state.doc = rescalePresetForLength(PRESET_PORTAL_SPLIT, getParams().length);
    state.selected = { kind: "portal", index: 1 };
    if (wMode) {
      wMode.value = "portal_path";
      wMode.callback?.("portal_path");
    }
    pushDocToWidget();
    drawStage();
  });

  btnGeniePreset.addEventListener("click", (e) => {
    e.stopPropagation();
    state.doc = rescalePresetForLength(PRESET_GENIE_TARGETS, getParams().length);
    state.selected = { kind: "target", index: 0 };
    if (wMode) {
      wMode.value = "point_targets";
      wMode.callback?.("point_targets");
    }
    pushDocToWidget();
    drawStage();
  });

  btnResetPreset.addEventListener("click", (e) => {
    e.stopPropagation();
    state.doc = rescalePresetForLength(PRESET_PORTAL_CIRCLE, getParams().length);
    state.selected = { kind: "portal", index: 1 };
    pushDocToWidget();
    drawStage();
  });

  shapeSelect.addEventListener("change", (e) => {
    e.stopPropagation();
    const sh = shapeSelect.value;
    state.doc.portal.forEach((p) => {
      p.shape = sh;
      p.label = sh === "vertical_split" ? "SPLIT WIPE" : "REALITY LENS";
    });
    pushDocToWidget();
    drawStage();
  });

  cueSelect.addEventListener("change", (e) => {
    e.stopPropagation();
    const idx = clamp(state.selected.index, 0, Math.max(0, state.doc.targets.length - 1));
    if (state.doc.targets[idx]) {
      state.doc.targets[idx].cue = cueSelect.value;
      pushDocToWidget();
      drawStage();
    }
  });

  btnAddTarget.addEventListener("click", (e) => {
    e.stopPropagation();
    const { length } = getParams();
    const nextId = state.doc.targets.length + 1;
    const s = clamp(state.frame, 0, Math.max(0, length - 12));
    const end = clamp(s + 24, s + 6, Math.max(s + 6, length - 1));
    state.doc.targets.push({
      id: nextId,
      label: `TARGET ${String.fromCharCode(64 + nextId)}`,
      start: s,
      end,
      ramp: 4,
      x: Number(clamp(0.25 + ((nextId * 0.23) % 0.5), 0.15, 0.85).toFixed(3)),
      y: Number(clamp(0.35 + ((nextId * 0.17) % 0.35), 0.2, 0.8).toFixed(3)),
      radius: 0.18,
      cue: "pop",
      gain: 1.15,
    });
    state.selected = { kind: "target", index: state.doc.targets.length - 1 };
    pushDocToWidget();
    drawStage();
  });

  slider.addEventListener("input", (e) => {
    e.stopPropagation();
    state.frame = parseInt(slider.value, 10) || 0;
    drawStage();
  });

  function stopPlayback() {
    state.playing = false;
    btnPlay.textContent = "▶ Play";
    if (state.rafId) {
      cancelAnimationFrame(state.rafId);
      state.rafId = 0;
    }
  }

  function stepPlayback(nowMs) {
    if (!state.playing) return;
    if (!state.lastTickMs) state.lastTickMs = nowMs;
    const elapsedMs = nowMs - state.lastTickMs;
    const frameDur = 1000.0 / VIDEO_FPS;
    if (elapsedMs >= frameDur) {
      const steps = Math.max(1, Math.floor(elapsedMs / frameDur));
      const { length } = getParams();
      state.frame = (state.frame + steps) % Math.max(1, length);
      state.lastTickMs = nowMs - (elapsedMs % frameDur);
      drawStage();
    }
    state.rafId = requestAnimationFrame(stepPlayback);
  }

  function togglePlayback() {
    if (state.playing) {
      stopPlayback();
    } else {
      state.playing = true;
      state.lastTickMs = 0;
      btnPlay.textContent = "❚❚ Pause";
      state.rafId = requestAnimationFrame(stepPlayback);
    }
  }

  btnPlay.addEventListener("click", (e) => {
    e.stopPropagation();
    togglePlayback();
  });

  // Bidirectional hooks onto ComfyUI widgets
  for (const w of [wMode, wWidth, wHeight, wLength, wFeather, wInvert]) {
    if (!w) continue;
    const origCb = w.callback;
    w.callback = function (...args) {
      const r = origCb ? origCb.apply(this, args) : undefined;
      drawStage();
      return r;
    };
  }
  if (wEvents) {
    const origEventsCb = wEvents.callback;
    wEvents.callback = function (...args) {
      const r = origEventsCb ? origEventsCb.apply(this, args) : undefined;
      if (!state.suppressWidgetSync) {
        loadDocFromWidget();
        drawStage();
      }
      return r;
    };
  }

  const domWidget = node.addDOMWidget("spatial_director_stage", "h3_spatial_director", root, {
    serialize: false,
    hideOnZoom: false,
  });
  domWidget.computeSize = (w) => [Math.max(380, w || 420), 348];

  loadDocFromWidget();
  drawStage();

  return {
    refreshFromWidgets() {
      loadDocFromWidget();
      drawStage();
    },
    destroy() {
      stopPlayback();
    },
  };
}

app.registerExtension({
  name: "ComfyUI.MiniMaxH3.SpatialDirector",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
      this._h3SpatialDirectorCtrl = buildSpatialDirectorUI(this);
      const curSize = this.size || [420, 500];
      if (curSize[0] < 440 || curSize[1] < 540) {
        this.setSize?.([Math.max(curSize[0], 440), Math.max(curSize[1], 540)]);
      }
      return r;
    };

    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
      this._h3SpatialDirectorCtrl?.refreshFromWidgets();
      return r;
    };

    const origOnRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this._h3SpatialDirectorCtrl?.destroy();
      return origOnRemoved ? origOnRemoved.apply(this, arguments) : undefined;
    };
  },
});
