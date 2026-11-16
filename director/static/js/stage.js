/**
 * Sliderless Interactive 7:4 Stage Canvas & Masked Latent Preview Compositor (`stage.js`).
 *
 * Renders:
 * - Anchor still / finished MP4 backdrop + optional 32x32px patch grid
 * - Pixel-masked live preview video/image strictly inside feathered Portal + Halo
 * - Outer dashed Influence Halo (`Shift + Scroll` adjusts `halo_radius` 0.02..0.30)
 * - Inner Macro <-> Micro Aperture ring (`Scroll` adjusts `rx`/`ry`) + trajectory curve
 * - Portal Patch HUD pill (`Portal @ [x, y] · 32px Patch [col, row] · Span fA..fB`)
 * - Cross-Entity Halo Coupling Arc + Subject Sprite badge (`clean 0.999 · Release 0.33`)
 *   and `AppendSpriteTrajectory` radial `Δx/Δy` preview whenever Halo overlaps Sprite
 *   or `Shift + Drag` links them
 * - External rim EditGuider Power & Coupling badge (`Alt + Scroll` / ArrowUp/ArrowDown)
 * - Point-and-Click World Targets + 3f cue markers
 * - Interactive RGBA Sprite cutout + 0ms onion-skin motion trail (`WASD` / `Space` hop)
 * - Interactive `◄ A | B ►` split comparison overlay (`ab_compare.js`)
 *
 * Zero `element.style.*` mutations: all stage visuals paint onto <canvas>.
 */

import { snapFrameTo3f } from "./foley_synth.js";

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

export function cosineEase(u) {
  const c = clamp(u, 0.0, 1.0);
  return 0.5 - 0.5 * Math.cos(Math.PI * c);
}

export function evalPortalAtFrame(doc, frame, length) {
  const list = Array.isArray(doc?.portal) ? doc.portal : [];
  const kfs = list
    .map((p, idx) => ({
      idx,
      frame: clamp(Math.round(Number(p.frame ?? 0)), 0, Math.max(0, length - 1)),
      x: Number(p.x ?? 0.5),
      y: Number(p.y ?? 0.5),
      rx: Number(p.rx ?? p.radius ?? 0.22),
      ry: Number(p.ry ?? p.rx ?? p.radius ?? 0.22),
      shape: String(p.shape || "circle"),
      label: String(p.label || "REALITY LENS"),
      weight: Number(p.weight ?? 1.0),
      halo_radius: Number(p.halo_radius ?? 0.10),
      halo_weight: Number(p.halo_weight ?? 0.30),
    }))
    .sort((a, b) => a.frame - b.frame);

  if (kfs.length === 0) {
    return {
      x: 0.44,
      y: 0.48,
      rx: 0.20,
      ry: 0.20,
      shape: "circle",
      label: "REALITY LENS",
      weight: 1.0,
      halo_radius: 0.10,
      halo_weight: 0.30,
      temporalWeight: 1.0,
      keyframes: [],
    };
  }

  const pStart = doc.portal_start != null ? clamp(Number(doc.portal_start), 0, length - 1) : 0;
  const pEnd = doc.portal_end != null ? clamp(Number(doc.portal_end), 0, length - 1) : length - 1;
  const pRamp = Math.max(0, Number(doc.portal_ramp ?? 0));
  let temporalWeight = 1.0;
  if (frame < pStart || frame > pEnd) {
    temporalWeight = 0.0;
  } else if (pRamp >= 1 && pEnd - pStart >= 2) {
    const rEff = Math.min(pRamp, Math.max(1, Math.floor((pEnd - pStart) / 2)));
    const relIn = frame - pStart;
    const relOut = pEnd - frame;
    if (relIn < rEff) temporalWeight = cosineEase((relIn + 1) / (rEff + 1));
    else if (relOut < rEff) temporalWeight = cosineEase((relOut + 1) / (rEff + 1));
  }

  let out;
  if (frame <= kfs[0].frame) {
    out = { ...kfs[0] };
  } else if (frame >= kfs[kfs.length - 1].frame) {
    out = { ...kfs[kfs.length - 1] };
  } else {
    out = { ...kfs[kfs.length - 1] };
    for (let i = 0; i < kfs.length - 1; i++) {
      const a = kfs[i];
      const b = kfs[i + 1];
      if (a.frame <= frame && frame <= b.frame) {
        const u = cosineEase((frame - a.frame) / Math.max(1, b.frame - a.frame));
        const src = u < 0.5 ? a : b;
        out = {
          idx: src.idx,
          frame,
          x: a.x + (b.x - a.x) * u,
          y: a.y + (b.y - a.y) * u,
          rx: a.rx + (b.rx - a.rx) * u,
          ry: a.ry + (b.ry - a.ry) * u,
          shape: src.shape,
          label: src.label,
          weight: a.weight + (b.weight - a.weight) * u,
          halo_radius: a.halo_radius + (b.halo_radius - a.halo_radius) * u,
          halo_weight: a.halo_weight + (b.halo_weight - a.halo_weight) * u,
        };
        break;
      }
    }
  }
  return { ...out, temporalWeight, keyframes: kfs };
}

export function evalTargetTemporalWeight(t, frame, length) {
  const s = clamp(Math.round(Number(t.start ?? 0)), 0, Math.max(0, length - 1));
  const e = clamp(Math.round(Number(t.end ?? length - 1)), 0, Math.max(0, length - 1));
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

export function evalSpriteAtFrame(doc, frame, length) {
  const wps = [...(doc?.dpad?.waypoints || [])]
    .map((w, idx) => ({
      idx,
      frame: clamp(Math.round(Number(w.frame ?? 0)), 0, Math.max(0, length - 1)),
      x: Number(w.x ?? 0.35),
      y: Number(w.y ?? 0.72),
      face: String(w.face || "right"),
      scale: Number(w.scale ?? 1.0),
    }))
    .sort((a, b) => a.frame - b.frame);
  if (wps.length === 0) return null;

  let pose = { x: wps[0].x, y: wps[0].y, face: wps[0].face, scale: wps[0].scale, hopping: false, lift: 0 };
  if (frame >= wps[wps.length - 1].frame) {
    const last = wps[wps.length - 1];
    pose = { x: last.x, y: last.y, face: last.face, scale: last.scale, hopping: false, lift: 0 };
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
          scale: a.scale + (b.scale - a.scale) * u,
          hopping: false,
          lift: 0,
        };
        break;
      }
    }
  }

  for (const hp of doc?.dpad?.hops || []) {
    const hs = clamp(Math.round(Number(hp.start ?? 0)), 0, Math.max(0, length - 1));
    const he = clamp(Math.round(Number(hp.end ?? hs + 12)), 0, Math.max(0, length - 1));
    if (hs <= frame && frame <= he && he > hs) {
      const u = (frame - hs) / (he - hs);
      const parabola = 4 * u * (1 - u);
      const lift = Number(hp.lift ?? 0.25) * parabola;
      pose.y -= lift * 0.42;
      pose.hopping = true;
      pose.lift = lift;
      break;
    }
  }
  return pose;
}

/** Compute real H3 radial coupling strength and AppendSpriteTrajectory Δx/Δy. */
export function computeCouplingInfo(st, pf, sp) {
  const hasSprite = Boolean(sp && (st.spriteMode || st.spriteCutout?.img || (st.couplings || []).length > 0));
  if (!hasSprite || !sp || !pf) {
    return { active: false, strength: 0, mode: "push", dx: 0, dy: 0 };
  }
  const explicit = Array.isArray(st.couplings) && st.couplings.length ? st.couplings[0] : null;
  const w = Number(st.doc.width || 1344);
  const h = Number(st.doc.height || 768);
  const shortEdge = Math.min(w, h);
  const vx = (sp.x - pf.x) * (w / shortEdge);
  const vy = (sp.y - pf.y) * (h / shortEdge);
  const dist = Math.hypot(vx, vy);
  const halo = Number(pf.halo_radius || 0);
  const reach = Number(pf.rx || 0.20) + halo;
  const overlapping = dist <= reach + 0.04;

  let strength = 0;
  let mode = explicit?.mode === "pull" ? "pull" : "push";
  if (explicit && Number(explicit.strength) > 0) {
    strength = clamp(Number(explicit.strength), 0, 1);
  } else if (overlapping && halo > 0.01) {
    strength = clamp(1 - Math.max(0, dist - pf.rx * 0.4) / Math.max(0.08, reach), 0.15, 1.0);
  }
  if (strength <= 0.01) {
    return { active: false, strength: 0, mode, dx: 0, dy: 0 };
  }

  const rawDx = sp.x - pf.x;
  const rawDy = sp.y - pf.y;
  const norm = Math.hypot(rawDx, rawDy) || 1;
  const ux = rawDx / norm;
  const uy = rawDy / norm;
  const sign = mode === "pull" ? -1 : 1;
  const stepMag = 0.12 * strength;
  return {
    active: true,
    strength: Number(strength.toFixed(2)),
    mode,
    dx: Number((sign * ux * stepMag).toFixed(3)),
    dy: Number((sign * uy * stepMag).toFixed(3)),
  };
}

export class StageController {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.getState = opts.getState;
    this.onMutate = opts.onMutate || (() => {});
    this.onCardInspect = opts.onCardInspect || (() => {});
    this.onInteractHud = opts.onInteractHud || (() => {});
    this.abCompare = opts.abCompare || null;

    this._taeCanvas = document.createElement("canvas");
    this._taeCtx = this._taeCanvas.getContext("2d");

    this._drag = null; // { kind, idx, linkEndX, linkEndY }
    this._dashPhase = 0;
    this._animTimer = 0;

    this._bindEvents();
    this.resize();
  }

  resize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(320, Math.round(rect.width * dpr));
    const h = Math.max(180, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this._taeCanvas.width = w;
      this._taeCanvas.height = h;
    }
    this.render();
  }

  _canvasCoords(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const nx = clamp((clientX - rect.left) / Math.max(1, rect.width), 0.0, 1.0);
    const ny = clamp((clientY - rect.top) / Math.max(1, rect.height), 0.0, 1.0);
    return { nx, ny, rect };
  }

  _bindEvents() {
    this.canvas.addEventListener("pointerdown", (ev) => this._onPointerDown(ev));
    window.addEventListener("pointermove", (ev) => this._onPointerMove(ev));
    window.addEventListener("pointerup", () => this._onPointerUp());
    this.canvas.addEventListener("wheel", (ev) => this._onWheel(ev), { passive: false });
  }

  _hitTest(nx, ny, st) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const shortEdge = Math.min(w, h);
    const frame = st.frame;
    const len = st.length;

    if (this.abCompare?.active && this.abCompare.hitTestHandle(nx, ny, w, h)) {
      return { kind: "ab_split" };
    }

    if (st.spriteMode || st.spriteCutout?.img) {
      const sp = evalSpriteAtFrame(st.doc, frame, len);
      if (sp) {
        const dx = (nx - sp.x) * w;
        const dy = (ny - sp.y) * h;
        if (Math.hypot(dx, dy) <= 44 * (window.devicePixelRatio || 1)) {
          return { kind: "sprite" };
        }
      }
    }

    if (st.doc.mode === "point_targets") {
      const targets = st.doc.targets || [];
      for (let i = targets.length - 1; i >= 0; i--) {
        const t = targets[i];
        const dx = (nx - t.x) * w;
        const dy = (ny - t.y) * h;
        const rPx = t.radius * shortEdge;
        const dist = Math.hypot(dx, dy);
        if (Math.abs(dist - rPx) <= 14 * (window.devicePixelRatio || 1)) {
          return { kind: "target_radius", idx: i };
        }
        if (dist <= rPx) {
          return { kind: "target_move", idx: i };
        }
      }
      return { kind: "stage_empty" };
    }

    const pf = evalPortalAtFrame(st.doc, frame, len);
    const cx = pf.x * w;
    const cy = pf.y * h;
    const rxPx = pf.rx * (pf.shape === "ellipse" ? w : shortEdge);
    const ryPx = pf.ry * (pf.shape === "ellipse" ? h : shortEdge);
    const px = nx * w;
    const py = ny * h;
    const dpr = window.devicePixelRatio || 1;

    if (pf.shape === "vertical_split") {
      if (Math.abs(px - cx) <= 20 * dpr) {
        return { kind: "portal_move" };
      }
    } else {
      const hx = cx + rxPx;
      const hy = cy;
      if (Math.hypot(px - hx, py - hy) <= 16 * dpr) {
        return { kind: "portal_resize" };
      }
      const haloPx = (pf.halo_radius || 0) * shortEdge;
      const normDist = Math.hypot((px - cx) / Math.max(8, rxPx), (py - cy) / Math.max(8, ryPx));
      if (haloPx > 4 && Math.abs(Math.hypot(px - cx, py - cy) - (Math.max(rxPx, ryPx) + haloPx)) <= 14 * dpr) {
        return { kind: "portal_halo" };
      }
      if (normDist <= 1.08) {
        return { kind: "portal_move" };
      }
    }
    return { kind: "stage_empty" };
  }

  _onPointerDown(ev) {
    const st = this.getState();
    if (!st) return;
    const { nx, ny } = this._canvasCoords(ev.clientX, ev.clientY);

    if (this.abCompare?.active) {
      this._drag = { kind: "ab_split" };
      this.abCompare.setSplit(nx);
      return;
    }

    // Shift + Drag from portal links Cross-Entity Coupling to Sprite
    if (ev.shiftKey && st.doc.mode === "portal_path") {
      this._drag = { kind: "couple_link", linkEndX: nx, linkEndY: ny };
      this.canvas.setPointerCapture?.(ev.pointerId);
      this.render();
      return;
    }

    const hit = this._hitTest(nx, ny, st);
    if (hit.kind === "sprite") {
      this._drag = { kind: "sprite" };
      this._upsertSpriteWaypoint(st, nx, ny);
      this.canvas.setPointerCapture?.(ev.pointerId);
      return;
    }
    if (hit.kind === "target_move") {
      st.selectedTargetIdx = hit.idx;
      this._drag = { kind: "target_move", idx: hit.idx };
      this.onCardInspect({ kind: "target", idx: hit.idx });
      this.canvas.setPointerCapture?.(ev.pointerId);
      this.render();
      return;
    }
    if (hit.kind === "target_radius") {
      st.selectedTargetIdx = hit.idx;
      this._drag = { kind: "target_radius", idx: hit.idx };
      this.onCardInspect({ kind: "target", idx: hit.idx });
      this.canvas.setPointerCapture?.(ev.pointerId);
      return;
    }
    if (hit.kind === "portal_resize" || hit.kind === "portal_halo" || hit.kind === "portal_move") {
      this._drag = { kind: hit.kind };
      this.onCardInspect({ kind: "portal", frame: st.frame });
      this.canvas.setPointerCapture?.(ev.pointerId);
      return;
    }

    if (st.doc.mode === "point_targets") {
      if (ev.shiftKey || !(st.doc.targets || []).length) {
        const nextId = (st.doc.targets || []).reduce((m, t) => Math.max(m, Number(t.id || 0)), 0) + 1;
        const start = snapFrameTo3f(st.frame, st.length);
        const end = Math.min(st.length - 1, Math.max(start + 12, start + 36));
        const defaultCue = st.defaultSoundPreset || "none";
        st.doc.targets.push({
          id: nextId,
          label: `TARGET ${nextId}`,
          start,
          end,
          ramp: 4,
          x: Number(nx.toFixed(3)),
          y: Number(ny.toFixed(3)),
          radius: 0.16,
          cue: defaultCue,
          gain: 1.15,
          weight: 1.0,
          halo_radius: 0.08,
          halo_weight: 0.30,
        });
        st.selectedTargetIdx = st.doc.targets.length - 1;
        this._drag = { kind: "target_move", idx: st.selectedTargetIdx };
        this.onMutate("add_target");
        this.onCardInspect({ kind: "target", idx: st.selectedTargetIdx, placed: true });
      } else {
        const idx = clamp(st.selectedTargetIdx ?? 0, 0, st.doc.targets.length - 1);
        st.doc.targets[idx].x = Number(nx.toFixed(3));
        st.doc.targets[idx].y = Number(ny.toFixed(3));
        this._drag = { kind: "target_move", idx };
        this.onMutate("move_target");
        this.onCardInspect({ kind: "target", idx, placed: true });
      }
    } else {
      this._upsertPortalKeyframe(st, nx, ny);
      this._drag = { kind: "portal_move" };
      this.onMutate("place_portal");
      this.onCardInspect({ kind: "portal", frame: st.frame, placed: true });
    }
    this.render();
  }

  _onPointerMove(ev) {
    if (!this._drag) return;
    const st = this.getState();
    if (!st) return;
    const { nx, ny } = this._canvasCoords(ev.clientX, ev.clientY);

    if (this._drag.kind === "ab_split" && this.abCompare) {
      this.abCompare.setSplit(nx);
      return;
    }

    if (this._drag.kind === "couple_link") {
      this._drag.linkEndX = nx;
      this._drag.linkEndY = ny;
      this._upsertSpriteWaypoint(st, nx, ny);
      st.couplings = [{
        portal_index: 0,
        target_kind: "sprite",
        strength: 0.85,
        mode: st.couplings?.[0]?.mode || "push",
        label: "Sprite Coupling",
      }];
      this.onMutate("couple_link", { live: true });
      this.render();
      return;
    }

    if (this._drag.kind === "sprite") {
      this._upsertSpriteWaypoint(st, nx, ny);
      this.render();
      return;
    }

    if (this._drag.kind === "target_move") {
      const t = st.doc.targets?.[this._drag.idx];
      if (t) {
        t.x = Number(clamp(nx, 0.04, 0.96).toFixed(3));
        t.y = Number(clamp(ny, 0.06, 0.94).toFixed(3));
        this.onMutate("drag_target", { live: true });
        this.render();
      }
      return;
    }

    if (this._drag.kind === "target_radius") {
      const t = st.doc.targets?.[this._drag.idx];
      if (t) {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const shortEdge = Math.min(w, h);
        const dist = Math.hypot((nx - t.x) * w, (ny - t.y) * h) / shortEdge;
        t.radius = Number(clamp(dist, 0.04, 0.45).toFixed(3));
        this.onMutate("resize_target", { live: true });
        this.render();
      }
      return;
    }

    if (this._drag.kind === "portal_move") {
      this._upsertPortalKeyframe(st, clamp(nx, 0.04, 0.96), clamp(ny, 0.06, 0.94));
      this.onMutate("drag_portal", { live: true });
      this.render();
      return;
    }

    if (this._drag.kind === "portal_resize") {
      const kf = this._ensurePortalKeyframeAtFrame(st);
      const w = this.canvas.width;
      const h = this.canvas.height;
      const shortEdge = Math.min(w, h);
      const dx = Math.abs(nx - kf.x) * (kf.shape === "ellipse" ? w : shortEdge);
      const dy = Math.abs(ny - kf.y) * (kf.shape === "ellipse" ? h : shortEdge);
      const r = clamp(Math.max(dx, dy) / shortEdge, 0.04, 0.45);
      kf.rx = Number(r.toFixed(3));
      kf.ry = kf.shape === "ellipse" ? Number(clamp(r * 1.25, 0.04, 0.48).toFixed(3)) : kf.rx;
      this.onMutate("resize_portal", { live: true });
      this.render();
      return;
    }

    if (this._drag.kind === "portal_halo") {
      const kf = this._ensurePortalKeyframeAtFrame(st);
      const w = this.canvas.width;
      const h = this.canvas.height;
      const shortEdge = Math.min(w, h);
      const dist = Math.hypot((nx - kf.x) * w, (ny - kf.y) * h) / shortEdge;
      kf.halo_radius = Number(clamp(dist - kf.rx, 0.0, 0.35).toFixed(3));
      this.onMutate("halo_portal", { live: true });
      this.render();
    }
  }

  _onPointerUp() {
    if (!this._drag) return;
    const kind = this._drag.kind;
    this._drag = null;
    if (kind !== "ab_split") {
      this.onMutate(`commit_${kind}`);
    }
  }

  _onWheel(ev) {
    ev.preventDefault();
    const st = this.getState();
    if (!st) return;
    const dir = ev.deltaY > 0 ? -1 : 1;

    if (ev.altKey) {
      const next = clamp(Number(st.editGuidance.scale ?? 1.5) + dir * 0.1, 0.5, 2.5);
      st.editGuidance.scale = Number(next.toFixed(1));
      this.onMutate("power_wheel");
      this.onInteractHud();
      this.render();
      return;
    }

    if (ev.shiftKey) {
      if (st.doc.mode === "point_targets" && st.doc.targets?.length) {
        const idx = clamp(st.selectedTargetIdx ?? 0, 0, st.doc.targets.length - 1);
        const cur = Number(st.doc.targets[idx].halo_radius ?? 0.08);
        st.doc.targets[idx].halo_radius = Number(clamp(cur + dir * 0.02, 0.0, 0.35).toFixed(3));
      } else {
        const kf = this._ensurePortalKeyframeAtFrame(st);
        const cur = Number(kf.halo_radius ?? 0.10);
        kf.halo_radius = Number(clamp(cur + dir * 0.02, 0.0, 0.35).toFixed(3));
      }
      this.onMutate("halo_wheel");
      this.onInteractHud();
      this.render();
      return;
    }

    if (st.doc.mode === "point_targets" && st.doc.targets?.length) {
      const idx = clamp(st.selectedTargetIdx ?? 0, 0, st.doc.targets.length - 1);
      const cur = Number(st.doc.targets[idx].radius ?? 0.18);
      st.doc.targets[idx].radius = Number(clamp(cur + dir * 0.015, 0.04, 0.45).toFixed(3));
    } else {
      const kf = this._ensurePortalKeyframeAtFrame(st);
      const nextRx = clamp(Number(kf.rx ?? 0.22) + dir * 0.015, 0.04, 0.45);
      const ratio = kf.rx > 0.01 ? kf.ry / kf.rx : 1.0;
      kf.rx = Number(nextRx.toFixed(3));
      kf.ry = kf.shape === "ellipse" ? Number(clamp(nextRx * ratio, 0.04, 0.48).toFixed(3)) : kf.rx;
    }
    this.onMutate("aperture_wheel");
    this.onInteractHud();
    this.render();
  }

  setHaloRadius(val) {
    const st = this.getState?.();
    if (!st) return;
    const kf = this._ensurePortalKeyframeAtFrame(st);
    kf.halo_radius = Number(clamp(Number(val), 0.0, 0.35).toFixed(3));
    this.render();
  }

  _ensurePortalKeyframeAtFrame(st) {
    const f = st.frame;
    if (!Array.isArray(st.doc.portal)) st.doc.portal = [];
    let kf = st.doc.portal.find((p) => Math.abs(Number(p.frame) - f) <= 2);
    if (!kf) {
      const cur = evalPortalAtFrame(st.doc, f, st.length);
      kf = {
        frame: f,
        x: Number(cur.x.toFixed(3)),
        y: Number(cur.y.toFixed(3)),
        rx: Number(cur.rx.toFixed(3)),
        ry: Number(cur.ry.toFixed(3)),
        shape: cur.shape || "circle",
        label: cur.label || "REALITY LENS",
        weight: Number((cur.weight ?? 1.0).toFixed(2)),
        halo_radius: Number((cur.halo_radius ?? 0.10).toFixed(3)),
        halo_weight: Number((cur.halo_weight ?? 0.30).toFixed(2)),
      };
      st.doc.portal.push(kf);
      st.doc.portal.sort((a, b) => a.frame - b.frame);
    }
    return kf;
  }

  _upsertPortalKeyframe(st, nx, ny) {
    const kf = this._ensurePortalKeyframeAtFrame(st);
    kf.x = Number(nx.toFixed(3));
    kf.y = Number(ny.toFixed(3));
    return kf;
  }

  _upsertSpriteWaypoint(st, nx, ny, faceOverride = null) {
    if (!st.doc.dpad) st.doc.dpad = { waypoints: [], hops: [] };
    if (!Array.isArray(st.doc.dpad.waypoints)) st.doc.dpad.waypoints = [];
    const f = st.frame;
    let wp = st.doc.dpad.waypoints.find((w) => Math.abs(Number(w.frame) - f) <= 2);
    if (!wp) {
      const prev = evalSpriteAtFrame(st.doc, f, st.length);
      wp = {
        frame: f,
        x: Number(nx.toFixed(3)),
        y: Number(ny.toFixed(3)),
        face: faceOverride || prev?.face || "right",
        scale: prev?.scale || 1.0,
      };
      st.doc.dpad.waypoints.push(wp);
      st.doc.dpad.waypoints.sort((a, b) => a.frame - b.frame);
    } else {
      wp.x = Number(clamp(nx, 0.04, 0.96).toFixed(3));
      wp.y = Number(clamp(ny, 0.08, 0.94).toFixed(3));
      if (faceOverride) wp.face = faceOverride;
    }
    this.onMutate("sprite_waypoint");
  }

  nudgeSprite(dx, dy, faceOverride = null) {
    const st = this.getState();
    if (!st) return;
    const cur = evalSpriteAtFrame(st.doc, st.frame, st.length) || { x: 0.40, y: 0.72, face: "right" };
    this._upsertSpriteWaypoint(st, cur.x + dx, cur.y + dy, faceOverride);
    this.render();
  }

  triggerSpriteHop() {
    const st = this.getState();
    if (!st) return;
    if (!st.doc.dpad) st.doc.dpad = { waypoints: [], hops: [] };
    if (!Array.isArray(st.doc.dpad.hops)) st.doc.dpad.hops = [];
    const hs = clamp(st.frame, 0, Math.max(0, st.length - 7));
    const he = clamp(hs + 15, hs + 4, st.length - 1);
    st.doc.dpad.hops.push({ start: hs, end: he, lift: 0.25 });
    this.onMutate("sprite_hop");
    this.render();
  }

  _drawContainOrCover(ctx, media, w, h) {
    const mw = media.videoWidth || media.naturalWidth || media.width || w;
    const mh = media.videoHeight || media.naturalHeight || media.height || h;
    if (!mw || !mh) return;
    const scale = Math.max(w / mw, h / mh);
    const dw = mw * scale;
    const dh = mh * scale;
    ctx.drawImage(media, (w - dw) * 0.5, (h - dh) * 0.5, dw, dh);
  }

  _roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  _drawPill(ctx, cx, y, text, { bg = "rgba(255,255,255,0.95)", border = "#CBD5E1", fg = "#0F172A", dot = null } = {}) {
    ctx.save();
    ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
    const tw = Math.ceil(ctx.measureText(text).width);
    const dotPad = dot ? 12 : 0;
    const pw = tw + 18 + dotPad;
    const ph = 22;
    const x = clamp(Math.round(cx - pw * 0.5), 10, this.canvas.width - pw - 10);
    const py = clamp(Math.round(y), 10, this.canvas.height - ph - 10);
    ctx.fillStyle = bg;
    ctx.strokeStyle = border;
    ctx.lineWidth = 1;
    this._roundRect(ctx, x, py, pw, ph, 11);
    ctx.fill();
    ctx.stroke();
    if (dot) {
      ctx.fillStyle = dot;
      ctx.beginPath();
      ctx.arc(x + 11, py + ph * 0.5, 3, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = fg;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + 9 + dotPad, py + ph * 0.5 + 0.5);
    ctx.restore();
  }

  _compositeMaskedLivePreview(ctx, st, w, h, shortEdge, pf) {
    const lp = st.livePreview;
    if (!lp || !lp.media) return;
    const media = lp.media;
    if (lp.kind === "video") {
      if (media.readyState < 2) return;
      if (Number.isFinite(media.duration) && media.duration > 0) {
        const phase = st.frame / Math.max(1, st.length - 1);
        const wantT = phase * media.duration;
        if (Math.abs(media.currentTime - wantT) > 0.03) {
          try {
            media.currentTime = wantT;
          } catch {}
        }
      }
    }

    if (lp.fullFrame) {
      ctx.save();
      ctx.globalAlpha = 0.94;
      this._drawContainOrCover(ctx, media, w, h);
      ctx.restore();
      return;
    }

    const tCtx = this._taeCtx;
    tCtx.clearRect(0, 0, w, h);
    tCtx.save();
    tCtx.globalCompositeOperation = "source-over";
    this._drawContainOrCover(tCtx, media, w, h);
    tCtx.globalCompositeOperation = "destination-in";

    const featherPx = Math.max(4, Number(st.doc.feather ?? 0.08) * shortEdge);
    if (st.doc.mode === "point_targets") {
      for (const t of st.doc.targets || []) {
        const tw = evalTargetTemporalWeight(t, st.frame, st.length);
        if (tw <= 0.01) continue;
        const tx = t.x * w;
        const ty = t.y * h;
        const rInner = Math.max(8, t.radius * shortEdge);
        const rHalo = Number(t.halo_radius || 0) * shortEdge;
        const rOuter = rInner + Math.max(featherPx, rHalo);
        const grad = tCtx.createRadialGradient(tx, ty, Math.max(1, rInner - featherPx * 0.5), tx, ty, rOuter);
        grad.addColorStop(0, `rgba(0,0,0,${tw.toFixed(3)})`);
        if (rHalo > featherPx * 0.5) {
          grad.addColorStop(clamp((rInner + featherPx * 0.4) / rOuter, 0.01, 0.95), `rgba(0,0,0,${(0.30 * tw).toFixed(3)})`);
        }
        grad.addColorStop(1, "rgba(0,0,0,0)");
        tCtx.fillStyle = grad;
        tCtx.beginPath();
        tCtx.arc(tx, ty, rOuter, 0, Math.PI * 2);
        tCtx.fill();
      }
    } else if (pf.temporalWeight > 0.01) {
      const cx = pf.x * w;
      const cy = pf.y * h;
      const tw = pf.temporalWeight;
      if (pf.shape === "vertical_split") {
        const g = tCtx.createLinearGradient(cx - featherPx, 0, cx + featherPx, 0);
        g.addColorStop(0, "rgba(0,0,0,0)");
        g.addColorStop(1, `rgba(0,0,0,${tw.toFixed(3)})`);
        tCtx.fillStyle = g;
        tCtx.fillRect(cx - featherPx, 0, featherPx * 2, h);
        tCtx.fillStyle = `rgba(0,0,0,${tw.toFixed(3)})`;
        tCtx.fillRect(cx + featherPx, 0, w - (cx + featherPx), h);
      } else {
        const rx = Math.max(8, pf.rx * (pf.shape === "ellipse" ? w : shortEdge));
        const ry = Math.max(8, pf.ry * (pf.shape === "ellipse" ? h : shortEdge));
        const haloPx = Math.max(0, Number(pf.halo_radius || 0) * shortEdge);
        const rMax = Math.max(rx, ry);
        const rOuter = rMax + Math.max(featherPx, haloPx);
        tCtx.save();
        tCtx.translate(cx, cy);
        tCtx.scale(rx / rMax, ry / rMax);
        const grad = tCtx.createRadialGradient(0, 0, Math.max(1, rMax - featherPx * 0.5), 0, 0, rOuter);
        grad.addColorStop(0, `rgba(0,0,0,${tw.toFixed(3)})`);
        if (haloPx > 4) {
          const haloAlpha = clamp(Number(pf.halo_weight ?? 0.30) * tw, 0, 1);
          grad.addColorStop(clamp((rMax + featherPx * 0.35) / rOuter, 0.05, 0.92), `rgba(0,0,0,${haloAlpha.toFixed(3)})`);
        }
        grad.addColorStop(1, "rgba(0,0,0,0)");
        tCtx.fillStyle = grad;
        tCtx.beginPath();
        tCtx.arc(0, 0, rOuter, 0, Math.PI * 2);
        tCtx.fill();
        tCtx.restore();
      }
    }
    tCtx.restore();

    ctx.drawImage(this._taeCanvas, 0, 0);
  }

  _drawPatchGrid(ctx, w, h, st) {
    const scaleX = w / Math.max(64, Number(st.doc.width || 1344));
    const step = Math.max(16, 32 * scaleX);
    ctx.save();
    ctx.strokeStyle = "rgba(37, 99, 235, 0.14)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x <= w; x += step) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let y = 0; y <= h; y += step) {
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  render() {
    const st = this.getState?.();
    if (!st || !this.ctx) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const shortEdge = Math.min(w, h);
    const dpr = window.devicePixelRatio || 1;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#EEF2F6";
    ctx.fillRect(0, 0, w, h);

    let baseMedia = st.anchorMedia || null;
    if (st.activeVideoEl && st.activeVideoEl.readyState >= 2) {
      if (Number.isFinite(st.activeVideoEl.duration) && st.activeVideoEl.duration > 0) {
        const wantT = (st.frame / Math.max(1, st.length - 1)) * st.activeVideoEl.duration;
        if (Math.abs(st.activeVideoEl.currentTime - wantT) > 0.03) {
          try {
            st.activeVideoEl.currentTime = wantT;
          } catch {}
        }
      }
      baseMedia = st.activeVideoEl;
    }

    if (this.abCompare?.active) {
      const mA = this.abCompare.getMediaA(st.anchorMedia);
      const mB = this.abCompare.getMediaB(st.livePreview?.media || baseMedia);
      this.abCompare.syncTime(st.frame, st.length);
      this.abCompare.drawSplitOverlay(ctx, w, h, mA, mB);
      return;
    }

    if (baseMedia) {
      this._drawContainOrCover(ctx, baseMedia, w, h);
    } else {
      this._drawPatchGrid(ctx, w, h, st);
    }

    if (st.showPatchGrid && baseMedia) {
      this._drawPatchGrid(ctx, w, h, st);
    }

    const pf = evalPortalAtFrame(st.doc, st.frame, st.length);
    if (st.livePreview?.media) {
      this._compositeMaskedLivePreview(ctx, st, w, h, shortEdge, pf);
    }

    if (!st.hasAnchor && !st.livePreview?.media) {
      return;
    }

    const sp = (st.spriteMode || st.spriteCutout?.img || (st.couplings || []).length > 0)
      ? evalSpriteAtFrame(st.doc, st.frame, st.length)
      : null;
    const coupling = computeCouplingInfo(st, pf, sp);

    if (st.doc.mode === "portal_path") {
      this._drawPortalOverlay(ctx, st, pf, sp, coupling, w, h, shortEdge, dpr);
    } else {
      this._drawTargetsOverlay(ctx, st, w, h, shortEdge, dpr);
    }

    if (sp) {
      this._drawSpriteOverlay(ctx, st, sp, coupling, w, h, dpr);
    }

    if (coupling.active && sp) {
      this._drawCouplingArc(ctx, pf, sp, coupling, w, h, dpr);
    }

    // Dotted visual tether linking active Portal/Target to the docked Contextual Popover Card
    if (st.hasAnchor) {
      const cardEl = document.getElementById("portal-card");
      if (cardEl && !cardEl.classList.contains("hidden") && !cardEl.classList.contains("collapsed")) {
        let srcX = pf.x * w + (pf.rx * shortEdge);
        let srcY = pf.y * h;
        let dockLeft = pf.x > 0.56;
        if (st.doc.mode === "point_targets" && st.doc.targets?.length) {
          const idx = clamp(st.selectedTargetIdx ?? 0, 0, st.doc.targets.length - 1);
          const t = st.doc.targets[idx];
          srcX = t.x * w + (dockLeft ? -t.radius * shortEdge : t.radius * shortEdge);
          srcY = t.y * h;
          dockLeft = t.x > 0.56;
        } else if (dockLeft) {
          srcX = pf.x * w - (pf.rx * shortEdge);
        }
        const dstX = dockLeft ? Math.min(w * 0.27, 344 * (w / 1280)) : Math.max(w * 0.72, w - 344 * (w / 1280));
        const dstY = h * 0.34;
        ctx.save();
        const tGrad = ctx.createLinearGradient(srcX, srcY, dstX, dstY);
        tGrad.addColorStop(0, "rgba(2, 132, 199, 0.72)");
        tGrad.addColorStop(1, "rgba(37, 99, 235, 0.42)");
        ctx.strokeStyle = tGrad;
        ctx.lineWidth = 1.5 * dpr;
        ctx.setLineDash([4 * dpr, 4 * dpr]);
        ctx.beginPath();
        const midX = (srcX + dstX) * 0.5;
        ctx.moveTo(srcX, srcY);
        ctx.bezierCurveTo(midX, srcY, midX, dstY, dstX, dstY);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#0284C7";
        ctx.beginPath();
        ctx.arc(srcX, srcY, 3 * dpr, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#2563EB";
        ctx.beginPath();
        ctx.arc(dstX, dstY, 3 * dpr, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }
  }

  _drawCouplingArc(ctx, pf, sp, coupling, w, h, dpr) {
    const x0 = pf.x * w;
    const y0 = pf.y * h;
    const x1 = sp.x * w;
    const y1 = sp.y * h;
    const mx = (x0 + x1) * 0.5 + (y1 - y0) * 0.14;
    const my = (y0 + y1) * 0.5 - (x1 - x0) * 0.08;

    ctx.save();
    const grad = ctx.createLinearGradient(x0, y0, x1, y1);
    grad.addColorStop(0, "rgba(37, 99, 235, 0.92)");
    grad.addColorStop(1, "rgba(4, 120, 87, 0.92)");
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.2 * dpr;
    ctx.setLineDash([5 * dpr, 4 * dpr]);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.quadraticCurveTo(mx, my, x1, y1);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#2563EB";
    ctx.beginPath();
    ctx.arc(x0, y0, 3.5 * dpr, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#047857";
    ctx.beginPath();
    ctx.arc(x1, y1, 4.5 * dpr, 0, Math.PI * 2);
    ctx.fill();

    const pct = Math.round(coupling.strength * 100);
    const sx = coupling.dx >= 0 ? `+${coupling.dx.toFixed(2)}` : coupling.dx.toFixed(2);
    const sy = coupling.dy >= 0 ? `+${coupling.dy.toFixed(2)}` : coupling.dy.toFixed(2);
    this._drawPill(ctx, mx, my - 12 * dpr, `Coupled ${pct}% · Δxy(${sx}, ${sy})`, {
      bg: "rgba(255, 255, 255, 0.95)",
      border: "#A7F3D0",
      fg: "#047857",
      dot: "#047857",
    });
    ctx.restore();
  }

  _drawPortalOverlay(ctx, st, pf, sp, coupling, w, h, shortEdge, dpr) {
    const cx = pf.x * w;
    const cy = pf.y * h;
    const kfs = pf.keyframes || [];

    if (kfs.length >= 2) {
      ctx.save();
      ctx.strokeStyle = "rgba(37, 99, 235, 0.72)";
      ctx.lineWidth = 1.6 * dpr;
      ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.beginPath();
      for (let f = 0; f < st.length; f += 2) {
        const pt = evalPortalAtFrame(st.doc, f, st.length);
        const px = pt.x * w;
        const py = pt.y * h;
        if (f === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      for (let i = 0; i < kfs.length; i++) {
        const kf = kfs[i];
        const kx = kf.x * w;
        const ky = kf.y * h;
        ctx.fillStyle = i === 0 ? "#2563EB" : "#0284C7";
        ctx.beginPath();
        ctx.arc(kx, ky, 4 * dpr, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 1.5 * dpr;
        ctx.stroke();

        const tag = `${kf.rx >= 0.18 ? "Macro" : "Micro"} (${kf.rx.toFixed(2)} @ f${kf.frame})`;
        const lblY = clamp(ky + (i % 2 === 0 ? 10 : -32) * dpr, 48 * dpr, h - 80 * dpr);
        this._drawPill(ctx, clamp(kx, 80 * dpr, w - 80 * dpr), lblY, tag, {
          bg: "rgba(255, 255, 255, 0.90)",
          border: "#BFDBFE",
          fg: "#1E3A8A",
        });
      }
      ctx.restore();
    }

    if (pf.shape === "vertical_split") {
      ctx.save();
      ctx.strokeStyle = "rgba(2, 132, 199, 0.30)";
      ctx.lineWidth = 8 * dpr;
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, h);
      ctx.stroke();

      ctx.strokeStyle = "#0284C7";
      ctx.lineWidth = 2.5 * dpr;
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, h);
      ctx.stroke();

      ctx.fillStyle = "#FFFFFF";
      ctx.strokeStyle = "#2563EB";
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      ctx.arc(cx, cy, 10 * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      return;
    }

    const rxPx = pf.rx * (pf.shape === "ellipse" ? w : shortEdge);
    const ryPx = pf.ry * (pf.shape === "ellipse" ? h : shortEdge);
    const haloPx = Math.max(0, Number(pf.halo_radius || 0) * shortEdge);
    const hrX = rxPx + haloPx;
    const hrY = ryPx + haloPx;

    // Outer Concentric Dashed Influence Halo Ring
    if (haloPx > 2) {
      ctx.save();
      ctx.fillStyle = "rgba(2, 132, 199, 0.04)";
      ctx.strokeStyle = "rgba(2, 132, 199, 0.62)";
      ctx.lineWidth = 1.5 * dpr;
      ctx.setLineDash([6 * dpr, 6 * dpr]);
      ctx.beginPath();
      ctx.ellipse(cx, cy, hrX, hrY, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // Portal Patch HUD + Outer Halo pills cleanly stacked above halo ring
    const patchCol = Math.floor((pf.x * Number(st.doc.width || 1344)) / 32);
    const patchRow = Math.floor((pf.y * Number(st.doc.height || 768)) / 32);
    const pStart = st.doc.portal_start != null ? st.doc.portal_start : 0;
    const pEnd = st.doc.portal_end != null ? st.doc.portal_end : st.length - 1;
    const topAnchorY = cy - Math.max(ryPx, hrY);
    this._drawPill(
      ctx,
      cx,
      Math.max(48 * dpr, topAnchorY - 52 * dpr),
      `Portal @ [${pf.x.toFixed(2)}, ${pf.y.toFixed(2)}] · 32px Patch [${patchCol}, ${patchRow}] · Span f${pStart}..f${pEnd}`,
      { border: "#BFDBFE", fg: "#0F172A", dot: "#2563EB" }
    );
    this._drawPill(
      ctx,
      cx,
      Math.max(74 * dpr, topAnchorY - 26 * dpr),
      `Outer Halo: r=${Number(pf.halo_radius || 0).toFixed(2)} · feather=${Number(st.doc.feather || 0.08).toFixed(2)}`,
      { border: "#BAE6FD", fg: "#0369A1", dot: "#0284C7" }
    );

    // Inner Solid Aperture Ring + 32px Patch Markers
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(cx, cy, rxPx, ryPx, 0, 0, Math.PI * 2);
    ctx.fillStyle = st.livePreview?.media ? "rgba(2, 132, 199, 0.04)" : "rgba(2, 132, 199, 0.12)";
    ctx.fill();

    ctx.save();
    ctx.clip();
    const scaleX = w / Math.max(64, Number(st.doc.width || 1344));
    const patchStep = Math.max(12, 32 * scaleX);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.28)";
    ctx.lineWidth = 1;
    const startX = Math.floor((cx - rxPx) / patchStep) * patchStep;
    const startY = Math.floor((cy - ryPx) / patchStep) * patchStep;
    ctx.beginPath();
    for (let x = startX; x <= cx + rxPx; x += patchStep) {
      ctx.moveTo(x, cy - ryPx);
      ctx.lineTo(x, cy + ryPx);
    }
    for (let y = startY; y <= cy + ryPx; y += patchStep) {
      ctx.moveTo(cx - rxPx, y);
      ctx.lineTo(cx + rxPx, y);
    }
    ctx.stroke();
    ctx.restore();

    ctx.strokeStyle = "#0284C7";
    ctx.lineWidth = 2.25 * dpr;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rxPx, ryPx, 0, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
    ctx.strokeStyle = "#2563EB";
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    ctx.arc(cx, cy, 11 * dpr, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.strokeStyle = "#2563EB";
    ctx.lineWidth = 1.6 * dpr;
    ctx.beginPath();
    ctx.moveTo(cx - 5 * dpr, cy);
    ctx.lineTo(cx + 5 * dpr, cy);
    ctx.moveTo(cx, cy - 5 * dpr);
    ctx.lineTo(cx, cy + 5 * dpr);
    ctx.stroke();

    ctx.fillStyle = "#FFFFFF";
    ctx.strokeStyle = "#2563EB";
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    ctx.arc(cx + rxPx, cy, 6.5 * dpr, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // Delicate 14px vertical dotted tether + External Rim Power & Preview Badge
    const rimBottomY = cy + Math.max(ryPx, hrY);
    const pillTopY = Math.min(h - 92 * dpr, rimBottomY + 14 * dpr);
    if (pillTopY - rimBottomY >= 6 * dpr) {
      ctx.save();
      ctx.strokeStyle = "rgba(37, 99, 235, 0.72)";
      ctx.lineWidth = 1.4 * dpr;
      ctx.setLineDash([2.5 * dpr, 3 * dpr]);
      ctx.beginPath();
      ctx.moveTo(cx, rimBottomY + 2 * dpr);
      ctx.lineTo(cx, pillTopY);
      ctx.stroke();
      ctx.restore();
    }
    const pwr = Number(st.editGuidance?.scale ?? 1.5).toFixed(1);
    const aud = Number(st.editGuidance?.audio_scale ?? 1.0).toFixed(1);
    const stepStr = st.livePreview?.total > 0 ? `Step ${st.livePreview.step}/${st.livePreview.total}` : `Mask ${pf.rx.toFixed(2)}`;
    const rimTxt = coupling.active
      ? `${stepStr} · Power ${pwr}x · Coupling ${Math.round(coupling.strength * 100)}%`
      : `${stepStr} · Power ${pwr}x · Audio ${aud}x`;
    this._drawPill(
      ctx,
      cx,
      pillTopY,
      rimTxt,
      { border: "#BFDBFE", fg: "#1D4ED8", dot: "#2563EB" }
    );
  }

  _drawTargetsOverlay(ctx, st, w, h, shortEdge, dpr) {
    const targets = st.doc.targets || [];
    for (let i = 0; i < targets.length; i++) {
      const t = targets[i];
      const tx = t.x * w;
      const ty = t.y * h;
      const rPx = t.radius * shortEdge;
      const haloPx = Number(t.halo_radius || 0) * shortEdge;
      const tw = evalTargetTemporalWeight(t, st.frame, st.length);
      const isSelected = i === (st.selectedTargetIdx ?? 0);
      const isActive = t.start <= st.frame && st.frame <= t.end;
      const isDone = st.frame > t.end;

      const strokeCol = isActive ? "#B45309" : (isDone ? "#047857" : "#64748B");
      const fillCol = isActive ? "rgba(180, 83, 9, 0.14)" : (isDone ? "rgba(4, 120, 87, 0.08)" : "rgba(100, 116, 139, 0.08)");

      ctx.save();
      if (haloPx > 2) {
        ctx.strokeStyle = "rgba(2, 132, 199, 0.45)";
        ctx.lineWidth = 1.2 * dpr;
        ctx.setLineDash([5 * dpr, 5 * dpr]);
        ctx.beginPath();
        ctx.arc(tx, ty, rPx + haloPx, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.fillStyle = fillCol;
      ctx.strokeStyle = isSelected ? "#2563EB" : strokeCol;
      ctx.lineWidth = (isSelected ? 2.5 : 1.75) * dpr;
      ctx.beginPath();
      ctx.arc(tx, ty, rPx, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = isSelected ? "#2563EB" : strokeCol;
      ctx.beginPath();
      ctx.arc(tx, ty, 5 * dpr, 0, Math.PI * 2);
      ctx.fill();

      const cueTxt = t.cue && t.cue !== "none" ? ` · ${t.cue}` : "";
      const stateTxt = isActive ? `ACTIVE ${(tw * 100).toFixed(0)}%` : (isDone ? "DONE" : `f${t.start}`);
      this._drawPill(ctx, tx, ty - rPx - 26 * dpr, `#${t.id} ${t.label}${cueTxt} (${stateTxt})`, {
        border: isSelected ? "#2563EB" : "#CBD5E1",
        fg: strokeCol,
        dot: strokeCol,
      });
      ctx.restore();
    }
  }

  _drawSpriteOverlay(ctx, st, cur, coupling, w, h, dpr) {
    const wps = st.doc.dpad?.waypoints || [];
    if (!wps.length || !cur) return;

    ctx.save();
    for (let dt = 24; dt >= 4; dt -= 5) {
      const fPast = st.frame - dt;
      if (fPast < 0) continue;
      const gp = evalSpriteAtFrame(st.doc, fPast, st.length);
      if (!gp) continue;
      const alpha = 0.12 + 0.28 * (1 - dt / 25);
      const gx = gp.x * w;
      const gy = gp.y * h;
      if (st.spriteCutout?.img) {
        const img = st.spriteCutout.img;
        const size = Math.min(w, h) * 0.22 * (gp.scale || 1.0);
        const aspect = img.naturalWidth / Math.max(1, img.naturalHeight);
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.translate(gx, gy);
        if (gp.face === "left") ctx.scale(-1, 1);
        ctx.drawImage(img, -size * aspect * 0.5, -size * 0.5, size * aspect, size);
        ctx.restore();
      } else {
        ctx.fillStyle = `rgba(4, 120, 87, ${alpha.toFixed(2)})`;
        ctx.beginPath();
        ctx.arc(gx, gy, 14 * dpr, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (wps.length >= 2) {
      ctx.strokeStyle = "rgba(4, 120, 87, 0.72)";
      ctx.lineWidth = 1.75 * dpr;
      ctx.setLineDash([3 * dpr, 3 * dpr]);
      ctx.beginPath();
      for (let f = 0; f < st.length; f += 3) {
        const sp = evalSpriteAtFrame(st.doc, f, st.length);
        if (!sp) continue;
        if (f === 0) ctx.moveTo(sp.x * w, sp.y * h);
        else ctx.lineTo(sp.x * w, sp.y * h);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const sx = cur.x * w;
    const sy = cur.y * h;
    if (st.spriteCutout?.img) {
      const img = st.spriteCutout.img;
      const size = Math.min(w, h) * 0.24 * (cur.scale || 1.0);
      const aspect = img.naturalWidth / Math.max(1, img.naturalHeight);
      ctx.save();
      ctx.translate(sx, sy);
      if (cur.face === "left") ctx.scale(-1, 1);
      ctx.drawImage(img, -size * aspect * 0.5, -size * 0.5, size * aspect, size);
      ctx.restore();
    } else {
      ctx.fillStyle = "rgba(4, 120, 87, 0.16)";
      ctx.strokeStyle = "#047857";
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      ctx.arc(sx, sy, 20 * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    const badgeTxt = coupling.active
      ? `Subject Sprite · clean 0.999 · Release 0.33 · ${coupling.mode.toUpperCase()}`
      : `Subject Sprite · ${cur.face.toUpperCase()}${cur.hopping ? " · HOP" : ""} · clean 0.999`;
    this._drawPill(ctx, sx, Math.min(h - 92 * dpr, sy + 28 * dpr), badgeTxt, {
      border: "#6EE7B7",
      fg: "#047857",
      dot: "#047857",
    });
    ctx.restore();
  }
}
