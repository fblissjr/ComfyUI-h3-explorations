/**
 * Interactive 3-Lane Keyframe Sequencer (`sequencer.js`).
 *
 * Renders and edits Portal keyframes/spans, World Target spans + 3f SoundCues,
 * and Sprite D-Pad waypoints + arbitrary multi-hop arcs along H3's [1, 4, 4, 4, 4]
 * 17-frame causal clips (`on_both_clocks`).
 *
 * Pure Canvas2D + pointer events; zero external deps, zero `element.style.*`.
 */

import { snapFrameTo3f } from "./foley_synth.js";
import { clamp } from "./stage.js";

const CLIP_OFFSETS = [0, 1, 5, 9, 13];

/** Return `[clipStarts, stepStarts]` frame indices across H3's `[1, 4, 4, 4, 4]` causal clips. */
export function causalStepBoundaries(length) {
  const len = Math.max(5, Math.round(Number(length || 90)));
  const clipStarts = [];
  const stepStarts = [];
  for (let clip0 = 0; clip0 < len; clip0 += 17) {
    clipStarts.push(clip0);
    for (const off of CLIP_OFFSETS) {
      const f = clip0 + off;
      if (f < len) stepStarts.push(f);
    }
  }
  return { clipStarts, stepStarts };
}

/** Snap `frame` to nearest `[1, 4, 4, 4, 4]` causal step boundary inside `[0, length - 1]`. */
export function snapFrameToCausalStep(frame, length) {
  const { stepStarts } = causalStepBoundaries(length);
  const target = clamp(Math.round(Number(frame || 0)), 0, Math.max(0, Number(length || 90) - 1));
  let best = stepStarts[0] ?? 0;
  let bestDist = Infinity;
  for (const s of stepStarts) {
    const d = Math.abs(s - target);
    if (d < bestDist || (d === bestDist && s < best)) {
      bestDist = d;
      best = s;
    }
  }
  return best;
}

export class SequencerController {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas ? canvas.getContext("2d") : null;
    this.getState = opts.getState;
    this.onMutate = opts.onMutate || (() => {});
    this.onSelect = opts.onSelect || (() => {});
    this.onScrub = opts.onScrub || (() => {});

    this._padLeft = 74;
    this._padRight = 14;
    this._drag = null;

    this._bindEvents();
    this.resize();
  }

  resize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cw = Math.max(240, Math.round((rect.width || 900) * dpr));
    const ch = Math.max(72, Math.round((rect.height || 86) * dpr));
    if (this.canvas.width !== cw || this.canvas.height !== ch) {
      this.canvas.width = cw;
      this.canvas.height = ch;
    }
    this.render();
  }

  _metrics() {
    const rect = this.canvas.getBoundingClientRect();
    const cssW = Math.max(240, rect.width || 900);
    const cssH = Math.max(72, rect.height || 86);
    const trackW = Math.max(40, cssW - this._padLeft - this._padRight);
    const laneH = (cssH - 8) / 3;
    return { rect, cssW, cssH, trackW, laneH };
  }

  _frameToCssX(frame, maxF, m) {
    const u = clamp(Number(frame || 0), 0, maxF) / Math.max(1, maxF);
    return this._padLeft + u * m.trackW;
  }

  _cssXToFrame(cssX, maxF, m) {
    const u = clamp((cssX - this._padLeft) / Math.max(1, m.trackW), 0, 1);
    return clamp(Math.round(u * maxF), 0, maxF);
  }

  _bindEvents() {
    if (!this.canvas) return;
    this.canvas.addEventListener("pointerdown", (ev) => this._onPointerDown(ev));
    window.addEventListener("pointermove", (ev) => this._onPointerMove(ev));
    window.addEventListener("pointerup", () => this._onPointerUp());
  }

  _hitTest(cssX, cssY, st, m) {
    const len = Math.max(5, Number(st.length || 90));
    const maxF = Math.max(1, len - 1);
    const laneIdx = Math.floor((cssY - 4) / Math.max(12, m.laneH));

    if (laneIdx === 0) {
      const portals = st.doc.portal || [];
      for (let i = portals.length - 1; i >= 0; i--) {
        const kx = this._frameToCssX(portals[i].frame, maxF, m);
        if (Math.abs(cssX - kx) <= 9) {
          return { kind: "portal_kf", part: "move", idx: i, ref: portals[i] };
        }
      }
      const pStart = st.doc.portal_start != null ? Number(st.doc.portal_start) : 0;
      const pEnd = st.doc.portal_end != null ? Number(st.doc.portal_end) : maxF;
      const x0 = this._frameToCssX(pStart, maxF, m);
      const x1 = this._frameToCssX(pEnd, maxF, m);
      if (Math.abs(cssX - x0) <= 8) return { kind: "portal_span", part: "start", idx: 0 };
      if (Math.abs(cssX - x1) <= 8) return { kind: "portal_span", part: "end", idx: 0 };
      if (cssX > x0 + 6 && cssX < x1 - 6) return { kind: "portal_span", part: "select", idx: 0 };
      return null;
    }

    if (laneIdx === 1) {
      const cues = st.doc.portal_cues || [];
      for (let i = cues.length - 1; i >= 0; i--) {
        const cx = this._frameToCssX(cues[i].frame, maxF, m);
        if (Math.abs(cssX - cx) <= 7) {
          return { kind: "cue", part: "move", idx: i, ref: cues[i] };
        }
      }
      const targets = st.doc.targets || [];
      for (let i = targets.length - 1; i >= 0; i--) {
        const t = targets[i];
        const tx0 = this._frameToCssX(t.start ?? 0, maxF, m);
        const tx1 = this._frameToCssX(t.end ?? maxF, maxF, m);
        if (Math.abs(cssX - tx0) <= 8) return { kind: "target", part: "start", idx: i, ref: t };
        if (Math.abs(cssX - tx1) <= 8) return { kind: "target", part: "end", idx: i, ref: t };
        if (cssX >= tx0 && cssX <= tx1) return { kind: "target", part: "body", idx: i, ref: t };
      }
      return null;
    }

    if (laneIdx === 2) {
      const wps = st.doc.dpad?.waypoints || [];
      for (let i = wps.length - 1; i >= 0; i--) {
        const wx = this._frameToCssX(wps[i].frame, maxF, m);
        if (Math.abs(cssX - wx) <= 8) {
          return { kind: "waypoint", part: "move", idx: i, ref: wps[i] };
        }
      }
      const hops = st.doc.dpad?.hops || [];
      for (let i = hops.length - 1; i >= 0; i--) {
        const hp = hops[i];
        const hx0 = this._frameToCssX(hp.start ?? 0, maxF, m);
        const hx1 = this._frameToCssX(hp.end ?? maxF, maxF, m);
        if (Math.abs(cssX - hx0) <= 8) return { kind: "hop_arc", part: "start", idx: i, ref: hp };
        if (Math.abs(cssX - hx1) <= 8) return { kind: "hop_arc", part: "end", idx: i, ref: hp };
        if (cssX >= hx0 && cssX <= hx1) return { kind: "hop_arc", part: "body", idx: i, ref: hp };
      }
    }
    return null;
  }

  _onPointerDown(ev) {
    const st = this.getState?.();
    if (!st || !this.canvas) return;
    const m = this._metrics();
    const cssX = ev.clientX - m.rect.left;
    const cssY = ev.clientY - m.rect.top;
    const maxF = Math.max(1, Number(st.length || 90) - 1);
    const clickFrame = this._cssXToFrame(cssX, maxF, m);

    const hit = this._hitTest(cssX, cssY, st, m);
    if (!hit) {
      st.seqSelection = null;
      st.frame = clickFrame;
      this._drag = { kind: "scrub" };
      this.canvas.setPointerCapture?.(ev.pointerId);
      this.onScrub(clickFrame);
      this.onSelect(null);
      this.render();
      return;
    }

    st.seqSelection = { kind: hit.kind, idx: hit.idx };
    if (hit.kind === "portal_kf" && hit.ref) {
      st.frame = clamp(Math.round(hit.ref.frame), 0, maxF);
    } else if (hit.kind === "target" && hit.ref) {
      st.selectedTargetIdx = hit.idx;
      st.frame = clamp(Math.round(hit.ref.start ?? 0), 0, maxF);
    } else if (hit.kind === "cue" && hit.ref) {
      st.frame = clamp(Math.round(hit.ref.frame), 0, maxF);
    } else if (hit.kind === "waypoint" && hit.ref) {
      st.frame = clamp(Math.round(hit.ref.frame), 0, maxF);
    } else if (hit.kind === "hop_arc" && hit.ref) {
      st.frame = clamp(Math.round(hit.ref.start ?? 0), 0, maxF);
    } else if (hit.kind === "portal_span") {
      st.frame = hit.part === "end"
        ? Number(st.doc.portal_end ?? maxF)
        : (hit.part === "start" ? Number(st.doc.portal_start ?? 0) : clickFrame);
    }

    this._drag = {
      ...hit,
      anchorFrame: clickFrame,
      origStart: hit.ref ? Number(hit.ref.start ?? 0) : 0,
      origEnd: hit.ref ? Number(hit.ref.end ?? maxF) : maxF,
      pushedUndo: false,
    };
    this.canvas.setPointerCapture?.(ev.pointerId);
    this.onSelect(st.seqSelection);
    this.render();
  }

  _onPointerMove(ev) {
    if (!this._drag) return;
    const st = this.getState?.();
    if (!st || !this.canvas) return;
    const m = this._metrics();
    const len = Math.max(5, Number(st.length || 90));
    const maxF = Math.max(1, len - 1);
    const rawF = this._cssXToFrame(ev.clientX - m.rect.left, maxF, m);
    const f = ev.shiftKey ? snapFrameToCausalStep(rawF, len) : rawF;

    if (this._drag.kind === "scrub") {
      st.frame = rawF;
      this.onScrub(rawF);
      this.render();
      return;
    }

    if (this._drag.part === "select") {
      return;
    }

    const ensureUndo = () => {
      if (!this._drag.pushedUndo) {
        this._drag.pushedUndo = true;
        this.onMutate("seq_drag_start", { live: false, beforeEdit: true });
      }
    };

    if (this._drag.kind === "portal_kf" && this._drag.ref) {
      ensureUndo();
      this._drag.ref.frame = f;
      st.frame = f;
      st.doc.portal.sort((a, b) => a.frame - b.frame);
      const nextIdx = st.doc.portal.indexOf(this._drag.ref);
      this._drag.idx = nextIdx;
      st.seqSelection = { kind: "portal_kf", idx: nextIdx };
    } else if (this._drag.kind === "portal_span") {
      ensureUndo();
      const curStart = st.doc.portal_start != null ? Number(st.doc.portal_start) : 0;
      const curEnd = st.doc.portal_end != null ? Number(st.doc.portal_end) : maxF;
      if (this._drag.part === "start") {
        const nextS = clamp(f, 0, Math.max(0, curEnd - 2));
        st.doc.portal_start = nextS;
        if (st.doc.portal_end == null) st.doc.portal_end = maxF;
        st.frame = nextS;
      } else if (this._drag.part === "end") {
        const nextE = clamp(f, Math.min(maxF, curStart + 2), maxF);
        st.doc.portal_end = nextE;
        if (st.doc.portal_start == null) st.doc.portal_start = 0;
        st.frame = nextE;
      }
    } else if (this._drag.kind === "target" && this._drag.ref) {
      ensureUndo();
      const t = this._drag.ref;
      const hasCue = Boolean(t.cue && t.cue !== "none");
      const minSpan = hasCue ? 3 : 2;
      if (this._drag.part === "start") {
        let nextS = hasCue ? snapFrameTo3f(f, len) : f;
        nextS = clamp(nextS, 0, Math.max(0, Number(t.end) - minSpan));
        t.start = nextS;
        st.frame = nextS;
      } else if (this._drag.part === "end") {
        t.end = clamp(f, Math.min(maxF, Number(t.start) + minSpan), maxF);
        st.frame = t.end;
      } else if (this._drag.part === "body") {
        const span = Math.max(minSpan, this._drag.origEnd - this._drag.origStart);
        let ns = clamp(this._drag.origStart + (rawF - this._drag.anchorFrame), 0, Math.max(0, maxF - span));
        if (hasCue) ns = clamp(snapFrameTo3f(ns, len), 0, Math.max(0, maxF - span));
        t.start = ns;
        t.end = Math.min(maxF, ns + span);
        st.frame = ns;
      }
    } else if (this._drag.kind === "cue" && this._drag.ref) {
      ensureUndo();
      const cf = snapFrameTo3f(rawF, len);
      this._drag.ref.frame = cf;
      st.frame = cf;
      st.doc.portal_cues.sort((a, b) => a.frame - b.frame);
      const nextIdx = st.doc.portal_cues.indexOf(this._drag.ref);
      this._drag.idx = nextIdx;
      st.seqSelection = { kind: "cue", idx: nextIdx };
    } else if (this._drag.kind === "waypoint" && this._drag.ref) {
      ensureUndo();
      this._drag.ref.frame = f;
      st.frame = f;
      st.doc.dpad.waypoints.sort((a, b) => a.frame - b.frame);
      const nextIdx = st.doc.dpad.waypoints.indexOf(this._drag.ref);
      this._drag.idx = nextIdx;
      st.seqSelection = { kind: "waypoint", idx: nextIdx };
    } else if (this._drag.kind === "hop_arc" && this._drag.ref) {
      ensureUndo();
      const hp = this._drag.ref;
      if (this._drag.part === "start") {
        hp.start = clamp(f, 0, Math.max(0, Number(hp.end) - 2));
        st.frame = hp.start;
      } else if (this._drag.part === "end") {
        hp.end = clamp(f, Math.min(maxF, Number(hp.start) + 2), maxF);
        st.frame = hp.end;
      } else if (this._drag.part === "body") {
        const span = Math.max(2, this._drag.origEnd - this._drag.origStart);
        const ns = clamp(this._drag.origStart + (rawF - this._drag.anchorFrame), 0, Math.max(0, maxF - span));
        hp.start = ns;
        hp.end = Math.min(maxF, ns + span);
        st.frame = ns;
      }
    }

    this.onMutate(`seq_drag_${this._drag.kind}`, { live: true });
    this.render();
  }

  _onPointerUp() {
    if (!this._drag) return;
    const drag = this._drag;
    this._drag = null;
    if (drag.kind !== "scrub" && drag.pushedUndo) {
      this.onMutate(`seq_commit_${drag.kind}`, { live: false });
    }
  }

  deleteSelected(st) {
    const sel = st?.seqSelection;
    if (!sel) return { ok: false, reason: "Select a keyframe, target, cue, or hop first" };
    const { kind, idx } = sel;

    if (kind === "portal_kf") {
      const portals = st.doc.portal || [];
      if (idx < 0 || idx >= portals.length) return { ok: false, reason: "Invalid portal keyframe" };
      const hasTargets = Boolean((st.doc.targets || []).length);
      if (portals.length <= 1 && (st.doc.mode === "portal_path" || !hasTargets)) {
        return { ok: false, reason: "Portal mode requires at least 1 keyframe" };
      }
      portals.splice(idx, 1);
      if (!portals.length && st.doc.mode === "combined" && hasTargets) {
        st.doc.mode = "point_targets";
      }
    } else if (kind === "portal_span") {
      st.doc.portal_start = null;
      st.doc.portal_end = null;
    } else if (kind === "target") {
      const targets = st.doc.targets || [];
      if (idx < 0 || idx >= targets.length) return { ok: false, reason: "Invalid target" };
      const hasPortals = Boolean((st.doc.portal || []).length);
      if (targets.length <= 1 && !hasPortals) {
        return { ok: false, reason: "Requires at least 1 portal or target" };
      }
      targets.splice(idx, 1);
      if (!targets.length && st.doc.mode === "point_targets" && hasPortals) {
        st.doc.mode = "portal_path";
      }
      st.selectedTargetIdx = clamp(st.selectedTargetIdx ?? 0, 0, Math.max(0, targets.length - 1));
    } else if (kind === "cue") {
      const cues = st.doc.portal_cues || [];
      if (idx < 0 || idx >= cues.length) return { ok: false, reason: "Invalid cue" };
      cues.splice(idx, 1);
    } else if (kind === "waypoint") {
      const wps = st.doc.dpad?.waypoints || [];
      if (idx < 0 || idx >= wps.length) return { ok: false, reason: "Invalid waypoint" };
      wps.splice(idx, 1);
    } else if (kind === "hop_arc") {
      const hops = st.doc.dpad?.hops || [];
      if (idx < 0 || idx >= hops.length) return { ok: false, reason: "Invalid hop" };
      hops.splice(idx, 1);
    } else {
      return { ok: false, reason: "Unsupported selection" };
    }

    st.seqSelection = null;
    this.render();
    return { ok: true, kind };
  }

  describeSelection(st) {
    const sel = st?.seqSelection;
    if (!sel) {
      return "3 Lanes · 17n+5 / 4f Causal Grid (Shift=Snap, Del=Remove)";
    }
    const { kind, idx } = sel;
    if (kind === "portal_kf") {
      const p = st.doc.portal?.[idx];
      return p ? `Portal KF #${idx + 1} @ f${p.frame} (rx=${Number(p.rx).toFixed(2)}, ry=${Number(p.ry).toFixed(2)})` : "Portal KF";
    }
    if (kind === "portal_span") {
      const maxF = Math.max(0, Number(st.length || 90) - 1);
      const s = st.doc.portal_start != null ? st.doc.portal_start : 0;
      const e = st.doc.portal_end != null ? st.doc.portal_end : maxF;
      return `Portal Window f${s}..f${e}`;
    }
    if (kind === "target") {
      const t = st.doc.targets?.[idx];
      return t ? `Target #${t.id || idx + 1} f${t.start}..f${t.end} (${t.cue || "none"})` : "Target Span";
    }
    if (kind === "cue") {
      const c = st.doc.portal_cues?.[idx];
      return c ? `3f Foley Cue @ f${c.frame}: ${c.preset}` : "3f Cue";
    }
    if (kind === "waypoint") {
      const w = st.doc.dpad?.waypoints?.[idx];
      return w ? `Sprite WP #${idx + 1} @ f${w.frame} (${w.face || "right"})` : "Waypoint";
    }
    if (kind === "hop_arc") {
      const h = st.doc.dpad?.hops?.[idx];
      return h ? `Sprite Hop #${idx + 1} f${h.start}..f${h.end} (lift=${Number(h.lift ?? 0.25).toFixed(2)})` : "Hop Arc";
    }
    return "Selected";
  }

  render() {
    const st = this.getState?.();
    if (!st || !this.canvas || !this.ctx) return;
    const ctx = this.ctx;
    const m = this._metrics();
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const dpr = cw / Math.max(1, m.cssW);
    const len = Math.max(5, Number(st.length || 90));
    const maxF = Math.max(1, len - 1);
    const sel = st.seqSelection;

    ctx.clearRect(0, 0, cw, ch);

    const toX = (f) => this._frameToCssX(f, maxF, m) * dpr;
    const laneTop = (i) => (4 + i * m.laneH) * dpr;
    const lanePxH = m.laneH * dpr;
    const laneMid = (i) => laneTop(i) + lanePxH * 0.5;
    const labels = ["Portal", "Targets+3f", "Sprite/Hop"];
    const labelColors = ["#1D4ED8", "#B45309", "#047857"];

    // Lane backgrounds & labels
    ctx.font = `600 ${Math.round(10 * dpr)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.textBaseline = "middle";
    for (let i = 0; i < 3; i++) {
      const y = laneTop(i);
      ctx.fillStyle = i % 2 === 0 ? "#F8FAFC" : "#F1F5F9";
      ctx.fillRect(0, y + 1 * dpr, cw, lanePxH - 2 * dpr);
      ctx.fillStyle = labelColors[i];
      ctx.textAlign = "left";
      ctx.fillText(labels[i], 6 * dpr, laneMid(i));
    }

    // 17n+5 Causal clip separators + 4f step ticks
    const { clipStarts, stepStarts } = causalStepBoundaries(len);
    ctx.save();
    ctx.strokeStyle = "rgba(100, 116, 139, 0.16)";
    ctx.lineWidth = 1;
    for (const s of stepStarts) {
      const sx = toX(s);
      ctx.beginPath();
      ctx.moveTo(sx, 4 * dpr);
      ctx.lineTo(sx, 8 * dpr);
      ctx.stroke();
    }
    ctx.strokeStyle = "rgba(37, 99, 235, 0.22)";
    ctx.setLineDash([2 * dpr, 2 * dpr]);
    for (const c of clipStarts) {
      if (c === 0) continue;
      const cx = toX(c);
      ctx.beginPath();
      ctx.moveTo(cx, 4 * dpr);
      ctx.lineTo(cx, ch - 4 * dpr);
      ctx.stroke();
    }
    ctx.restore();

    const drawRound = (x, y, w, h, r, fill, stroke = null, lw = 1) => {
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
      else ctx.rect(x, y, w, h);
      if (fill) {
        ctx.fillStyle = fill;
        ctx.fill();
      }
      if (stroke) {
        ctx.strokeStyle = stroke;
        ctx.lineWidth = lw * dpr;
        ctx.stroke();
      }
    };

    // Lane 0: Portal span + Keyframe diamonds
    const pStart = st.doc.portal_start != null ? Number(st.doc.portal_start) : 0;
    const pEnd = st.doc.portal_end != null ? Number(st.doc.portal_end) : maxF;
    const px0 = toX(pStart);
    const px1 = toX(pEnd);
    const spanSel = sel?.kind === "portal_span";
    drawRound(
      px0,
      laneMid(0) - 6 * dpr,
      Math.max(8 * dpr, px1 - px0),
      12 * dpr,
      5 * dpr,
      "rgba(37, 99, 235, 0.16)",
      spanSel ? "#1D4ED8" : "rgba(37, 99, 235, 0.45)",
      spanSel ? 1.8 : 1
    );
    for (const hx of [px0, px1]) {
      ctx.fillStyle = "#1D4ED8";
      ctx.fillRect(hx - 1.5 * dpr, laneMid(0) - 7 * dpr, 3 * dpr, 14 * dpr);
    }
    const portals = st.doc.portal || [];
    for (let i = 0; i < portals.length; i++) {
      const kx = toX(portals[i].frame);
      const ky = laneMid(0);
      const isSel = sel?.kind === "portal_kf" && sel.idx === i;
      const r = (isSel ? 6.5 : 5) * dpr;
      ctx.save();
      ctx.fillStyle = isSel ? "#1D4ED8" : "#2563EB";
      ctx.strokeStyle = "#FFFFFF";
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath();
      ctx.moveTo(kx, ky - r);
      ctx.lineTo(kx + r, ky);
      ctx.lineTo(kx, ky + r);
      ctx.lineTo(kx - r, ky);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // Lane 1: World Target spans + 3f SoundCue pins
    const targets = st.doc.targets || [];
    for (let i = 0; i < targets.length; i++) {
      const t = targets[i];
      const tx0 = toX(t.start ?? 0);
      const tx1 = toX(t.end ?? maxF);
      const isSel = sel?.kind === "target" && sel.idx === i;
      drawRound(
        tx0,
        laneMid(1) - 7 * dpr,
        Math.max(10 * dpr, tx1 - tx0),
        14 * dpr,
        4 * dpr,
        isSel ? "rgba(217, 119, 6, 0.34)" : "rgba(217, 119, 6, 0.20)",
        isSel ? "#2563EB" : "#B45309",
        isSel ? 1.8 : 1.1
      );
      ctx.fillStyle = "#92400E";
      ctx.fillRect(tx0 - 1.5 * dpr, laneMid(1) - 7 * dpr, 3 * dpr, 14 * dpr);
      ctx.fillRect(tx1 - 1.5 * dpr, laneMid(1) - 7 * dpr, 3 * dpr, 14 * dpr);
    }
    const cues = st.doc.portal_cues || [];
    for (let i = 0; i < cues.length; i++) {
      const cx = toX(cues[i].frame);
      const isSel = sel?.kind === "cue" && sel.idx === i;
      drawRound(
        cx - 2 * dpr,
        laneTop(1) + 2 * dpr,
        4 * dpr,
        lanePxH - 4 * dpr,
        2 * dpr,
        isSel ? "#065F46" : "#047857",
        isSel ? "#FFFFFF" : null,
        1.2
      );
    }

    // Lane 2: Sprite D-Pad waypoints + Multi-Hop arcs
    const hops = st.doc.dpad?.hops || [];
    for (let i = 0; i < hops.length; i++) {
      const hp = hops[i];
      const hx0 = toX(hp.start ?? 0);
      const hx1 = toX(hp.end ?? maxF);
      const isSel = sel?.kind === "hop_arc" && sel.idx === i;
      drawRound(
        hx0,
        laneMid(2) - 6 * dpr,
        Math.max(10 * dpr, hx1 - hx0),
        12 * dpr,
        6 * dpr,
        isSel ? "rgba(4, 120, 87, 0.32)" : "rgba(4, 120, 87, 0.18)",
        isSel ? "#2563EB" : "#047857",
        isSel ? 1.8 : 1.1
      );
    }
    const wps = st.doc.dpad?.waypoints || [];
    for (let i = 0; i < wps.length; i++) {
      const wx = toX(wps[i].frame);
      const wy = laneMid(2);
      const isSel = sel?.kind === "waypoint" && sel.idx === i;
      ctx.fillStyle = isSel ? "#2563EB" : "#047857";
      ctx.strokeStyle = "#FFFFFF";
      ctx.lineWidth = 1.4 * dpr;
      ctx.beginPath();
      ctx.arc(wx, wy, (isSel ? 5 : 4) * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    // Playhead line
    const headX = toX(st.frame);
    ctx.strokeStyle = "#2563EB";
    ctx.lineWidth = 1.8 * dpr;
    ctx.beginPath();
    ctx.moveTo(headX, 2 * dpr);
    ctx.lineTo(headX, ch - 2 * dpr);
    ctx.stroke();
  }
}
