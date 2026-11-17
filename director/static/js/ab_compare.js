/**
 * Same-Seed A|B Split Wipe & Export Controller (`ab_compare.js`).
 *
 * Draws an interactive `◄ A | B ►` vertical split wipe directly onto the 7:4 stage
 * <canvas> (zero element.style mutations), syncs baseline/original vs directed media
 * by normalized loop phase, fetches real /api/ab_metrics localization ratios, and
 * triggers Clean / HUD MP4 exports via POST /api/export.
 */

export function videoItemViewUrl(item) {
  if (!item) return "";
  if (typeof item === "string") return item;
  if (item.view_url) return item.view_url;
  if (item.url) return item.url;
  if (item.filename) {
    return `/api/view?${new URLSearchParams({
      filename: item.filename,
      subfolder: item.subfolder || "",
      type: item.type || "temp",
    })}`;
  }
  return "";
}

export class ABCompareController {
  constructor({ onChange, onStatus } = {}) {
    this.active = false;
    this.splitX = 0.5;
    this.dragging = false;
    this.hudMode = false; // false = Clean (Node 162), true = HUD (Node 13)
    this.audioSource = "B"; // "B" | "A" | "mute"
    this.baselineTake = null;
    this.directedTake = null;
    this.metrics = null;
    this.metricsLoading = false;
    this.onChange = onChange || (() => {});
    this.onStatus = onStatus || (() => {});

    this._videoA = null;
    this._videoB = null;
    this._videoSink = null;
    this._srcA = "";
    this._srcB = "";
  }

  setVideoSink(el) {
    this._videoSink = el;
  }

  _ensureVideo(slot) {
    const key = slot === "A" ? "_videoA" : "_videoB";
    if (!this[key]) {
      const v = document.createElement("video");
      v.muted = true;
      v.playsInline = true;
      v.loop = true;
      v.preload = "auto";
      v.className = "offscreen-media";
      if (this._videoSink) {
        this._videoSink.appendChild(v);
      }
      this[key] = v;
    }
    return this[key];
  }

  setTakes({ baselineTake = null, directedTake = null } = {}) {
    this.baselineTake = baselineTake;
    this.directedTake = directedTake;
    this.metrics = null;
    this._syncSources();
    this.onChange();
  }

  _pickItem(take, preferHud) {
    if (!take) return null;
    if (preferHud && take.hud_video) return take.hud_video;
    if (take.clean_video) return take.clean_video;
    if (take.baseline_video) return take.baseline_video;
    if (take.hud_video) return take.hud_video;
    if (take.clean_url || take.video_url || take.url) return take;
    return null;
  }

  _pickBaselineItem() {
    if (this.baselineTake) {
      return this.baselineTake.baseline_video || this.baselineTake.clean_video || this._pickItem(this.baselineTake, false);
    }
    if (this.directedTake?.baseline_video) {
      return this.directedTake.baseline_video;
    }
    return null;
  }

  _syncSources() {
    const urlA = videoItemViewUrl(this._pickBaselineItem());
    const urlB = videoItemViewUrl(this._pickItem(this.directedTake, this.hudMode));
    if (urlA !== this._srcA) {
      this._srcA = urlA;
      if (urlA) {
        const va = this._ensureVideo("A");
        va.src = urlA;
        try { va.load(); } catch {}
      }
    }
    if (urlB !== this._srcB) {
      this._srcB = urlB;
      if (urlB) {
        const vb = this._ensureVideo("B");
        vb.src = urlB;
        try { vb.load(); } catch {}
      }
    }
  }

  toggle(forceState) {
    this.active = typeof forceState === "boolean" ? forceState : !this.active;
    this._syncSources();
    if (!this.active) {
      this.syncPlayback({ playing: false, stageMuted: true, loop: true });
    }
    this.onChange();
    return this.active;
  }

  cycleAudioSource() {
    const order = ["B", "A", "mute"];
    const idx = order.indexOf(this.audioSource);
    this.audioSource = order[(idx + 1) % order.length];
    this.onChange();
    return this.audioSource;
  }

  getAudioLabel() {
    if (this.audioSource === "A") return "Audio: A (Original)";
    if (this.audioSource === "mute") return "Audio: Muted";
    return "Audio: B (Directed)";
  }

  getMasterVideo() {
    if (!this.active) return null;
    if (this.audioSource === "A" && this._videoA && this._videoA.readyState >= 2) {
      return this._videoA;
    }
    if (this._videoB && this._videoB.readyState >= 2) return this._videoB;
    if (this._videoA && this._videoA.readyState >= 2) return this._videoA;
    return null;
  }

  getMasterProgress() {
    const v = this.getMasterVideo();
    if (!v || !Number.isFinite(v.duration) || v.duration <= 0) return null;
    return {
      progress: Math.max(0, Math.min(1, v.currentTime / v.duration)),
      ended: Boolean(v.ended),
      paused: Boolean(v.paused),
      audible: Boolean(!v.muted && !v.paused),
    };
  }

  syncPlayback({ playing = false, stageMuted = false, loop = true } = {}) {
    const va = this._videoA;
    const vb = this._videoB;
    const slot = !this.active || !playing || stageMuted ? "mute" : this.audioSource;
    if (va) {
      va.loop = Boolean(loop);
      va.muted = slot !== "A";
    }
    if (vb) {
      vb.loop = Boolean(loop);
      vb.muted = slot !== "B";
    }
    if (!this.active || !playing) {
      for (const v of [va, vb]) {
        if (v && !v.paused) {
          try { v.pause(); } catch {}
        }
      }
      return;
    }
    const master = this.getMasterVideo();
    const follower = master === vb ? va : vb;
    if (master && follower && follower.readyState >= 2 && Number.isFinite(master.duration) && master.duration > 0 && Number.isFinite(follower.duration) && follower.duration > 0) {
      const mFrac = master.currentTime / master.duration;
      const target = mFrac * follower.duration;
      if (Math.abs(follower.currentTime - target) > 0.08) {
        try { follower.currentTime = target; } catch {}
      }
    }
    for (const v of [va, vb]) {
      if (v && v.readyState >= 2 && v.paused) {
        v.play().catch(() => {
          v.muted = true;
          v.play().catch(() => {});
        });
      }
    }
  }

  setHudMode(enabled) {
    this.hudMode = Boolean(enabled);
    this._syncSources();
    this.onChange();
  }

  hitTestHandle(nx, ny, canvasW, canvasH) {
    if (!this.active) return false;
    const dxPx = (nx - this.splitX) * canvasW;
    return Math.abs(dxPx) <= 22;
  }

  setSplit(nx) {
    this.splitX = Math.max(0.05, Math.min(0.95, Number(nx)));
    this.onChange();
  }

  syncTime(frame, length) {
    const frac = Math.max(0, Math.min(1, Number(frame || 0) / Math.max(1, Number(length || 141) - 1)));
    for (const v of [this._videoA, this._videoB]) {
      if (v && v.readyState >= 2 && Number.isFinite(v.duration) && v.duration > 0) {
        const target = frac * v.duration;
        if (Math.abs(v.currentTime - target) > 0.025) {
          try {
            v.currentTime = target;
          } catch {}
        }
      }
    }
  }

  getMediaA(fallbackImgOrVideo) {
    if (this._srcA && this._videoA && this._videoA.readyState >= 2) {
      return this._videoA;
    }
    return fallbackImgOrVideo || null;
  }

  getMediaB(fallbackImgOrVideo) {
    if (this._srcB && this._videoB && this._videoB.readyState >= 2) {
      return this._videoB;
    }
    return fallbackImgOrVideo || null;
  }

  drawSplitOverlay(ctx, w, h, mediaA, mediaB) {
    const splitPx = Math.round(this.splitX * w);

    // Draw Left (A: Original / Baseline)
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, splitPx, h);
    ctx.clip();
    if (mediaA) {
      this._drawContainOrCover(ctx, mediaA, w, h);
    } else {
      ctx.fillStyle = "#EEF2F6";
      ctx.fillRect(0, 0, splitPx, h);
    }
    ctx.restore();

    // Draw Right (B: Directed)
    ctx.save();
    ctx.beginPath();
    ctx.rect(splitPx, 0, w - splitPx, h);
    ctx.clip();
    if (mediaB) {
      this._drawContainOrCover(ctx, mediaB, w, h);
    } else {
      ctx.fillStyle = "#EEF2F6";
      ctx.fillRect(splitPx, 0, w - splitPx, h);
    }
    ctx.restore();

    // Crisp vertical divider + draggable ◄ A | B ► handle
    ctx.save();
    ctx.strokeStyle = "rgba(15, 23, 42, 0.35)";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(splitPx, 0);
    ctx.lineTo(splitPx, h);
    ctx.stroke();

    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(splitPx, 0);
    ctx.lineTo(splitPx, h);
    ctx.stroke();

    const cy = Math.round(h * 0.5);
    const pillW = 76;
    const pillH = 28;
    const px = Math.max(8, Math.min(w - pillW - 8, splitPx - pillW / 2));
    const py = cy - pillH / 2;

    ctx.fillStyle = "rgba(255, 255, 255, 0.96)";
    ctx.strokeStyle = "#2563EB";
    ctx.lineWidth = 1.75;
    this._roundRect(ctx, px, py, pillW, pillH, 14);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#0F172A";
    ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("◄ A | B ►", px + pillW / 2, cy + 0.5);

    const hasBaselineVideo = Boolean(this._pickBaselineItem());
    this._drawSideTag(ctx, 14, 48, hasBaselineVideo ? "A · BASELINE" : "A · ANCHOR", "#475569");
    this._drawSideTag(ctx, w - 128, 48, this.hudMode ? "B · HUD (N13)" : "B · CLEAN (N162)", "#1D4ED8");
    ctx.restore();
  }

  _drawSideTag(ctx, x, y, label, color) {
    ctx.save();
    ctx.font = "600 10px ui-monospace, SFMono-Regular, Menlo, monospace";
    const metrics = ctx.measureText(label);
    const pw = Math.ceil(metrics.width) + 16;
    const ph = 22;
    ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
    ctx.strokeStyle = "#CBD5E1";
    ctx.lineWidth = 1;
    this._roundRect(ctx, x, y, pw, ph, 6);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(label, x + 8, y + ph / 2 + 0.5);
    ctx.restore();
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

  async fetchMetrics({ events_json, director_mode = "portal_path", feather = 0.08, width, height } = {}) {
    const editedItem = this._pickItem(this.directedTake, false);
    const baselineItem = this._pickBaselineItem();
    if (!editedItem || !baselineItem) {
      this.metrics = { ok: true, available: false, reason: "Requires both directed & baseline takes" };
      this.onChange();
      return this.metrics;
    }
    this.metricsLoading = true;
    this.onChange();
    try {
      const resp = await fetch("/api/ab_metrics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          edited_video: editedItem,
          baseline_video: baselineItem,
          events_json,
          director_mode,
          feather,
          width,
          height,
        }),
      });
      const data = await resp.json();
      this.metrics = resp.ok ? data : { ok: false, available: false, error: data.error || "Metrics failed" };
      return this.metrics;
    } catch (err) {
      this.metrics = { ok: false, available: false, error: err.message || "Network error" };
      return this.metrics;
    } finally {
      this.metricsLoading = false;
      this.onChange();
    }
  }

  async exportTake(variant = "clean") {
    const take = this.directedTake || this.baselineTake;
    const item = variant === "hud" ? (take?.hud_video || take?.clean_video) : (take?.clean_video || take?.hud_video);
    if (!item || !item.filename) {
      this.onStatus("No rendered take available to export yet.", "warn");
      return null;
    }
    try {
      const resp = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          take_id: take?.take_id || undefined,
          variant,
          filename: item.filename,
          subfolder: item.subfolder || "",
          type: item.type || "temp",
        }),
      });
      const data = await resp.json();
      if (!resp.ok || data.ok === false) {
        this.onStatus(data.error || "Export failed", "err");
        return null;
      }
      const sidecarNote = data.recipe_filename ? ` + ${data.recipe_filename}` : "";
      this.onStatus(`Exported ${variant.toUpperCase()} MP4 -> Video/${data.filename || ""}${sidecarNote}`, "ok");
      return data;
    } catch (err) {
      this.onStatus(`Export error: ${err.message}`, "err");
      return null;
    }
  }
}
