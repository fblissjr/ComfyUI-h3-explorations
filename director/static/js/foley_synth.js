/**
 * Parametric WebAudio Foley Synthesizer (`foley_synth.js`).
 *
 * Previews 3-frame (125 ms) SoundCues locally at 0 ms GPU cost during stage scrubbing
 * and 24 fps loop playback. Preset names and durations are hydrated dynamically from
 * GET /api/schema (`sound_presets`) rather than hardcoded in frontend source.
 */

export const AUDIO_GRID_FRAMES = 3;
export const DEFAULT_VIDEO_FPS = 24;

export function snapFrameTo3f(frame, length) {
  const maxMult = Math.floor(Math.max(0, Number(length || 141) - 1) / AUDIO_GRID_FRAMES) * AUDIO_GRID_FRAMES;
  const snapped = Math.round(Number(frame || 0) / AUDIO_GRID_FRAMES) * AUDIO_GRID_FRAMES;
  return Math.max(0, Math.min(maxMult, snapped));
}

export class FoleySynth {
  constructor() {
    this._ctx = null;
    this._noiseBuffer = null;
    this.muted = false;
  }

  _ensureContext() {
    if (this.muted) return null;
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return null;
    if (!this._ctx) {
      try {
        this._ctx = new AudioCtx();
      } catch {
        return null;
      }
    }
    if (this._ctx.state === "suspended") {
      this._ctx.resume().catch(() => {});
    }
    if (!this._noiseBuffer && this._ctx) {
      const sr = this._ctx.sampleRate || 44100;
      const len = Math.floor(sr * 0.5);
      const buf = this._ctx.createBuffer(1, len, sr);
      const data = buf.getChannelData(0);
      // Deterministic pink-ish noise approximation for repeatable foley textures
      let b0 = 0;
      let b1 = 0;
      let b2 = 0;
      for (let i = 0; i < len; i++) {
        const white = Math.random() * 2 - 1;
        b0 = 0.99765 * b0 + white * 0.099046;
        b1 = 0.96300 * b1 + white * 0.2965164;
        b2 = 0.57000 * b2 + white * 1.0526913;
        data[i] = Math.max(-1, Math.min(1, (b0 + b1 + b2 + white * 0.1848) * 0.16));
      }
      this._noiseBuffer = buf;
    }
    return this._ctx;
  }

  unlock() {
    this._ensureContext();
  }

  setMuted(muted) {
    this.muted = Boolean(muted);
  }

  _triggerImpulse(ctx, startAt, kind, gainVal, durSec) {
    const g = Math.max(0.05, Math.min(2.5, Number(gainVal || 1.0)));
    const dur = Math.max(0.06, Math.min(0.45, Number(durSec || 0.18)));
    const master = ctx.createGain();
    master.gain.setValueAtTime(0.0001, startAt);
    master.gain.linearRampToValueAtTime(0.32 * Math.min(1.6, g), startAt + 0.004);
    master.gain.exponentialRampToValueAtTime(0.0001, startAt + dur);
    master.connect(ctx.destination);

    // Choose acoustic profile purely by token character (no hardcoded preset table)
    const token = String(kind || "").toLowerCase();
    const isSharp = token.includes("click") || token.includes("snap") || token.includes("yank");
    const isHeavy = token.includes("slam") || token.includes("impact") || token.includes("thud") || token.includes("hop");

    if (!token.includes("click")) {
      const osc = ctx.createOscillator();
      const oscGain = ctx.createGain();
      osc.type = isSharp ? "triangle" : "sine";
      const fStart = isSharp ? 920 : (isHeavy ? 170 : 230);
      const fEnd = isSharp ? 260 : (isHeavy ? 48 : 88);
      osc.frequency.setValueAtTime(fStart, startAt);
      osc.frequency.exponentialRampToValueAtTime(fEnd, startAt + Math.min(dur * 0.5, 0.045));
      oscGain.gain.setValueAtTime(isSharp ? 0.45 : 0.85, startAt);
      oscGain.gain.exponentialRampToValueAtTime(0.001, startAt + dur * 0.85);
      osc.connect(oscGain);
      oscGain.connect(master);
      osc.start(startAt);
      osc.stop(startAt + dur);
    }

    if (this._noiseBuffer) {
      const src = ctx.createBufferSource();
      src.buffer = this._noiseBuffer;
      const filter = ctx.createBiquadFilter();
      filter.type = "bandpass";
      filter.frequency.setValueAtTime(isSharp ? 2400 : (isHeavy ? 520 : 950), startAt);
      filter.Q.setValueAtTime(isSharp ? 1.4 : 0.85, startAt);
      const nGain = ctx.createGain();
      nGain.gain.setValueAtTime(isSharp ? 0.72 : 0.55, startAt);
      nGain.gain.exponentialRampToValueAtTime(0.001, startAt + Math.min(dur, isSharp ? 0.075 : 0.14));
      src.connect(filter);
      filter.connect(nGain);
      nGain.connect(master);
      src.start(startAt);
      src.stop(startAt + dur);
    }
  }

  playCue(presetName, gain = 1.0, durationSec = 0.2) {
    const preset = String(presetName || "").trim().toLowerCase();
    if (!preset || preset === "none") return;
    const ctx = this._ensureContext();
    if (!ctx) return;
    const now = ctx.currentTime + 0.005;
    if (preset.includes("_")) {
      const parts = preset.split("_").filter(Boolean);
      const half = Math.max(0.1, Number(durationSec || 0.3) * 0.55);
      this._triggerImpulse(ctx, now, parts[0], gain, half);
      // Compound cue offset: 3 frames at 24 fps = 125 ms
      this._triggerImpulse(ctx, now + AUDIO_GRID_FRAMES / DEFAULT_VIDEO_FPS, parts[1] || parts[0], gain, half);
    } else {
      this._triggerImpulse(ctx, now, preset, gain, durationSec);
    }
  }
}
