/**
 * ComfyUI / Spatial Director WebSocket Client (`ws_client.js`).
 *
 * Streams live `kj_preview_override` payloads (`video/mp4`, `image/webp`, `image/jpeg`),
 * turns base64 media into CSP-safe `blob:` URLs, manages `<video muted playsInline>`
 * offscreen decode elements, and dispatches step progress + `executed` completion events.
 */

export function b64ToBlobUrl(b64, mime) {
  const raw = atob(b64);
  const len = raw.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = raw.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: mime || "application/octet-stream" });
  return URL.createObjectURL(blob);
}

export class DirectorWSClient {
  constructor(handlers = {}) {
    this.handlers = handlers;
    this.clientId = this._makeClientId();
    this.ws = null;
    this._reconnectTimer = null;
    this._retryMs = 1000;
    this._closedByUser = false;
    this._activePreviewBlobUrl = null;
    this._previewVideo = null;
    this._previewImage = null;
    this._videoSink = null;
  }

  _makeClientId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID().replace(/-/g, "");
    }
    return `dir${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  }

  setVideoSink(el) {
    this._videoSink = el;
  }

  connect() {
    this._closedByUser = false;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/api/ws?clientId=${encodeURIComponent(this.clientId)}`;
    let sock;
    try {
      sock = new WebSocket(url);
    } catch (err) {
      this._scheduleReconnect();
      return;
    }
    sock.binaryType = "arraybuffer";
    this.ws = sock;

    sock.addEventListener("open", () => {
      this._retryMs = 1000;
      this.handlers.onConnection?.("connected");
    });

    sock.addEventListener("close", () => {
      this.handlers.onConnection?.("disconnected");
      if (!this._closedByUser) {
        this._scheduleReconnect();
      }
    });

    sock.addEventListener("error", () => {
      this.handlers.onConnection?.("error");
    });

    sock.addEventListener("message", (ev) => {
      if (typeof ev.data === "string") {
        this._handleJsonText(ev.data);
      } else if (ev.data instanceof ArrayBuffer) {
        this._handleBinaryPreview(ev.data);
      }
    });
  }

  disconnect() {
    this._closedByUser = true;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this.ws) {
      try {
        this.ws.close();
      } catch {}
      this.ws = null;
    }
  }

  _scheduleReconnect() {
    if (this._reconnectTimer || this._closedByUser) return;
    const delay = this._retryMs;
    this._retryMs = Math.min(6000, Math.round(this._retryMs * 1.5));
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this.connect();
    }, delay);
  }

  _revokeLater(url) {
    if (!url || !url.startsWith("blob:")) return;
    setTimeout(() => {
      try {
        URL.revokeObjectURL(url);
      } catch {}
    }, 250);
  }

  _handleJsonText(rawText) {
    let msg;
    try {
      msg = JSON.parse(rawText);
    } catch {
      return;
    }
    const type = msg?.type;
    const data = msg?.data || {};
    if (type === "status") {
      this.handlers.onStatus?.(data);
    } else if (type === "execution_start") {
      this.handlers.onExecutionStart?.(data);
    } else if (type === "executing") {
      this.handlers.onExecuting?.(data);
    } else if (type === "progress") {
      this.handlers.onProgress?.(data);
    } else if (type === "executed") {
      this.handlers.onExecuted?.(data);
    } else if (type === "execution_error" || type === "execution_interrupted") {
      this.handlers.onExecutionError?.(type, data);
    } else if (type === "kj_preview_override") {
      this.ingestPreviewOverride(data);
    } else if (type === "proxy_error") {
      this.handlers.onConnection?.("proxy_error", data);
    }
  }

  _handleBinaryPreview(buf) {
    if (buf.byteLength <= 8) return;
    const view = new DataView(buf);
    const evtType = view.getUint32(0, false);
    if (evtType !== 1) return; // standard ComfyUI preview image
    const imgFmt = view.getUint32(4, false);
    const mime = imgFmt === 2 ? "image/png" : "image/jpeg";
    const blob = new Blob([buf.slice(8)], { type: mime });
    const url = URL.createObjectURL(blob);
    this._loadPreviewImage(url, { mime, step: null, total: null, fallbackBinary: true });
  }

  ingestPreviewOverride(data) {
    if (!data || typeof data !== "object") return;
    this.handlers.onPreviewTelemetry?.(data);
    if (!data.image) return;
    const mime = String(data.mime || "image/jpeg").toLowerCase();
    let blobUrl;
    try {
      blobUrl = b64ToBlobUrl(data.image, mime);
    } catch {
      return;
    }

    if (mime === "video/mp4") {
      this._loadPreviewVideo(blobUrl, data);
    } else {
      this._loadPreviewImage(blobUrl, { ...data, mime });
    }
  }

  _loadPreviewVideo(blobUrl, meta) {
    if (!this._previewVideo) {
      const v = document.createElement("video");
      v.muted = true;
      v.playsInline = true;
      v.loop = true;
      v.preload = "auto";
      v.className = "offscreen-media";
      if (this._videoSink) {
        this._videoSink.appendChild(v);
      }
      this._previewVideo = v;
    }
    const video = this._previewVideo;
    const prevUrl = this._activePreviewBlobUrl;
    this._activePreviewBlobUrl = blobUrl;

    const onLoaded = () => {
      video.removeEventListener("loadeddata", onLoaded);
      if (prevUrl && prevUrl !== blobUrl) this._revokeLater(prevUrl);
      this.handlers.onPreviewMedia?.({
        kind: "video",
        media: video,
        mime: "video/mp4",
        step: meta.step,
        total: meta.total,
        fps: meta.fps || 7,
        w: meta.w || video.videoWidth || 0,
        h: meta.h || video.videoHeight || 0,
        nodeId: meta.node_id,
        avgStepMs: meta.avg_step_ms,
        delta: meta.delta,
      });
    };
    video.addEventListener("loadeddata", onLoaded);
    video.src = blobUrl;
    try {
      video.load();
    } catch {}
  }

  _loadPreviewImage(blobUrl, meta) {
    const img = new Image();
    const prevUrl = this._activePreviewBlobUrl;
    this._activePreviewBlobUrl = blobUrl;
    img.onload = () => {
      if (prevUrl && prevUrl !== blobUrl) this._revokeLater(prevUrl);
      this._previewImage = img;
      this.handlers.onPreviewMedia?.({
        kind: "image",
        media: img,
        mime: meta.mime || "image/jpeg",
        step: meta.step,
        total: meta.total,
        fps: meta.fps || null,
        w: meta.w || img.naturalWidth || 0,
        h: meta.h || img.naturalHeight || 0,
        nodeId: meta.node_id,
        avgStepMs: meta.avg_step_ms,
        delta: meta.delta,
        fallbackBinary: Boolean(meta.fallbackBinary),
      });
    };
    img.onerror = () => {
      this._revokeLater(blobUrl);
    };
    img.src = blobUrl;
  }

  clearPreviewMedia() {
    if (this._previewVideo) {
      try {
        this._previewVideo.pause();
        this._previewVideo.removeAttribute("src");
        this._previewVideo.load();
      } catch {}
    }
    if (this._activePreviewBlobUrl) {
      this._revokeLater(this._activePreviewBlobUrl);
      this._activePreviewBlobUrl = null;
    }
    this._previewImage = null;
  }
}
