/**
 * Interactive Spatial Director — Takes Ledger, Lineage Chaining & Audio Re-Dub (`takes_ui.js`).
 *
 * Encapsulates takes list rendering, stage clip loading, Tail -> Shot chaining,
 * Pin -> Last frame bridging, FrozenVideoCache audio re-dubbing, and in-flight
 * render rehydration.
 */

import { videoItemViewUrl } from "./ab_compare.js";
import { clamp } from "./stage.js";

const KIND_LABELS = {
  anchor: "Still Anchor",
  mask_preview: "Pre-Viz",
  stamp: "Cutout",
  redub: "Audio Re-Dub",
  imported: "Imported",
  spatial: "World Clip",
};

export function findBaselineTake(takes, directedTake) {
  if (!directedTake) return null;
  const list = Array.isArray(takes) ? takes : [];
  const latest = directedTake.take_id
    ? (list.find((t) => t.take_id === directedTake.take_id) || directedTake)
    : directedTake;
  if (latest.baseline_video) return latest;
  if (latest.baseline_prompt_id) {
    return list.find(
      (t) =>
        t.prompt_id === latest.baseline_prompt_id
        || (t.baseline_prompt_id === latest.baseline_prompt_id && t.baseline_video)
    ) || null;
  }
  return list.find(
    (t) => t.seed === latest.seed && t.prompt_id !== latest.prompt_id && (t.baseline_video || t.clean_video)
  ) || null;
}

export function videoItemRef(item) {
  if (!item) return "";
  if (item.type && item.type !== "input") {
    return videoItemViewUrl(item);
  }
  return `${item.subfolder ? `${item.subfolder}/` : ""}${item.filename}`;
}

export function createTakesController({
  state,
  $,
  apiJson,
  showToast,
  getWsClient,
  getAbCompare,
  getStage,
  setActiveRender,
  syncMediaPlayback,
  syncAllUI,
  schedulePreflight,
  loadAnchorImageUrl,
  applyImportedRecipe,
  onOpenABCompare,
}) {
  function findBaseline(directedTake) {
    return findBaselineTake(state.takes, directedTake);
  }

  function rehydrateInFlightRender() {
    if (state.activeRender || !(state.activeQueuePromptIds instanceof Set) || state.activeQueuePromptIds.size === 0) {
      return;
    }
    const wsClient = getWsClient?.();
    const queuedMatches = state.takes.filter(
      (t) =>
        t &&
        (
          (t.status === "queued" && state.activeQueuePromptIds.has(String(t.prompt_id || "")))
          || (t.baseline_prompt_id && !t.baseline_video && state.activeQueuePromptIds.has(String(t.baseline_prompt_id)))
        )
    );
    if (!queuedMatches.length) return;
    const take = queuedMatches.find((t) => t.client_id && t.client_id === wsClient?.clientId) || queuedMatches[0];
    const primaryActive = take.status === "queued" && state.activeQueuePromptIds.has(String(take.prompt_id || ""));
    const baseActive = Boolean(
      take.baseline_prompt_id && !take.baseline_video && state.activeQueuePromptIds.has(String(take.baseline_prompt_id))
    );
    const pid = primaryActive ? String(take.prompt_id || "") : (baseActive ? String(take.baseline_prompt_id || "") : "");
    if (!pid) return;

    const kind = primaryActive ? String(take.kind || "render") : "baseline";
    const wf = String(take.mode || take.params?.workflow || "");
    let total = Number(take.params?.steps || 6);
    let phase = `Generating ${take.length || state.length}f World Clip (${wf || "i2v_portal"})…`;
    let meta = null;

    if (!primaryActive && baseActive) {
      state.activeBaselinePromptId = pid;
      phase = `Rendering Same-Seed Baseline (${take.length || state.length}f A|B)…`;
    } else if (kind === "anchor") {
      state.activeAnchorPromptId = pid;
      total = wf === "ph3_image" ? 8 : 6;
      phase = `Generating Still (${wf || "h3_image"})…`;
    } else if (kind === "mask_preview") {
      state.activeMaskPreviewPromptId = pid;
      total = wf === "ph3_image" ? 8 : 6;
      phase = `Mask Preview (${wf || "h3_image"})…`;
      meta = {
        region: take.params?.region || {},
        feather: Number(take.params?.feather ?? state.doc.feather ?? 0.08),
        snapToInclude: Boolean(take.params?.snap_to_include_mask),
        maskRef: take.params?.mask_ref || "",
      };
    } else if (kind === "stamp") {
      state.activeStampPromptId = pid;
      total = wf === "ph3_image" ? 8 : 6;
      phase = "Generating Actor Cutout…";
    } else if (kind === "redub") {
      state.activeRenderPromptId = pid;
      phase = "Re-Dubbing Audio…";
    } else {
      state.activeRenderPromptId = pid;
      if (take.baseline_prompt_id) state.activeBaselinePromptId = String(take.baseline_prompt_id);
    }

    setActiveRender({
      kind,
      pid,
      phase,
      step: 0,
      total,
      meta,
      startedEpochMs: take.created_at ? Number(take.created_at) * 1000 : Date.now(),
    });
  }

  async function restoreTakeRecipe(take, btn = null) {
    if (!take?.take_id) return;
    if (btn) btn.disabled = true;
    try {
      const res = await apiJson("/api/import_recipe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ take_id: take.take_id }),
      });
      await applyImportedRecipe(res.recipe);
      showToast(`Restored recipe (${take.mode || "take"} · seed ${res.recipe?.stage?.seed ?? "?"})`, "ok");
    } catch (err) {
      showToast(`Restore recipe failed: ${err.message}`, "err");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  function loadTakeIntoStage(take) {
    const url = videoItemViewUrl(take.clean_video || take.hud_video);
    if (!url) return;
    if (!state.activeVideoEl) {
      const v = document.createElement("video");
      v.muted = true;
      v.playsInline = true;
      v.loop = true;
      v.preload = "auto";
      v.className = "offscreen-media";
      $("#offscreen-media-sink")?.appendChild(v);
      state.activeVideoEl = v;
    }
    const v = state.activeVideoEl;
    const wsClient = getWsClient?.();
    const abCompare = getAbCompare?.();
    const cleanupListeners = () => {
      v.removeEventListener("loadeddata", onReady);
      v.removeEventListener("error", onError);
    };
    const onReady = () => {
      cleanupListeners();
      state.livePreview = null;
      wsClient?.clearPreviewMedia();
      syncMediaPlayback();
      syncAllUI();
    };
    const onError = () => {
      cleanupListeners();
      if (state.activeStageTake === take) {
        state.activeStageTake = null;
        state.activeVideoUrl = "";
        state.playing = false;
        syncAllUI();
      }
      showToast("Video file unavailable (temp output cleared on ComfyUI restart).", "err", 4500);
    };
    cleanupListeners();
    v.addEventListener("loadeddata", onReady);
    v.addEventListener("error", onError);

    state.hasAnchor = true;
    $("#empty-anchor-card")?.classList.add("hidden");
    if (!state.anchorRef && take.params?.source_ref) {
      state.anchorRef = String(take.params.source_ref);
      state.anchorName = state.anchorRef.split("/").pop() || take.mode || "Rendered Take";
    } else if (!state.anchorName) {
      state.anchorName = take.mode || "Rendered Take";
    }
    const takeLen = Number(take.length || 0);
    if (takeLen > 0 && state.lengths.includes(takeLen) && takeLen !== state.length) {
      state.length = takeLen;
      state.frame = clamp(state.frame, 0, Math.max(0, state.length - 1));
    }
    state.livePreview = null;
    wsClient?.clearPreviewMedia();
    state.activeStageTake = take;
    state.activeVideoUrl = url;
    v.src = url;
    try { v.load(); } catch {}
    syncMediaPlayback();
    syncAllUI();

    abCompare?.setTakes({
      directedTake: take,
      baselineTake: findBaseline(take),
    });
  }

  async function chainTakeFrameToAnchor(take, frameIndex = -1, btn = null, opts = {}) {
    if (!take?.take_id) return;
    const asLastFrame = Boolean(opts.asLastFrame);
    if (btn) btn.disabled = true;
    try {
      const res = await apiJson("/api/extract_frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ take_id: take.take_id, frame_index: frameIndex }),
      });
      const url = res.view_url || videoItemViewUrl(res);
      if (!url) throw new Error("Extracted frame returned no view URL");
      if (asLastFrame) {
        state.lastFrameRef = res.ref || "";
        state.lastFrameName = res.filename || `shot_f${res.frame_index}.png`;
        const hasI2V = state.workflows.some((w) => w.id === "i2v_portal" && w.available !== false);
        if (hasI2V) {
          state.workflowId = "i2v_portal";
          const wfSel = $("#engine-workflow");
          if (wfSel) wfSel.value = "i2v_portal";
        }
        syncAllUI();
        schedulePreflight();
        showToast(`Pinned f${res.frame_index} as destination Last Frame`, "ok");
      } else {
        if (state.anchorRef) {
          state.shotHistory.push({
            anchorRef: state.anchorRef,
            anchorName: state.anchorName,
            anchorUrl: state.anchorMedia?.src || "",
            lastFrameRef: state.lastFrameRef,
            lastFrameName: state.lastFrameName,
            workflowId: state.workflowId,
          });
          if (state.shotHistory.length > 8) state.shotHistory.shift();
        }
        state.lastFrameRef = "";
        state.lastFrameName = "";
        state.parentTakeId = String(take.take_id || "");
        state.parentFrame = Number(res.frame_index ?? frameIndex);
        await loadAnchorImageUrl(url, {
          ref: res.ref,
          name: res.filename || `shot_f${res.frame_index}.png`,
        });
        const hasI2V = state.workflows.some((w) => w.id === "i2v_portal" && w.available !== false);
        if (hasI2V) {
          state.workflowId = "i2v_portal";
          const wfSel = $("#engine-workflow");
          if (wfSel) wfSel.value = "i2v_portal";
        }
        syncAllUI();
        schedulePreflight();
        const maxIdx = Math.max(0, Number(res.total_frames || 1) - 1);
        showToast(`Chained f${res.frame_index}/${maxIdx} → Shot ${state.shotHistory.length + 1} anchor`, "ok");
      }
    } catch (err) {
      showToast(`Frame chain failed: ${err.message}`, "err");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async function redubTakeAudio(take, btn = null) {
    if (!take?.take_id) return;
    const wsClient = getWsClient?.();
    if (btn) btn.disabled = true;
    try {
      const redubMode = $("#redub-mode-select")?.value || "polish";
      const customAudioPrompt = ($("#redub-prompt-input")?.value || "").trim();
      const payload = {
        take_id: take.take_id,
        redub_mode: redubMode,
        steps: 6,
        source_length: Number(state.length),
        events_json: JSON.stringify(state.doc),
        client_id: wsClient.clientId,
      };
      if (state.seedLocked) payload.seed = Number(state.seed);
      if (customAudioPrompt) payload.audio_prompt = customAudioPrompt;
      else if (state.targetPrompt) payload.audio_prompt = state.targetPrompt;
      const res = await apiJson("/api/redub", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      state.activeRenderPromptId = res.prompt_id || "";
      setActiveRender({
        kind: "redub",
        pid: res.prompt_id || "",
        phase: "Re-Dubbing Audio…",
        step: 0,
        total: Number(payload.steps || 6),
      });
      const modeTag = res.redub_mode === "fresh" ? "Fresh 1.0x" : "Polish 0.5x";
      showToast(`Queued ${modeTag} audio re-dub (${res.take_id || "queued"})`, "ok");
      await refreshTakes();
    } catch (err) {
      showToast(`Audio re-dub failed: ${err.message}`, "err");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  function renderTakesList() {
    const list = $("#takes-list");
    if (!list) return;
    const abCompare = getAbCompare?.();
    if (!state.takes.length) {
      list.replaceChildren();
      const empty = document.createElement("div");
      empty.className = "empty-takes";
      empty.textContent = "No takes rendered yet.";
      list.appendChild(empty);
      return;
    }
    list.replaceChildren();
    for (const take of state.takes) {
      const card = document.createElement("div");
      card.className = "take-card";

      const top = document.createElement("div");
      top.className = "take-card-top";
      const linTag = take.parent_take_id
        ? ` · ↳ ${String(take.parent_take_id).slice(-6)}${take.parent_frame != null ? `@f${take.parent_frame}` : ""}`
        : "";
      const kindLabel = KIND_LABELS[take.kind] || take.kind || "Take";
      const lenLabel = take.still_image ? "1f" : `${take.length || "?"}f`;
      const titleSpan = document.createElement("span");
      titleSpan.className = "take-card-title";
      titleSpan.textContent = `${kindLabel} · ${take.mode || "take"} · ${lenLabel} · seed ${take.seed || "?"}${linTag}`;
      titleSpan.title = titleSpan.textContent;
      top.appendChild(titleSpan);

      const badges = document.createElement("div");
      badges.className = "take-card-badges";
      if (take.clean_video && take.baseline_video) {
        const abBadge = document.createElement("span");
        abBadge.className = "take-ab-badge";
        abBadge.textContent = "A|B";
        abBadge.title = "Same-seed counterfactual baseline ready for split comparison";
        badges.appendChild(abBadge);
      }

      const statusChip = document.createElement("span");
      const statusStr = take.baseline_pending ? "baseline running" : (take.status || "completed");
      statusChip.className = `take-status-chip ${
        statusStr === "error" || statusStr === "interrupted"
          ? "err"
          : (statusStr === "queued" || take.baseline_pending ? "pending" : "ok")
      }`;
      statusChip.textContent = statusStr;
      badges.appendChild(statusChip);
      top.appendChild(badges);
      card.appendChild(top);

      const promptText = String(take.prompt_preview || take.params?.target_prompt || take.params?.prompt || "").trim();
      if (promptText) {
        const pEl = document.createElement("div");
        pEl.className = "take-card-prompt";
        pEl.textContent = promptText;
        pEl.title = promptText;
        card.appendChild(pEl);
      }

      if ((take.status === "error" || take.status === "interrupted" || take.baseline_error) && (take.error || take.baseline_error)) {
        const errEl = document.createElement("div");
        errEl.className = "take-card-error";
        errEl.textContent = String(take.error || take.baseline_error).trim();
        card.appendChild(errEl);
      }

      const actions = document.createElement("div");
      actions.className = "take-card-actions";

      if (take.params && typeof take.params === "object" && Object.keys(take.params).length > 0 && take.kind !== "imported") {
        const restoreBtn = document.createElement("button");
        restoreBtn.type = "button";
        restoreBtn.className = "btn-outline-sm";
        restoreBtn.textContent = "Restore Recipe";
        restoreBtn.title = "Restore saved prompts, portals, cues, seed, and controls back onto stage";
        restoreBtn.addEventListener("click", () => restoreTakeRecipe(take, restoreBtn));
        actions.appendChild(restoreBtn);
      }

      if (take.still_image && (take.kind === "anchor" || take.kind === "mask_preview")) {
        const useStillBtn = document.createElement("button");
        useStillBtn.type = "button";
        useStillBtn.className = "btn-outline-sm";
        useStillBtn.textContent = "Use Still";
        useStillBtn.title = "Load this 1-frame still as the active World State anchor";
        useStillBtn.addEventListener("click", async () => {
          try {
            await loadAnchorImageUrl(videoItemViewUrl(take.still_image), {
              ref: videoItemRef(take.still_image),
              name: take.still_image.filename || "Anchor Still",
              preserveOriginalPrompt: true,
            });
            showToast("Loaded still onto stage", "ok");
          } catch (err) {
            showToast(err.message || "Still image expired from temp/", "err");
          }
        });
        actions.appendChild(useStillBtn);
      }

      if (take.clean_video || take.hud_video) {
        const playBtn = document.createElement("button");
        playBtn.type = "button";
        playBtn.className = "btn-outline-sm";
        playBtn.textContent = "Load Stage";
        playBtn.addEventListener("click", () => loadTakeIntoStage(take));
        actions.appendChild(playBtn);

        const abBtn = document.createElement("button");
        abBtn.type = "button";
        abBtn.className = "btn-outline-sm";
        abBtn.textContent = "Open A|B";
        abBtn.addEventListener("click", () => onOpenABCompare(take));
        actions.appendChild(abBtn);

        const expClean = document.createElement("button");
        expClean.type = "button";
        expClean.className = "btn-outline-sm";
        expClean.textContent = "Export Clean";
        expClean.addEventListener("click", () => {
          abCompare?.setTakes({ directedTake: take, baselineTake: findBaseline(take) });
          abCompare?.exportTake("clean");
        });
        actions.appendChild(expClean);

        const chainBtn = document.createElement("button");
        chainBtn.type = "button";
        chainBtn.className = "btn-outline-sm";
        chainBtn.textContent = "Tail → Shot";
        chainBtn.title = "Extract last frame as lossless PNG anchor for Shot N+1 (i2v_portal)";
        chainBtn.addEventListener("click", () => chainTakeFrameToAnchor(take, -1, chainBtn));
        actions.appendChild(chainBtn);

        const pinLastBtn = document.createElement("button");
        pinLastBtn.type = "button";
        pinLastBtn.className = "btn-outline-sm";
        pinLastBtn.textContent = "Pin → Last";
        pinLastBtn.title = "Extract last frame as destination last_frame bridge (i2v_portal node 16)";
        pinLastBtn.addEventListener("click", () => chainTakeFrameToAnchor(take, -1, pinLastBtn, { asLastFrame: true }));
        actions.appendChild(pinLastBtn);

        const redubBtn = document.createElement("button");
        redubBtn.type = "button";
        redubBtn.className = "btn-outline-sm";
        redubBtn.textContent = "Re-Dub Audio";
        redubBtn.title = "Queue 2nd-pass FrozenVideoCache audio re-dub keeping video locked";
        redubBtn.addEventListener("click", () => redubTakeAudio(take, redubBtn));
        actions.appendChild(redubBtn);
      }

      card.appendChild(actions);
      list.appendChild(card);
    }
  }

  async function refreshTakes() {
    try {
      const data = await apiJson("/api/takes");
      state.takes = Array.isArray(data.takes) ? data.takes : [];
      if (data.queue_ok === true && Array.isArray(data.active_prompt_ids)) {
        state.activeQueuePromptIds = new Set(data.active_prompt_ids.map(String));
      }
      if (state.activeStageTake?.take_id) {
        const freshStage = state.takes.find((t) => t.take_id === state.activeStageTake.take_id);
        if (freshStage) {
          state.activeStageTake = freshStage;
          const abCompare = getAbCompare?.();
          if (abCompare?.active && abCompare.directedTake?.take_id === freshStage.take_id) {
            abCompare.setTakes({
              directedTake: freshStage,
              baselineTake: findBaseline(freshStage),
            });
          }
        }
      }
      rehydrateInFlightRender();
      renderTakesList();
    } catch {}
  }

  return {
    findBaselineTake: findBaseline,
    rehydrateInFlightRender,
    refreshTakes,
    renderTakesList,
    loadTakeIntoStage,
    chainTakeFrameToAnchor,
    redubTakeAudio,
    restoreTakeRecipe,
  };
}
