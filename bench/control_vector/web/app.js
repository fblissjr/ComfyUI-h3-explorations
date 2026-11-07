const APP = {
  families: 'fl2va,ref2va',
  vectors: [],
  tomlPacks: [],
  expandedPairs: [],
  sculptRows: [
    { vector: 'film_16mm_1974', strength: 1.0, mode: 'add' },
    { vector: 'chaos_panic', strength: 1.0, mode: 'hold' }
  ],
  mediaDataUrlA: null,
  mediaDataUrlB: null,
  systemPromptSummary: ''
};

function setStatus(text, kind = '') {
  const el = document.getElementById('status-indicator');
  if (!el) return;
  el.textContent = text;
  el.className = 'status-pill' + (kind ? ' ' + kind : '');
}

function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function setGlobalFamily(fam) {
  APP.families = fam;
  document.querySelectorAll('#global-family-group .family-pill').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-fam') === fam);
  });
}

function switchScreen(screenId) {
  document.querySelectorAll('.screen').forEach(sec => {
    sec.classList.toggle('active', sec.id === screenId);
  });
  document.querySelectorAll('.nav-rail .nav-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-screen') === screenId);
  });
}

function switchMeterSubtab(which) {
  const isReceptor = which === 'receptor';
  document.getElementById('subtab-receptor').style.display = isReceptor ? 'block' : 'none';
  document.getElementById('subtab-jlens').style.display = isReceptor ? 'none' : 'block';
  document.getElementById('subtab-btn-receptor').classList.toggle('active', isReceptor);
  document.getElementById('subtab-btn-jlens').classList.toggle('active', !isReceptor);
}

async function apiGet(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

async function apiPost(url, payload) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function renderBanner(containerId, html, isError = false) {
  const box = document.getElementById(containerId);
  if (!box) return;
  box.innerHTML = `<div class="result-banner ${isError ? 'error' : ''}">${html}</div>`;
}

async function loadInitData() {
  try {
    setStatus('Loading library...', 'busy');
    const data = await apiGet('/api/init');
    APP.vectors = data.vectors || [];
    APP.tomlPacks = data.toml_files || data.toml_packs || data.tomls || [];
    APP.systemPromptSummary = data.system_prompt || '';
    const provSel = document.getElementById('llm-provider-select');
    if (provSel && data.default_llm_provider) {
      provSel.value = data.default_llm_provider;
    }
    const modInp = document.getElementById('llm-model-input');
    if (modInp && data.default_llm_model && !modInp.value) {
      modInp.value = data.default_llm_model;
    }
    renderLibraryTables();
    populateVectorSelects();
    renderSculptRows();
    setStatus(`Ready • ${APP.vectors.length} vectors`, 'ok');
  } catch (err) {
    setStatus('Offline / Error', 'err');
    console.error(err);
  }
}

function populateVectorSelects() {
  const selects = ['quick-invert-source', 'meter-vector-select', 'jlens-inspect-select'];
  selects.forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = APP.vectors.map(v => {
      const val = typeof v === 'string' ? v : (v.file || v.name || v.stem);
      const label = typeof v === 'string' ? v : (`${v.name || v.file} (${v.family || 'fl2va'})`);
      return `<option value="${escapeHtml(val)}">${escapeHtml(label)}</option>`;
    }).join('');
    if (prev && Array.from(sel.options).some(o => o.value === prev)) {
      sel.value = prev;
    }
  });

  const tomlSel = document.getElementById('pairs-toml-select');
  if (tomlSel) {
    tomlSel.innerHTML = '<option value="">-- Load existing .toml pack --</option>' +
      APP.tomlPacks.map(t => {
        const stem = typeof t === 'string' ? t.replace(/\.toml$/, '') : (t.name || (t.file || '').replace(/\.toml$/, ''));
        return `<option value="${escapeHtml(stem)}">${escapeHtml(stem)}.toml</option>`;
      }).join('');
  }
}

function renderLibraryTables() {
  const q = (document.getElementById('lib-filter-input')?.value || '').trim().toLowerCase();
  const vecBody = document.getElementById('lib-vectors-tbody');
  const tomlBody = document.getElementById('lib-tomls-tbody');

  if (vecBody) {
    const filteredVecs = APP.vectors.filter(v => {
      const file = (typeof v === 'string' ? v : (v.file || v.name || '')).toLowerCase();
      return !q || file.includes(q);
    });
    if (!filteredVecs.length) {
      vecBody.innerHTML = '<tr><td colspan="2" style="color:var(--text-muted);">No matching vectors found.</td></tr>';
    } else {
      vecBody.innerHTML = filteredVecs.map(v => {
        const file = typeof v === 'string' ? v : (v.file || v.name);
        const disp = typeof v === 'string' ? v : `${v.name || v.file} [${v.family || 'fl2va'}]`;
        return `<tr>
          <td class="mono" style="font-size:13px;" title="${escapeHtml(file)}">${escapeHtml(disp)}</td>
          <td>
            <div class="btn-row">
              <button type="button" class="btn btn-sm" onclick="prefillInvert('${escapeHtml(file)}')">Invert (-1x)</button>
              <button type="button" class="btn btn-sm" onclick="prefillSculpt('${escapeHtml(file)}')">Sculpt</button>
              <button type="button" class="btn btn-sm" onclick="prefillMeter('${escapeHtml(file)}')">Test on Prompt</button>
              <button type="button" class="btn btn-sm" onclick="prefillJLens('${escapeHtml(file)}')">J-Lens Readout</button>
            </div>
          </td>
        </tr>`;
      }).join('');
    }
  }

  if (tomlBody) {
    const filteredTomls = APP.tomlPacks.filter(t => {
      const stem = (typeof t === 'string' ? t : (t.name || t.file || '')).toLowerCase();
      return !q || stem.includes(q);
    });
    if (!filteredTomls.length) {
      tomlBody.innerHTML = '<tr><td colspan="2" style="color:var(--text-muted);">No matching .toml packs found.</td></tr>';
    } else {
      tomlBody.innerHTML = filteredTomls.map(t => {
        const stem = typeof t === 'string' ? t.replace(/\.toml$/, '') : (t.name || (t.file || '').replace(/\.toml$/, ''));
        return `<tr>
          <td class="mono" style="font-size:13px;">${escapeHtml(stem)}.toml</td>
          <td>
            <div class="btn-row">
              <button type="button" class="btn btn-sm" onclick="openTomlInStudio('${escapeHtml(stem)}')">Edit Pairs in Studio &rarr;</button>
            </div>
          </td>
        </tr>`;
      }).join('');
    }
  }
}

function prefillInvert(fileOrStem) {
  switchScreen('screen-sculpt');
  const sel = document.getElementById('quick-invert-source');
  if (sel) sel.value = fileOrStem;
  const clean = fileOrStem.split('__')[0].replace(/\.safetensors$/, '');
  document.getElementById('quick-invert-name').value = `${clean}_inverted`;
}

function prefillSculpt(fileOrStem) {
  switchScreen('screen-sculpt');
  APP.sculptRows[0] = { vector: fileOrStem, strength: 1.0, mode: 'add' };
  renderSculptRows();
}

function prefillMeter(fileOrStem) {
  switchScreen('screen-meter');
  switchMeterSubtab('receptor');
  const sel = document.getElementById('meter-vector-select');
  if (sel) sel.value = fileOrStem;
}

function prefillJLens(fileOrStem) {
  switchScreen('screen-meter');
  switchMeterSubtab('jlens');
  const sel = document.getElementById('jlens-inspect-select');
  if (sel) sel.value = fileOrStem;
  runJLensInspect();
}

/* ========================================================================= */
/* SCREEN 2: CLAUSE CONTRAST BUILDER                                         */
/* ========================================================================= */
function fillClauseExample(kind) {
  if (kind === 'scorching') {
    document.getElementById('clause-name').value = 'scorching_heatwave';
    document.getElementById('clause-slot').value = 'opening_framing';
    document.getElementById('clause-pole-a').value = 'Crisp overcast afternoon light with cool slate shadows, still air, dry pavement, and neutral color balance across every surface.';
    document.getElementById('clause-pole-b').value = 'Blistering midday desert heatwave under a bleached white sun, rising thermal shimmer warping the horizon, harsh amber glare, and sun-baked dust.';
    document.getElementById('clause-sound-a').value = 'Quiet open-air room tone with gentle distant breeze.';
    document.getElementById('clause-sound-b').value = 'Low parched thermal wind drone, dry cicada buzz, and heat-cracked grit underfoot.';
  } else if (kind === 'noir') {
    document.getElementById('clause-name').value = 'chiaroscuro_noir';
    document.getElementById('clause-slot').value = 'opening_framing';
    document.getElementById('clause-pole-a').value = 'Soft even commercial fill lighting with balanced midtones, open shadows, and natural daylight color rendition.';
    document.getElementById('clause-pole-b').value = 'Hard single-source chiaroscuro key light slicing through deep ink-black shadows, sharp Venetian-blind light slashes, and silver rim highlights.';
    document.getElementById('clause-sound-a').value = '';
    document.getElementById('clause-sound-b').value = '';
  } else if (kind === 'handheld') {
    document.getElementById('clause-name').value = 'urgent_handheld';
    document.getElementById('clause-slot').value = 'camera_motion';
    document.getElementById('clause-pole-a').value = 'Locked-off tripod composition with level horizon, steady 50mm prime framing, and zero camera shake.';
    document.getElementById('clause-pole-b').value = 'Urgent shoulder-mounted documentary camera with visceral micro-jitter, reactive reframing, and breathing lens drift.';
    document.getElementById('clause-sound-a').value = '';
    document.getElementById('clause-sound-b').value = '';
  }
}

async function expandClauseScenes() {
  try {
    setStatus('Expanding 10 scene pairs...', 'busy');
    const payload = {
      name: document.getElementById('clause-name').value.trim(),
      pole_a: document.getElementById('clause-pole-a').value.trim(),
      pole_b: document.getElementById('clause-pole-b').value.trim(),
      slot: document.getElementById('clause-slot').value,
      sound_a: document.getElementById('clause-sound-a').value.trim(),
      sound_b: document.getElementById('clause-sound-b').value.trim(),
      hold_b: document.getElementById('clause-hold-text').value.trim(),
      target_prompt: document.getElementById('clause-scene-prompt').value.trim()
    };
    const data = await apiPost('/api/expand-scenes', payload);
    APP.expandedPairs = (data.pairs || []).map((p, idx) => ({
      index: idx,
      label: p.scene_title || p.label || `Scene Rig #${idx + 1}`,
      prompt_a: p.a || p.prompt_a || '',
      prompt_b: p.b || p.prompt_b || '',
      weight: p.weight ?? 1.0
    }));
    renderExpandedClausePairs();
    document.getElementById('clause-pairs-card').style.display = 'block';
    renderBanner('clause-status-box', `Generated <strong>${APP.expandedPairs.length} matched H3 scene pairs</strong>. Review or edit any pair below, or click <strong>Encode &amp; Build Control Vector Now (GPU)</strong>.`);
    setStatus('10 pairs ready', 'ok');
  } catch (err) {
    renderBanner('clause-status-box', `Error expanding scenes: ${escapeHtml(err.message)}`, true);
    setStatus('Expand failed', 'err');
  }
}

function renderExpandedClausePairs() {
  const container = document.getElementById('clause-pairs-container');
  if (!container) return;
  container.innerHTML = APP.expandedPairs.map((pair, idx) => `
    <div class="pair-card">
      <div class="pair-card-head">
        <div style="display:flex;align-items:center;gap:10px;">
          <span class="badge">#${idx + 1}</span>
          <strong>${escapeHtml(pair.label)}</strong>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
          <label style="font-size:12.5px;color:var(--text-secondary);">Pair Weight:</label>
          <input type="number" step="0.1" value="${pair.weight}" style="width:80px;padding:4px 8px;" onchange="APP.expandedPairs[${idx}].weight = parseFloat(this.value || '1')" />
        </div>
      </div>
      <div class="grid-2">
        <div class="field pole-a-box" style="margin:0;">
          <label class="field-label"><span class="badge pole-a">Prompt A (Baseline)</span></label>
          <textarea class="mono" rows="7" onchange="APP.expandedPairs[${idx}].prompt_a = this.value">${escapeHtml(pair.prompt_a)}</textarea>
        </div>
        <div class="field pole-b-box" style="margin:0;">
          <label class="field-label"><span class="badge pole-b">Prompt B (Target)</span></label>
          <textarea class="mono" rows="7" onchange="APP.expandedPairs[${idx}].prompt_b = this.value">${escapeHtml(pair.prompt_b)}</textarea>
        </div>
      </div>
    </div>
  `).join('');
}

function buildMarkdownFromExpandedPairs() {
  return APP.expandedPairs.map((p, idx) =>
    `#### Pair ${idx + 1} — ${p.label}\n##### Prompt A (Baseline)\n${p.prompt_a.trim()}\n\n##### Prompt B (Steered)\n${p.prompt_b.trim()}\n`
  ).join('\n');
}

async function buildFromClauseScreen() {
  try {
    if (!APP.expandedPairs.length) {
      await expandClauseScenes();
    }
    const name = document.getElementById('clause-name').value.trim();
    if (!name) throw new Error('Please enter an Output Vector Name.');
    setStatus(`Building ${name} on GPU...`, 'busy');
    renderBanner('clause-status-box', `Encoding 10 pairs through Qwen3-VL + DiT <code>condition_proj</code> + <code>TokenRefiner</code> for <strong>${escapeHtml(name)}</strong>...`);

    const mdText = buildMarkdownFromExpandedPairs();
    const saveToml = document.getElementById('clause-save-toml').checked;
    const rowsPayload = APP.expandedPairs.map((p, idx) => ({
      index: idx,
      a: p.prompt_a,
      b: p.prompt_b,
      weight: p.weight,
      enabled: true
    }));

    const res = await apiPost('/api/build', {
      name,
      families: APP.families,
      text: mdText,
      rows: rowsPayload,
      save_toml: saveToml,
      overwrite: true
    });

    await loadInitData();
    const files = (res.built || res.written || []).map(f => `<code>${escapeHtml(typeof f === 'string' ? f : f.file)}</code>`).join(', ');
    renderBanner('clause-status-box', `<strong>Built Control Vector <code>${escapeHtml(name)}</code>!</strong><br/>Saved files: ${files || 'see models/control_vectors/'}`);
    setStatus(`Built ${name}`, 'ok');
  } catch (err) {
    renderBanner('clause-status-box', `Build error: ${escapeHtml(err.message)}`, true);
    setStatus('Build failed', 'err');
  }
}

async function repoolCurrentVector() {
  try {
    const name = document.getElementById('clause-name').value.trim();
    const weights = APP.expandedPairs.map(p => p.weight);
    const rowsPayload = APP.expandedPairs.map((p, idx) => ({
      index: idx,
      a: p.prompt_a,
      b: p.prompt_b,
      weight: p.weight,
      enabled: true
    }));
    setStatus(`Re-pooling ${name}...`, 'busy');
    const res = await apiPost('/api/repool', {
      name,
      families: APP.families,
      weights,
      rows: rowsPayload
    });
    const builtInfo = (res.built || []).map(b => `${b.file} (MPC=${Number(b.mpc || 0).toFixed(3)})`).join(', ');
    renderBanner('clause-status-box', `<strong>Instant CPU Re-Pool complete</strong> in ${escapeHtml(res.elapsed_ms || '<1')} ms: <code>${escapeHtml(builtInfo)}</code>.`);
    setStatus('Re-pooled (<1 ms)', 'ok');
  } catch (err) {
    renderBanner('clause-status-box', `Re-pool note: Build the vector on GPU once first so per-pair directions are cached in memory (${escapeHtml(err.message)}).`, true);
    setStatus('Re-pool error', 'err');
  }
}
/* ========================================================================= */
/* SCREEN 3: FULL PAIR & LLM STUDIO                                          */
/* ========================================================================= */
async function generatePairsWithLocalLLM() {
  try {
    const provider = document.getElementById('llm-provider-select').value;
    const model = document.getElementById('llm-model-input').value.trim();
    const concept = document.getElementById('llm-concept-input').value.trim();
    const loadIn4bit = document.getElementById('llm-4bit-check').checked;
    if (!model) {
      throw new Error('Please enter a local LLM Model Directory Path or HuggingFace Repo ID.');
    }
    if (!concept) {
      throw new Error('Please enter a Concept / Axis description to generate.');
    }
    setStatus(`Generating pairs via ${provider}...`, 'busy');
    renderBanner('llm-status-box', `Running <strong>${escapeHtml(provider)}</strong> on <code>${escapeHtml(model)}</code> with the H3 Control Vector System Prompt...`);
    const res = await apiPost('/api/generate-pairs-llm', {
      provider,
      model,
      concept,
      load_in_4bit: loadIn4bit
    });
    document.getElementById('pairs-markdown-editor').value = res.markdown || '';
    document.getElementById('pairs-axis').value = concept;
    renderBanner('llm-status-box', `<strong>Generated Markdown pairs in ${escapeHtml(res.elapsed_s)}s</strong> via <code>${escapeHtml(res.provider)}</code>! Review or edit the pairs in the editor below, then click Validate, Save .toml, or Build Vector.`);
    setStatus(`Generated pairs (${res.elapsed_s}s)`, 'ok');
  } catch (err) {
    renderBanner('llm-status-box', `LLM generation error: ${escapeHtml(err.message)}`, true);
    setStatus('LLM generation failed', 'err');
  }
}

function insertPairTemplate() {
  const tpl = `#### Pair 1
##### Prompt A
integrated_multimodal_description:
[Shot 1] A cinematic medium shot on a 50mm prime lens under crisp overcast afternoon light with cool slate shadows and still air. [00s-05s] An elderly watchmaker sits at a wooden workbench on a covered terrace, adjusting a brass gear with fine tweezers while glancing toward the horizon.

overall_soundscape:
Quiet open-air room tone with gentle distant breeze and soft metallic clicks.

non_diegetic_music:
N/A

##### Prompt B
integrated_multimodal_description:
[Shot 1] A cinematic medium shot on a 50mm prime lens under a blistering midday desert heatwave, rising thermal shimmer warping the horizon, and harsh amber glare. [00s-05s] An elderly watchmaker sits at a wooden workbench on a covered terrace, adjusting a brass gear with fine tweezers while glancing toward the horizon.

overall_soundscape:
Quiet open-air room tone with gentle distant breeze and soft metallic clicks.

non_diegetic_music:
N/A
`;
  const ed = document.getElementById('pairs-markdown-editor');
  if (ed) ed.value = tpl;
}

async function copySystemPrompt() {
  const promptRef = APP.systemPromptSummary ||
    'See docs/wiki/system_prompts/h3_control_vector_system_prompt.md — generate 10 matched H3 Prompt A / Prompt B pairs where subject, camera framing, and [00s-05s] timing are held identical inside each pair and only the target attribute varies.';
  try {
    await navigator.clipboard.writeText(promptRef);
    renderBanner('pairs-status-box', 'Copied the H3 Control Vector System Prompt instructions to your clipboard. Paste into your LLM, then paste the `#### Pair 1` Markdown output into the editor above.');
  } catch (_) {
    renderBanner('pairs-status-box', `System prompt reference: <code>docs/wiki/system_prompts/h3_control_vector_system_prompt.md</code>`);
  }
}

async function openTomlInStudio(stem) {
  switchScreen('screen-pairs');
  const sel = document.getElementById('pairs-toml-select');
  if (sel) sel.value = stem;
  await loadSelectedTomlIntoStudio();
}

async function loadSelectedTomlIntoStudio() {
  try {
    const stem = document.getElementById('pairs-toml-select').value;
    if (!stem) throw new Error('Select a .toml pack from the dropdown first.');
    setStatus(`Loading ${stem}.toml...`, 'busy');
    const data = await apiPost('/api/load-toml', { file: `${stem}.toml` });
    document.getElementById('pairs-name').value = data.name || stem;
    document.getElementById('pairs-axis').value = data.quality || `${data.a_side || ''} -> ${data.b_side || ''}`;
    document.getElementById('pairs-markdown-editor').value = data.markdown || data.text || '';
    renderBanner('pairs-status-box', `Loaded <strong>${escapeHtml(stem)}.toml</strong> (${(data.pairs || []).length} pairs). You can edit the full prompts above, preview token diffs, save back to <code>.toml</code>, or build the vector.`);
    setStatus(`Loaded ${stem}.toml`, 'ok');
  } catch (err) {
    renderBanner('pairs-status-box', `Failed to load .toml: ${escapeHtml(err.message)}`, true);
    setStatus('Load error', 'err');
  }
}

async function previewMarkdownPairs() {
  try {
    const text = document.getElementById('pairs-markdown-editor').value.trim();
    if (!text) throw new Error('Paste or write at least one ##### Prompt A / ##### Prompt B pair first.');
    setStatus('Validating pairs...', 'busy');
    const data = await apiPost('/api/preview', { text });
    const card = document.getElementById('pairs-preview-card');
    const body = document.getElementById('pairs-preview-body');
    card.style.display = 'block';
    const pairs = data.pairs || [];
    body.innerHTML = `
      <div style="margin-bottom:12px;font-size:13.5px;color:var(--text-secondary);">
        Parsed <strong>${pairs.length} pair(s)</strong>. Verify that each pair has balanced token diffs and valid H3 sections:
      </div>
      <table class="lib-table">
        <thead>
          <tr>
            <th>Pair</th>
            <th>Scene Tokens (A / B)</th>
            <th>Changed Span Tokens (Diff A / Diff B)</th>
            <th>Section &amp; Validity</th>
          </tr>
        </thead>
        <tbody>
          ${pairs.map((p, i) => {
            const tok = p.tokens || {};
            return `
            <tr>
              <td class="mono">#${p.index ?? (i + 1)}</td>
              <td class="mono">${tok.scene_a ?? '-'} / ${tok.scene_b ?? '-'}</td>
              <td class="mono"><span class="badge pole-a">${tok.diff_span_a ?? '-'}</span> &rarr; <span class="badge pole-b">${tok.diff_span_b ?? '-'}</span></td>
              <td>${p.valid === false ? `<span style="color:var(--danger);">${escapeHtml(p.error || 'Invalid pair')}</span>` : `<span style="color:var(--success);">${escapeHtml(p.section || 'Valid 3-section pair')}</span>`}</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    `;
    renderBanner('pairs-status-box', `Validated <strong>${pairs.length} pairs</strong>.`);
    setStatus('Pairs validated', 'ok');
  } catch (err) {
    renderBanner('pairs-status-box', `Validation error: ${escapeHtml(err.message)}`, true);
    setStatus('Validation failed', 'err');
  }
}

async function saveMarkdownPairsToToml() {
  try {
    const name = document.getElementById('pairs-name').value.trim();
    const axis = document.getElementById('pairs-axis').value.trim();
    const text = document.getElementById('pairs-markdown-editor').value.trim();
    if (!name || !text) throw new Error('Enter a Vector Name and Markdown pairs first.');
    setStatus(`Saving ${name}.toml...`, 'busy');
    const res = await apiPost('/api/save-toml', { name, axis, text, overwrite: true });
    await loadInitData();
    renderBanner('pairs-status-box', `Saved editable pack to <code>${escapeHtml(res.toml_path || res.toml || ('prompt_bank/control_vectors/' + name + '.toml'))}</code>.`);
    setStatus(`Saved ${name}.toml`, 'ok');
  } catch (err) {
    renderBanner('pairs-status-box', `Save .toml error: ${escapeHtml(err.message)}`, true);
    setStatus('Save error', 'err');
  }
}

async function buildFromMarkdownStudio() {
  try {
    const name = document.getElementById('pairs-name').value.trim();
    const axis = document.getElementById('pairs-axis').value.trim();
    const text = document.getElementById('pairs-markdown-editor').value.trim();
    if (!name || !text) throw new Error('Enter a Vector Name and Markdown pairs first.');
    setStatus(`Building ${name} on GPU...`, 'busy');
    renderBanner('pairs-status-box', `Encoding pairs on GPU for <strong>${escapeHtml(name)}</strong>...`);
    const res = await apiPost('/api/build', {
      name,
      axis,
      families: APP.families,
      text,
      save_toml: true,
      overwrite: true
    });
    await loadInitData();
    const files = (res.built || res.written || []).map(f => `<code>${escapeHtml(typeof f === 'string' ? f : f.file)}</code>`).join(', ');
    renderBanner('pairs-status-box', `<strong>Built Control Vector <code>${escapeHtml(name)}</code>!</strong><br/>Saved: ${files}`);
    setStatus(`Built ${name}`, 'ok');
  } catch (err) {
    renderBanner('pairs-status-box', `Build error: ${escapeHtml(err.message)}`, true);
    setStatus('Build error', 'err');
  }
}

/* ========================================================================= */
/* SCREEN 4: REFERENCE IMAGE CONTRAST                                        */
/* ========================================================================= */
function handleImageUpload(which, inputEl) {
  const file = inputEl.files && inputEl.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(ev) {
    const dataUrl = ev.target.result;
    if (which === 'a') {
      APP.mediaDataUrlA = dataUrl;
      const img = document.getElementById('media-preview-a');
      img.src = dataUrl;
      img.style.display = 'inline-block';
      document.getElementById('media-label-a').innerHTML = `Loaded <strong>${escapeHtml(file.name)}</strong> (${Math.round(file.size / 1024)} KB)`;
    } else {
      APP.mediaDataUrlB = dataUrl;
      const img = document.getElementById('media-preview-b');
      img.src = dataUrl;
      img.style.display = 'inline-block';
      document.getElementById('media-label-b').innerHTML = `Loaded <strong>${escapeHtml(file.name)}</strong> (${Math.round(file.size / 1024)} KB)`;
    }
  };
  reader.readAsDataURL(file);
}

async function buildMediaVector() {
  try {
    const name = document.getElementById('media-name').value.trim();
    const caption = document.getElementById('media-caption').value.trim();
    if (!name) throw new Error('Please enter an Output Vector Name.');
    if (!APP.mediaDataUrlA || !APP.mediaDataUrlB) {
      throw new Error('Please select both Image A (Baseline) and Image B (Target) in your browser.');
    }
    setStatus(`Extracting image vector ${name}...`, 'busy');
    renderBanner('media-status-box', `Running Qwen3-VL multimodal vision-language pass on Image A &amp; Image B...`);
    const res = await apiPost('/api/make-media', {
      name,
      output_name: name,
      families: APP.families,
      image_a: APP.mediaDataUrlA,
      image_b: APP.mediaDataUrlB,
      scene_prompt: caption,
      overwrite: true
    });
    await loadInitData();
    const files = (res.built || res.written || []).map(f => `<code>${escapeHtml(typeof f === 'string' ? f : f.file)}</code>`).join(', ');
    renderBanner('media-status-box', `<strong>Extracted Multimodal Control Vector <code>${escapeHtml(name)}</code>!</strong><br/>Saved: ${files}`);
    setStatus(`Built ${name}`, 'ok');
  } catch (err) {
    renderBanner('media-status-box', `Image vector error: ${escapeHtml(err.message)}`, true);
    setStatus('Media build failed', 'err');
  }
}

/* ========================================================================= */
/* SCREEN 5: SCULPT, INVERT & COMBINE                                        */
/* ========================================================================= */
async function runQuickInvert() {
  try {
    const src = document.getElementById('quick-invert-source').value;
    const outName = document.getElementById('quick-invert-name').value.trim();
    if (!src || !outName) throw new Error('Choose a source vector and enter a new name.');
    setStatus(`Inverting ${src}...`, 'busy');
    const res = await apiPost('/api/sculpt', {
      name: outName,
      output_name: outName,
      families: APP.families,
      steps: [{ vector: src, strength: -1.0, mode: 'add' }],
      overwrite: true
    });
    await loadInitData();
    const files = (res.built || res.saved || []).map(f => `<code>${escapeHtml(typeof f === 'string' ? f : f.file)}</code>`).join(', ');
    renderBanner('sculpt-status-box', `<strong>Saved inverted vector <code>${escapeHtml(outName)}</code> (-1.0 &times; ${escapeHtml(src)})!</strong><br/>Files: ${files}`);
    setStatus(`Saved ${outName}`, 'ok');
  } catch (err) {
    renderBanner('sculpt-status-box', `Invert error: ${escapeHtml(err.message)}`, true);
    setStatus('Invert failed', 'err');
  }
}

function renderSculptRows() {
  const container = document.getElementById('sculpt-rows-container');
  if (!container) return;
  const optionsHtml = APP.vectors.map(v => {
    const val = typeof v === 'string' ? v : (v.file || v.name);
    const label = typeof v === 'string' ? v : (`${v.name || v.file} (${v.family || 'fl2va'})`);
    return `<option value="${escapeHtml(val)}">${escapeHtml(label)}</option>`;
  }).join('');

  container.innerHTML = APP.sculptRows.map((row, idx) => `
    <div style="display:grid;grid-template-columns:2fr 1.4fr 1fr auto;gap:10px;align-items:center;margin-bottom:10px;background:var(--panel);padding:10px 12px;border:1px solid var(--border);border-radius:8px;">
      <select onchange="APP.sculptRows[${idx}].vector = this.value">
        ${optionsHtml}
      </select>
      <select onchange="APP.sculptRows[${idx}].mode = this.value">
        <option value="add" ${row.mode === 'add' ? 'selected' : ''}>add (weighted sum)</option>
        <option value="add_norm_matched" ${row.mode === 'add_norm_matched' ? 'selected' : ''}>add_norm_matched (match L2)</option>
        <option value="hold" ${row.mode === 'hold' ? 'selected' : ''}>hold (Gram-Schmidt orthogonalize)</option>
      </select>
      <input type="number" step="0.1" value="${row.strength}" onchange="APP.sculptRows[${idx}].strength = parseFloat(this.value || '0')" placeholder="Weight (e.g. 1.0 or -1.0)" />
      <div class="btn-row">
        <button type="button" class="btn btn-sm" onclick="APP.sculptRows[${idx}].strength = -1 * (APP.sculptRows[${idx}].strength || 1); renderSculptRows();">Flip &plusmn;</button>
        <button type="button" class="btn btn-sm" onclick="APP.sculptRows.splice(${idx}, 1); renderSculptRows();">Remove</button>
      </div>
    </div>
  `).join('');

  const selects = container.querySelectorAll('select:first-child');
  APP.sculptRows.forEach((row, idx) => {
    if (selects[idx] && row.vector) selects[idx].value = row.vector;
  });
}

function addSculptRow() {
  const firstVec = APP.vectors.length ? (typeof APP.vectors[0] === 'string' ? APP.vectors[0] : (APP.vectors[0].file || APP.vectors[0].name)) : 'film_16mm_1974';
  APP.sculptRows.push({ vector: firstVec, strength: 0.5, mode: 'add' });
  renderSculptRows();
}

async function runSculptRecipe() {
  try {
    const name = document.getElementById('sculpt-name').value.trim();
    const holdWords = document.getElementById('sculpt-hold-words').value.trim();
    if (!name) throw new Error('Enter an Output Sculpted Vector Name.');
    setStatus(`Sculpting ${name}...`, 'busy');
    const res = await apiPost('/api/sculpt', {
      name,
      output_name: name,
      families: APP.families,
      steps: APP.sculptRows,
      hold_words: holdWords,
      overwrite: true
    });
    await loadInitData();
    const files = (res.built || res.saved || []).map(f => `<code>${escapeHtml(typeof f === 'string' ? f : f.file)}</code>`).join(', ');
    renderBanner('sculpt-status-box', `<strong>Sculpted &amp; saved <code>${escapeHtml(name)}</code> in &lt; 5 ms CPU!</strong><br/>Saved: ${files}`);
    setStatus(`Sculpted ${name}`, 'ok');
  } catch (err) {
    renderBanner('sculpt-status-box', `Sculpt error: ${escapeHtml(err.message)}`, true);
    setStatus('Sculpt failed', 'err');
  }
}

/* ========================================================================= */
/* SCREEN 6: PROMPT RECEPTOR METER & SEPARATE J-LENS WORD LAB                */
/* ========================================================================= */
function fillMeterSamplePrompt() {
  document.getElementById('meter-prompt-text').value =
`A medium tracking shot on a 35mm anamorphic lens. A traveler in a weathered wool coat walks along a stone pier under low slate clouds as wind whips across the harbor water.

[00s-05s] The traveler pulls the collar tight against the gust while waves spray over the seawall.

Soundscape: Howling coastal wind, crashing water spray, and heavy boots on wet stone.`;
}

async function runPromptReceptorMeter() {
  try {
    const vec = document.getElementById('meter-vector-select').value;
    const prompt = document.getElementById('meter-prompt-text').value.trim();
    if (!vec || !prompt) throw new Error('Select a Control Vector and enter your H3 prompt.');
    const fam = APP.families.includes('fl2va') ? 'fl2va' : 'ref2va';
    setStatus(`Scanning prompt tokens...`, 'busy');
    const res = await apiPost('/api/prompt-meter', {
      vector: vec,
      family: fam,
      prompt
    });
    const card = document.getElementById('meter-results-card');
    const body = document.getElementById('meter-results-body');
    card.style.display = 'block';

    const tokens = res.tokens || [];
    const topAlignedWords = Array.from(new Set(
      tokens
        .filter(t => t.in_scene_mask && Math.abs(Number(t.cosine || 0)) >= 0.08)
        .sort((a, b) => Math.abs(Number(b.cosine || 0)) - Math.abs(Number(a.cosine || 0)))
        .map(t => String(t.token || t.text || '').trim().replace(/^[^\w]+|[^\w]+$/g, ''))
        .filter(w => w.length >= 3)
    )).slice(0, 8);
    const suggested = (res.suggested_mask_words || topAlignedWords).join(', ');
    body.innerHTML = `
      <div style="margin-bottom:14px;">
        <strong>Top receptor words in your prompt (for <code>MiniMaxH3ApplyControlVectors</code>):</strong>
        <input type="text" class="mono" readonly value="${escapeHtml(suggested)}" style="margin-top:6px;" onclick="this.select()" />
      </div>
      <div class="token-strip">
        ${tokens.map(t => {
          const score = Number(t.cosine ?? t.score ?? 0);
          let bg = 'rgba(255,255,255,0.04)';
          if (score > 0.08) bg = 'rgba(245, 158, 11, 0.28)';
          else if (score < -0.08) bg = 'rgba(16, 185, 129, 0.25)';
          return `<span class="tok-chip" style="background:${bg};" title="cosine: ${score.toFixed(4)}">${escapeHtml(t.token || t.text || '')}</span>`;
        }).join('')}
      </div>
    `;
    renderBanner('meter-status-box', `Scanned <strong>${tokens.length} contextual tokens</strong> against <code>${escapeHtml(vec)}</code> (${escapeHtml(res.elapsed_ms || '')} ms).`);
    setStatus('Prompt meter complete', 'ok');
  } catch (err) {
    renderBanner('meter-status-box', `Prompt meter error: ${escapeHtml(err.message)}`, true);
    setStatus('Meter failed', 'err');
  }
}

async function runJLensInspect() {
  try {
    const vec = document.getElementById('jlens-inspect-select').value;
    if (!vec) throw new Error('Select a vector first.');
    const fam = APP.families.includes('fl2va') ? 'fl2va' : 'ref2va';
    const data = await apiPost('/api/j-lens', { vector: vec, family: fam });
    const ro = data.readout || data.post_refiner || data;
    const posList = ro.positive_words || ro.top_positive || ro.positive || [];
    const negList = ro.negative_words || ro.top_negative || ro.negative || [];
    const pos = posList.slice(0, 12).map(w => `<span class="badge pole-b">${escapeHtml(w.word || w[0])} (${Number(w.gain ?? w.cosine ?? w[1] ?? 0).toFixed(2)})</span>`).join(' ');
    const neg = negList.slice(0, 12).map(w => `<span class="badge pole-a">${escapeHtml(w.word || w[0])} (${Number(w.gain ?? w.cosine ?? w[1] ?? 0).toFixed(2)})</span>`).join(' ');
    const rd = data.readiness || ro.readiness || {};
    const summaryHtml = rd.summary ? `<div style="margin-bottom:10px;font-size:12.5px;color:var(--text-secondary);" class="mono">${escapeHtml(rd.summary)}</div>` : '';
    document.getElementById('jlens-inspect-output').innerHTML = `
      ${summaryHtml}
      <div style="margin-bottom:10px;"><strong>Top Target Pole B Words (+gain):</strong><div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;">${pos || 'None'}</div></div>
      <div><strong>Top Baseline Pole A Words (-gain):</strong><div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;">${neg || 'None'}</div></div>
    `;
  } catch (err) {
    document.getElementById('jlens-inspect-output').innerHTML = `<div class="result-banner error">${escapeHtml(err.message)}</div>`;
  }
}

async function runJLensWordSynthesis(saveFile) {
  try {
    const name = document.getElementById('jlens-synth-name').value.trim();
    const posWords = document.getElementById('jlens-words-b').value.trim();
    const negWords = document.getElementById('jlens-words-a').value.trim();
    const holdWords = document.getElementById('jlens-words-hold').value.trim();
    const endpoint = saveFile ? '/api/build-words' : '/api/preview-words';
    const res = await apiPost(endpoint, {
      name,
      save_name: name,
      families: APP.families,
      from_words: posWords,
      against_words: negWords,
      hold_words: holdWords
    });
    if (saveFile) await loadInitData();
    const ro = res.readout || {};
    const posList = ro.positive_words || ro.top_positive || [];
    const topPos = posList.slice(0, 10).map(w => `<span class="badge pole-b">${escapeHtml(w.word || w[0])} (+${Number(w.gain ?? w.cosine ?? 0).toFixed(2)})</span>`).join(' ');
    document.getElementById('jlens-synth-output').innerHTML = `
      <div class="result-banner">
        <strong>${saveFile ? `Saved J-Lens prototype <code>${escapeHtml((res.saved_files || [name]).join(', '))}</code>!` : 'J-Lens Word-Bag Preview:'}</strong>
        <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;">${topPos || 'None'}</div>
      </div>
    `;
  } catch (err) {
    document.getElementById('jlens-synth-output').innerHTML = `<div class="result-banner error">${escapeHtml(err.message)}</div>`;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  fillMeterSamplePrompt();
  loadInitData();
});
