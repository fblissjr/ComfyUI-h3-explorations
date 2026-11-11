// Looking at a saved vector: its word readout, a prompt metered against it, `inspect` and `grade`.

import { studio, onListing, apiPost, h, setStatus } from '../api.js';
import { methodCard, resultPanel } from '../form.js';
import { readoutView } from '../readout.js';

function vectorPicker() {
  const sel = h('select', {});
  const fill = () => {
    const keep = sel.value;
    sel.replaceChildren(...studio.vectors.map((v) => h('option', { value: v.file }, v.file)));
    if (keep) sel.value = keep;
  };
  fill();
  onListing(fill);
  return sel;
}

function guarded(btn, panel, text, fn) {
  btn.addEventListener('click', async () => {
    btn.disabled = true;
    panel.busy(text);
    setStatus(text, 'busy');
    try {
      await fn();
      setStatus('Ready', 'ok');
    } catch (err) {
      panel.error(err);
      setStatus('Refused', 'err');
    } finally {
      btn.disabled = false;
    }
  });
}

export function mount(root, nav) {
  // --- word readout -------------------------------------------------------------------------
  const readVec = vectorPicker();
  const readPanel = resultPanel();
  const readBtn = h('button', { type: 'button', class: 'btn btn-primary' }, 'Read out');
  guarded(readBtn, readPanel, 'Reading out...', async () => {
    const res = await apiPost('/api/j-lens', { vector: readVec.value });
    readPanel.note(readoutView(res.readout, `${res.name || readVec.value} (${res.family || ''})`));
  });

  // --- prompt meter -------------------------------------------------------------------------
  const meterVec = vectorPicker();
  const meterPrompt = h('textarea', { class: 'mono', rows: '8', placeholder: 'the prompt you are about to render' });
  // The node's masks and the node's default, from the schema: the meter scores the rows the
  // node would steer.
  const meterMask = h('select', {}, studio.node.mask_modes.map((m) => h('option', { value: m }, m)));
  meterMask.value = studio.node.defaults.mask;
  const meterPanel = resultPanel();
  const meterBtn = h('button', { type: 'button', class: 'btn btn-primary' }, 'Meter the prompt');
  guarded(meterBtn, meterPanel, 'Metering (loads the encoder)...', async () => {
    const res = await apiPost('/api/prompt-meter', { vector: meterVec.value, prompt: meterPrompt.value, mask: meterMask.value });
    const tokens = res.tokens || [];
    meterPanel.note(
      h('div', { class: 'field-hint' }, `${tokens.length} token(s), mask ${res.pooling}. Amber leans toward the vector, green away; hover for the cosine.`),
      h('div', { class: 'token-strip' }, tokens.map((t) => {
        const score = Number(t.cosine ?? t.score ?? 0);
        const cls = score > 0.08 ? 'tok-chip tok-toward' : score < -0.08 ? 'tok-chip tok-away' : 'tok-chip';
        return h('span', { class: cls, title: `cosine ${score.toFixed(4)}` }, t.token || t.text || '');
      })));
  });

  root.append(
    h('div', { class: 'screen-header' },
      h('h2', { class: 'screen-title' }, 'Inspect'),
      h('p', { class: 'screen-desc' }, 'What a saved vector reads as, how a prompt sits against it, and the two measuring methods.')),
    h('section', { class: 'card' },
      h('div', { class: 'card-header' }, h('h3', { class: 'card-title' }, 'Word readout')),
      h('p', { class: 'field-hint' }, 'The words the vector leans toward and away from, by J-Lens. CPU only.'),
      h('div', { class: 'btn-row' }, readVec, readBtn),
      readPanel.el),
    h('section', { class: 'card' },
      h('div', { class: 'card-header' }, h('h3', { class: 'card-title' }, 'Prompt meter')),
      h('p', { class: 'field-hint' }, 'Each token of a prompt scored against the vector, over the rows the chosen mask steers.'),
      h('div', { class: 'form-grid' },
        h('div', { class: 'field' }, h('label', { class: 'field-label' }, 'vector'), meterVec),
        h('div', { class: 'field' }, h('label', { class: 'field-label' }, 'mask'), meterMask),
        h('div', { class: 'field field-wide' }, h('label', { class: 'field-label' }, 'prompt'), meterPrompt)),
      h('div', { class: 'btn-row' }, meterBtn),
      meterPanel.el),
    methodCard('inspect', { title: 'Inspect' }).el,
    methodCard('grade', { title: 'Grade' }).el);

  nav.on('inspect', (params) => {
    if (params && params.vector) {
      readVec.value = params.vector;
      meterVec.value = params.vector;
      readBtn.click();
    }
  });
}
