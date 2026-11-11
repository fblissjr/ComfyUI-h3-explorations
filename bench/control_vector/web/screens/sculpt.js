// Composing saved vectors into one (`sculpt`).
//
// The recipe rows stand for three of the method's arguments: the vectors, `--modes` and
// `--strengths`. The mode list is the node's own (`control_vectors.APPLY_MODES`), whole; a mode
// `sculpt` cannot compose is refused by the method, with its reason shown.

import { studio, onListing, apiPost, h } from '../api.js';
import { methodCard } from '../form.js';
import { readoutView } from '../readout.js';

export function mount(root, nav) {
  const spec = studio.methods.sculpt;
  const arg = (dest) => spec.args.find((a) => a.dest === dest);
  const firstOf = (dest) => String(arg(dest).default || '').split(',')[0].trim();
  const rows = [];
  const body = h('div', { class: 'stack' });

  function draw() {
    body.replaceChildren(...rows.map((row, idx) => {
      const vec = h('select', {}, studio.vectors.map((v) => h('option', { value: v.file }, v.file)));
      if (row.vector) vec.value = row.vector; else row.vector = vec.value;
      vec.addEventListener('change', () => { row.vector = vec.value; });
      const mode = h('select', {}, studio.node.apply_modes.map((m) => h('option', { value: m }, m)));
      mode.value = row.mode;
      mode.addEventListener('change', () => { row.mode = mode.value; });
      const strength = h('input', { type: 'number', step: 'any', class: 'mono' });
      strength.value = row.strength;
      strength.addEventListener('input', () => { row.strength = strength.value; });
      const drop = h('button', { type: 'button', class: 'btn btn-sm' }, 'Remove');
      drop.addEventListener('click', () => { rows.splice(idx, 1); draw(); });
      return h('div', { class: 'recipe-row' }, vec, mode, strength, drop);
    }));
  }
  function addRow(vector) {
    rows.push({ vector: vector || '', mode: firstOf('modes'), strength: firstOf('strengths') });
    draw();
  }
  const addBtn = h('button', { type: 'button', class: 'btn btn-sm' }, 'Add a vector');
  addBtn.addEventListener('click', () => addRow());
  onListing(draw);

  const card = methodCard('sculpt', {
    title: 'Sculpt',
    runLabel: 'Save the vector',
    exclude: ['vectors', 'modes', 'strengths'],
    extra: () => ({
      vectors: rows.map((r) => r.vector),
      modes: rows.map((r) => r.mode).join(','),
      strengths: rows.map((r) => r.strength).join(','),
    }),
    before: [
      h('div', { class: 'field' },
        h('label', { class: 'field-label' }, h('code', {}, '<vectors> --modes --strengths'),
          h('span', { class: 'help-tip', tabindex: '0' }, '?',
            h('span', { class: 'tip-box' }, `${arg('vectors').help} ${arg('modes').help} ${arg('strengths').help}`))),
        body,
        h('div', { class: 'btn-row' }, addBtn)),
    ],
    actions: [{
      label: 'Preview',
      async run(values, panel) {
        const res = await apiPost('/api/preview-recipe', { args: values });
        panel.note(res.families.map((f) => readoutView(f.readout, f.family)));
      },
    }],
  });
  root.append(
    h('div', { class: 'screen-header' },
      h('h2', { class: 'screen-title' }, 'Sculpt'),
      h('p', { class: 'screen-desc' }, 'Add, scale, invert or hold out saved vectors, and save the result as a new one.')),
    card.el);

  nav.on('sculpt', (params) => {
    if (params && params.vector) addRow(params.vector);
    else if (!rows.length) addRow();
  });
}
