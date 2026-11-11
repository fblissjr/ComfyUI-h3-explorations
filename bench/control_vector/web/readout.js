// Shared renderers for what the previews return: a J-Lens word readout, and compiled pairs.

import { h } from './api.js';

function wordBadges(words, cls) {
  const list = (words || []).slice(0, 12);
  if (!list.length) return h('span', { class: 'field-hint' }, 'none');
  return list.map((w) => {
    const word = w.word ?? w[0];
    const gain = Number(w.gain ?? w.cosine ?? w[1] ?? 0);
    return h('span', { class: `badge ${cls}` }, `${word} (${gain >= 0 ? '+' : ''}${gain.toFixed(2)})`);
  });
}

/** The words a vector reads as, toward and away. `readout` is J-Lens's own report. */
export function readoutView(readout, heading) {
  const ro = readout || {};
  const readiness = ro.readiness || {};
  return h('div', { class: 'readout' },
    heading ? h('div', { class: 'readout-head' }, heading) : null,
    readiness.summary ? h('div', { class: 'mono field-hint' }, readiness.summary) : null,
    h('div', { class: 'readout-row' }, h('strong', {}, 'Toward: '), wordBadges(ro.positive_words || ro.top_positive, 'pole-b')),
    h('div', { class: 'readout-row' }, h('strong', {}, 'Away from: '), wordBadges(ro.negative_words || ro.top_negative, 'pole-a')),
    ro.prompt_coupling && ro.prompt_coupling.summary
      ? h('div', { class: 'mono field-hint' }, `prompt coupling: ${ro.prompt_coupling.summary}`) : null);
}

/** The pairs a contrast compiles to, as the build will encode them. */
export function pairsView(res) {
  return h('div', {},
    h('div', { class: 'field-hint' },
      `${res.pairs.length} pair(s), ${res.holds.length} hold(s), mode ${res.mode}, name ${res.name}. `,
      `Sections that differ: ${(res.differing_sections || []).join(', ') || 'none found'}.`),
    res.pairs.map((p) => h('div', { class: 'pair-card' },
      h('div', { class: 'pair-card-head' },
        h('span', {}, h('span', { class: 'badge' }, `#${p.index + 1}`), ` ${p.section || ''}`),
        h('span', { class: 'field-hint' }, `weight ${p.weight}`)),
      h('div', { class: 'grid-2' },
        h('div', { class: 'pole-a-box' }, h('span', { class: 'badge pole-a' }, 'A'), h('div', { class: 'mono span-text' }, p.a_span)),
        h('div', { class: 'pole-b-box' }, h('span', { class: 'badge pole-b' }, 'B'), h('div', { class: 'mono span-text' }, p.b_span))),
      h('details', { class: 'more-options' }, h('summary', {}, 'Both prompts in full'),
        h('div', { class: 'grid-2' }, h('pre', { class: 'run-log' }, p.a), h('pre', { class: 'run-log' }, p.b))))));
}
