// A vector from a few words (`words`): read out before saving, then saved.

import { apiPost, h } from '../api.js';
import { methodCard } from '../form.js';
import { readoutView } from '../readout.js';

export function mount(root) {
  const card = methodCard('words', {
    title: 'From words',
    runLabel: 'Save the vector',
    actions: [{
      label: 'Preview',
      async run(values, panel) {
        const res = await apiPost('/api/preview-words', { args: values });
        panel.note(res.families.map((f) => h('div', {},
          readoutView(f.readout, `${f.family}: matched ${f.pos_words_matched.join(', ') || 'nothing'}`
            + (f.pos_words_unmatched.length ? `; not in the vocabulary: ${f.pos_words_unmatched.join(', ')}` : '')),
          h('div', { class: 'field-hint' }, `recommended strength: ${f.rec_strength}`))));
      },
    }],
  });
  root.append(
    h('div', { class: 'screen-header' },
      h('h2', { class: 'screen-title' }, 'Words'),
      h('p', { class: 'screen-desc' }, 'A vector synthesized from a few words on the CPU, with no prompt and no pair file.')),
    card.el);
}
