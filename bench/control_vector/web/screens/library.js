// The library: the vectors and pair files on disk, and `rebuild`.

import { studio, onListing, h } from '../api.js';
import { methodCard } from '../form.js';

function num(value) {
  return typeof value === 'number' ? value.toFixed(3) : '';
}

export function mount(root, nav) {
  const filter = h('input', { type: 'text', placeholder: 'filter by name, family or pooling' });
  const vecBody = h('tbody');
  const tomlBody = h('tbody');
  const count = h('span', { class: 'field-hint' });

  const draw = () => {
    const q = filter.value.trim().toLowerCase();
    const rows = studio.vectors.filter((v) => !q || `${v.file} ${v.family} ${v.pooling}`.toLowerCase().includes(q));
    count.textContent = `${rows.length} of ${studio.vectors.length} vector file(s) in ${studio.init.vec_dir}`;
    vecBody.replaceChildren(...rows.map((v) => h('tr', {},
      h('td', { class: 'mono' }, v.file),
      h('td', {}, v.family),
      h('td', {}, v.pooling),
      h('td', {}, `${v.n_pairs} / ${v.n_holds}`),
      h('td', {}, num(v.raw_norm)),
      h('td', {}, num(v.mpc)),
      h('td', {}, num(v.split_half)),
      h('td', {}, (v.differing_sections || []).join(', ')),
      h('td', {}, v.rec_strength),
      h('td', {},
        h('button', { type: 'button', class: 'btn btn-sm', onclick: () => nav.go('inspect', { vector: v.file }) }, 'Inspect'),
        ' ',
        h('button', { type: 'button', class: 'btn btn-sm', onclick: () => nav.go('sculpt', { vector: v.file }) }, 'Sculpt')))));
    tomlBody.replaceChildren(...studio.tomlFiles.map((t) => h('tr', {},
      h('td', { class: 'mono' }, t.file),
      h('td', {}, t.mode),
      h('td', {}, `${t.n_pairs} / ${t.n_holds}`),
      h('td', {}, t.quality),
      h('td', {}, h('button', { type: 'button', class: 'btn btn-sm', onclick: () => nav.go('pairs', { pairFile: t.stem }) }, 'Open')))));
  };
  filter.addEventListener('input', draw);
  onListing(draw);

  const head = (titles) => h('thead', {}, h('tr', {}, titles.map((t) => h('th', {}, t))));
  root.append(
    h('div', { class: 'screen-header' },
      h('h2', { class: 'screen-title' }, 'Library'),
      h('p', { class: 'screen-desc' }, 'The control vectors and pair files the studio reads and writes.')),
    h('section', { class: 'card' },
      h('div', { class: 'card-header' }, h('h3', { class: 'card-title' }, 'Vectors'), count),
      filter,
      h('div', { class: 'table-scroll' }, h('table', { class: 'lib-table' },
        head(['file', 'family', 'pooling', 'pairs / holds', 'norm', 'pairwise cosine', 'split half', 'sections that differ', 'recommended strength', '']),
        vecBody))),
    h('section', { class: 'card' },
      h('div', { class: 'card-header' }, h('h3', { class: 'card-title' }, 'Pair files'),
        h('span', { class: 'field-hint' }, `${studio.init.toml_dir} and the pack's bank`)),
      h('div', { class: 'table-scroll' }, h('table', { class: 'lib-table' },
        head(['file', 'mode', 'pairs / holds', 'quality', '']), tomlBody))),
    methodCard('rebuild', { title: 'Rebuild the library' }).el);
  draw();
}
