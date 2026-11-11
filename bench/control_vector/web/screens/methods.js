// Every method the CLI has, each as its own form. Whatever the screens above do not cover is
// here, and a subcommand added to the CLI appears here with no edit to the page.

import { studio, h } from '../api.js';
import { methodCard } from '../form.js';

/** A screen that is one method (the image contrast screen). */
export function mountOne(method, title, desc) {
  return (root) => {
    root.append(
      h('div', { class: 'screen-header' },
        h('h2', { class: 'screen-title' }, title),
        h('p', { class: 'screen-desc' }, desc)),
      methodCard(method, { title }).el);
  };
}

export function mount(root) {
  root.append(
    h('div', { class: 'screen-header' },
      h('h2', { class: 'screen-title' }, 'All methods'),
      h('p', { class: 'screen-desc' }, 'Each subcommand of bench/control_vectors.py, with every argument it takes.')),
    ...Object.keys(studio.methods).map((name) => {
      const holder = h('div', {});
      const open = h('details', { class: 'method-fold' }, h('summary', {}, h('code', {}, name), ' ', studio.methods[name].help), holder);
      // Drawn when first opened: a dozen forms at once is slow to scan and to build.
      open.addEventListener('toggle', () => {
        if (open.open && !holder.firstChild) holder.append(methodCard(name).el);
      });
      return open;
    }));
}
