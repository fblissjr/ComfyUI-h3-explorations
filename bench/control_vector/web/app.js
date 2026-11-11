// Entry: load the schema and the listing, draw the navigation, mount the screens.

import { apiPost, loadStudio, setStatus, studio, h } from './api.js';
import * as library from './screens/library.js';
import * as pairs from './screens/pairs.js';
import * as words from './screens/words.js';
import * as sculpt from './screens/sculpt.js';
import * as inspect from './screens/inspect.js';
import * as methods from './screens/methods.js';

const SCREENS = [
  { key: 'library', title: 'Library', sub: 'Vectors and pair files on disk', mount: library.mount },
  { key: 'pairs', title: 'Pairs', sub: 'From a prompt changed in place, or a pair file', mount: pairs.mount },
  { key: 'images', title: 'Images', sub: 'From two contrasting pictures',
    mount: methods.mountOne('make-media', 'Images', 'A vector from two pictures that differ in one look, read through the same neutral prompt.') },
  { key: 'words', title: 'Words', sub: 'From a few words, on the CPU', mount: words.mount },
  { key: 'sculpt', title: 'Sculpt', sub: 'Compose, invert and hold out saved vectors', mount: sculpt.mount },
  { key: 'inspect', title: 'Inspect', sub: 'Readouts, the prompt meter, grading', mount: inspect.mount },
  { key: 'methods', title: 'All methods', sub: 'Every subcommand, every argument', mount: methods.mount },
];

const arrivals = new Map();
const nav = {
  /** Show a screen, handing it `params` (a vector to inspect, a pair file to open). */
  go(key, params) {
    document.querySelectorAll('.screen').forEach((el) => el.classList.toggle('active', el.dataset.screen === key));
    document.querySelectorAll('.nav-btn').forEach((el) => el.classList.toggle('active', el.dataset.screen === key));
    const arrive = arrivals.get(key);
    if (arrive) arrive(params || null);
  },
  /** A screen registers what to do when it is shown. */
  on(key, fn) { arrivals.set(key, fn); },
};

async function start() {
  const rail = document.getElementById('nav-rail');
  const workspace = document.getElementById('workspace');
  try {
    await loadStudio();
  } catch (err) {
    setStatus('Not connected', 'err');
    workspace.append(h('div', { class: 'result-banner error' }, `The studio's server did not answer: ${err.message}`));
    return;
  }
  for (const screen of SCREENS) {
    const section = h('section', { class: 'screen', 'data-screen': screen.key });
    workspace.append(section);
    try {
      screen.mount(section, nav);
    } catch (err) {
      section.append(h('div', { class: 'result-banner error' }, `This screen could not be drawn: ${err.message}`));
    }
    rail.append(h('button', { type: 'button', class: 'nav-btn', 'data-screen': screen.key, onclick: () => nav.go(screen.key) },
      h('span', { class: 'nav-btn-title' }, screen.title),
      h('span', { class: 'nav-btn-sub' }, screen.sub)));
  }
  document.getElementById('quit-btn').addEventListener('click', async () => {
    await apiPost('/api/quit', {});
    setStatus('Server stopped', 'err');
  });
  nav.go('library');
  setStatus(`Ready, ${studio.vectors.length} vectors`, 'ok');
}

start();
