// The studio's calls to its server, and the little shared state the screens read.
//
// Every route named here is in `ROUTES` in bench/control_vector/ui_server.py;
// bench/check_control_vector_studio.py holds the two lists to each other.

export const studio = {
  init: {},
  methods: {},
  node: { apply_modes: [], mask_modes: [], defaults: {} },
  vectors: [],
  tomlFiles: [],
};

const listeners = new Set();

/** Call `fn` whenever the vector or pair file listing changes. */
export function onListing(fn) {
  listeners.add(fn);
}

/** Take the listing a response carries, if it carries one. */
export function absorb(res) {
  if (!res) return res;
  let changed = false;
  if (Array.isArray(res.vectors)) { studio.vectors = res.vectors; changed = true; }
  if (Array.isArray(res.toml_files)) { studio.tomlFiles = res.toml_files; changed = true; }
  if (changed) listeners.forEach((fn) => fn());
  return res;
}

async function readJson(resp) {
  let data;
  try {
    data = await resp.json();
  } catch (err) {
    throw new Error(`the server answered ${resp.status} with no JSON`);
  }
  if (!resp.ok || data.ok === false) {
    const failure = new Error(data.error || `the server answered ${resp.status}`);
    failure.data = data;
    throw failure;
  }
  return absorb(data);
}

export async function apiGet(url) {
  return readJson(await fetch(url));
}

export async function apiPost(url, payload) {
  return readJson(await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  }));
}

/** Run a CLI method with `{argument: value}`, as the command line would. */
export function runMethod(method, args) {
  return apiPost('/api/run', { method, args });
}

export async function loadStudio() {
  const [init, schema] = await Promise.all([apiGet('/api/init'), apiGet('/api/schema')]);
  studio.init = init;
  studio.methods = schema.methods;
  studio.node = schema.node;
  return studio;
}

export function setStatus(text, kind = '') {
  const el = document.getElementById('status-indicator');
  if (!el) return;
  el.textContent = text;
  el.className = `status-pill ${kind}`.trim();
}

/**
 * `h('div', {class: 'card'}, child, 'text', ...)`: a small element builder. Children that are
 * strings become text nodes, so nothing a file, a prompt or the server says is ever read as markup.
 */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2), value);
    else if (value === true) el.setAttribute(key, '');
    else el.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}
