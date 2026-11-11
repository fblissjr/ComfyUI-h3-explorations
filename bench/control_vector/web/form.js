// Draws a CLI method's controls from the server's description of it (GET /api/schema).
//
// Nothing here knows a method, an argument, a choice or a default: each comes from the entry the
// server sends, which is the CLI's parser walked. `values()` returns `{dest: value}` for POST
// /api/run, where the same parser reads it.

import { studio, onListing, h, runMethod, setStatus } from './api.js';

let uid = 0;

function label(arg) {
  const flag = arg.positional ? `<${arg.dest}>` : arg.flags[0];
  const parts = [h('code', {}, flag)];
  if (arg.required) parts.push(h('span', { class: 'badge hold' }, 'required'));
  if (arg.help) {
    parts.push(h('span', { class: 'help-tip', tabindex: '0' }, '?', h('span', { class: 'tip-box' }, arg.help)));
  }
  return h('label', { class: 'field-label' }, parts);
}

function datalist(optionsOf) {
  const list = h('datalist', { id: `dl-${++uid}` });
  const fill = () => list.replaceChildren(...optionsOf().map((value) => h('option', { value })));
  fill();
  onListing(fill);
  return list;
}

const vectorNames = () => studio.vectors.map((v) => v.file);
const pairFileNames = () => studio.tomlFiles.map((t) => t.stem);

// One control per argument kind. Each returns {el, get(), set(value)}; get() returns undefined
// when the control is empty, which leaves the argument off and the parser's default in place.
function control(arg) {
  if (arg.kind === 'bool') {
    const box = h('input', { type: 'checkbox', disabled: arg.fixed });
    box.checked = Boolean(arg.default);
    return {
      el: h('label', { class: 'check-row' }, box, arg.fixed ? ' always on (the command line has no flag to turn it off)' : ' on'),
      get: () => (arg.fixed ? undefined : box.checked),
      set: (v) => { if (!arg.fixed) box.checked = Boolean(v); },
    };
  }
  if (arg.kind === 'choice') {
    const sel = h('select', {},
      arg.default === null || arg.default === undefined ? h('option', { value: '' }, '(not set)') : null,
      arg.choices.map((c) => h('option', { value: c }, c)));
    sel.value = arg.default ?? '';
    return { el: sel, get: () => sel.value || undefined, set: (v) => { sel.value = v ?? ''; } };
  }
  if (arg.kind === 'int' || arg.kind === 'float') {
    const num = h('input', { type: 'number', step: arg.kind === 'int' ? '1' : 'any', class: 'mono' });
    num.value = arg.default ?? '';
    return {
      el: num,
      get: () => (num.value === '' ? undefined : Number(num.value)),
      set: (v) => { num.value = v ?? ''; },
    };
  }
  if (arg.kind === 'list') {
    const area = h('textarea', { class: 'mono list-area', rows: '3', placeholder: 'one per line' });
    area.value = (arg.default || []).join('\n');
    const parts = [area];
    if (arg.widget === 'vectors') {
      const pick = h('select', {}, h('option', { value: '' }, 'add a vector from the library...'));
      const fill = () => pick.replaceChildren(
        h('option', { value: '' }, 'add a vector from the library...'),
        ...vectorNames().map((v) => h('option', { value: v }, v)));
      fill();
      onListing(fill);
      pick.addEventListener('change', () => {
        if (!pick.value) return;
        area.value = [area.value.trim(), pick.value].filter(Boolean).join('\n');
        pick.value = '';
      });
      parts.push(pick);
    }
    const lines = () => area.value.split('\n').map((s) => s.trim()).filter(Boolean);
    return {
      el: h('div', { class: 'stack' }, parts),
      get: () => (lines().length ? lines() : undefined),
      set: (v) => { area.value = (Array.isArray(v) ? v : [v]).filter(Boolean).join('\n'); },
    };
  }
  if (arg.widget === 'image') {
    // The method takes a path or a data URL; a picture chosen here is sent as a data URL.
    const path = h('input', { type: 'text', class: 'mono', placeholder: 'a path on this machine, or choose a picture' });
    const file = h('input', { type: 'file', accept: 'image/*' });
    const thumb = h('img', { class: 'thumb', alt: '' });
    let dataUrl = '';
    file.addEventListener('change', () => {
      const f = file.files && file.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => {
        dataUrl = String(reader.result || '');
        thumb.src = dataUrl;
        thumb.style.display = 'block';
        path.value = '';
        path.placeholder = `chosen: ${f.name}`;
      };
      reader.readAsDataURL(f);
    });
    path.addEventListener('input', () => { dataUrl = ''; thumb.style.display = 'none'; });
    return {
      el: h('div', { class: 'stack' }, path, file, thumb),
      get: () => dataUrl || path.value.trim() || undefined,
      set: (v) => { path.value = v ?? ''; dataUrl = ''; },
    };
  }
  if (arg.widget === 'textarea') {
    const area = h('textarea', { class: 'mono', rows: '5' });
    area.value = arg.default ?? '';
    return { el: area, get: () => (area.value.trim() ? area.value : undefined), set: (v) => { area.value = v ?? ''; } };
  }
  const input = h('input', { type: 'text', class: 'mono' });
  input.value = arg.default ?? '';
  const extra = [];
  const suggestions = arg.suggest
    ? () => arg.suggest
    : arg.widget === 'vector' ? vectorNames : arg.widget === 'pairfile' ? pairFileNames : null;
  if (suggestions) {
    const list = datalist(suggestions);
    input.setAttribute('list', list.id);
    extra.push(list);
  }
  return {
    el: extra.length ? h('div', { class: 'stack' }, input, extra) : input,
    get: () => (input.value.trim() ? input.value : undefined),
    set: (v) => { input.value = v ?? ''; },
  };
}

/**
 * Build the form for `method`.
 *   exclude: dests another control on the screen supplies (they come back through `extra`)
 *   preset:  {dest: value} to start from in place of the parser's default
 * Returns {el, values(), set(dest, value), get(dest), has(dest)}.
 */
export function buildForm(method, { exclude = [], preset = {} } = {}) {
  const spec = studio.methods[method];
  if (!spec) throw new Error(`the server describes no method ${method}`);
  const controls = new Map();
  const primary = h('div', { class: 'form-grid' });
  const more = h('div', { class: 'form-grid' });
  let hidden = 0;
  for (const arg of spec.args) {
    if (exclude.includes(arg.dest)) continue;
    const ctl = control(arg);
    controls.set(arg.dest, ctl);
    if (arg.dest in preset) ctl.set(preset[arg.dest]);
    const wide = arg.widget === 'textarea' || arg.kind === 'list' || arg.widget === 'image';
    const field = h('div', { class: `field${wide ? ' field-wide' : ''}`, 'data-dest': arg.dest }, label(arg), ctl.el);
    if (arg.primary) primary.append(field);
    else { more.append(field); hidden += 1; }
  }
  const el = h('div', { class: 'method-form', 'data-method': method },
    primary,
    hidden ? h('details', { class: 'more-options' }, h('summary', {}, `More options (${hidden})`), more) : null);
  return {
    el,
    method,
    values() {
      const out = {};
      for (const [dest, ctl] of controls) {
        const value = ctl.get();
        if (value !== undefined) out[dest] = value;
      }
      return out;
    },
    set(dest, value) { if (controls.has(dest)) controls.get(dest).set(value); },
    get(dest) { return controls.has(dest) ? controls.get(dest).get() : undefined; },
    has(dest) { return controls.has(dest); },
  };
}

function cell(value) {
  if (value === null || value === undefined || value === '') return '';
  return typeof value === 'number' ? value.toFixed(3) : String(value);
}

/** The result of a run: the command it stood for, what it built, and what it printed. */
export function resultPanel() {
  const el = h('div', { class: 'run-result' });
  const fill = (...kids) => el.replaceChildren(...kids.filter(Boolean));
  return {
    el,
    busy(text) { fill(h('div', { class: 'result-banner' }, text)); },
    error(err) {
      const data = err && err.data;
      fill(
        h('div', { class: 'result-banner error' }, err.message || String(err)),
        data && data.log ? h('pre', { class: 'run-log' }, data.log) : null);
    },
    note(...children) { fill(h('div', { class: 'result-banner' }, children)); },
    show(res) {
      const built = res.built || [];
      fill(
        h('div', { class: 'result-banner' },
          h('strong', {}, res.rc === 0 ? 'Done.' : `Finished with code ${res.rc}.`),
          res.command ? h('div', { class: 'mono command-line' }, res.command) : null),
        built.length ? h('table', { class: 'lib-table' },
          h('thead', {}, h('tr', {}, ['file', 'family', 'pooling', 'norm', 'pairwise cosine', 'split half', 'recommended strength'].map((t) => h('th', {}, t)))),
          h('tbody', {}, built.map((row) => h('tr', {},
            h('td', { class: 'mono' }, cell(row.file)), h('td', {}, cell(row.family)), h('td', {}, cell(row.pooling)),
            h('td', {}, cell(row.raw_norm)), h('td', {}, cell(row.mpc)), h('td', {}, cell(row.split_half)),
            h('td', {}, cell(row.rec_strength)))))) : null,
        res.log ? h('details', { class: 'more-options', open: built.length === 0 },
          h('summary', {}, 'What the method printed'), h('pre', { class: 'run-log' }, res.log)) : null);
    },
  };
}

/**
 * A card that is one CLI method: its form, a button that runs it, and the result.
 *   extra():   more {dest: value} from controls the screen draws itself
 *   actions:   [{label, run(values, panel)}] buttons beside Run (previews)
 */
export function methodCard(method, { title, blurb, runLabel, exclude, preset, extra, actions = [], before = [] } = {}) {
  const spec = studio.methods[method];
  const form = buildForm(method, { exclude, preset });
  const panel = resultPanel();
  const collect = () => ({ ...form.values(), ...(extra ? extra() : {}) });
  const runBtn = h('button', { type: 'button', class: 'btn btn-primary' }, runLabel || `Run ${method}`);
  const guard = (btn, text, fn) => async () => {
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
  };
  runBtn.addEventListener('click', guard(runBtn, `Running ${method}...`, async () => panel.show(await runMethod(method, collect()))));
  const buttons = actions.map((action) => {
    const btn = h('button', { type: 'button', class: 'btn' }, action.label);
    btn.addEventListener('click', guard(btn, `${action.label}...`, () => action.run(collect(), panel)));
    return btn;
  });
  const el = h('section', { class: 'card', 'data-card': method },
    h('div', { class: 'card-header' },
      h('h3', { class: 'card-title' }, title || method),
      h('code', { class: 'method-tag' }, [method, ...(spec.aliases || [])].join(' / '))),
    h('p', { class: 'field-hint' }, blurb || spec.help),
    before,
    form.el,
    h('div', { class: 'btn-row' }, buttons, runBtn),
    panel.el);
  return { el, form, panel, collect };
}
