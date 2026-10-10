// Turret Control — browser side. Talks to the laptop app (turret_ui.py) over /ws,
// which relays to turretd on the Pi. The Pi's watchdog needs our heartbeat every
// 100 ms; if this tab dies or loses focus, movement stops.
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : String(k));
  return e;
};
const fmt = (x, d = 1) => (x == null || Number.isNaN(x) ? '—' : Number(x).toFixed(d));

const AIM_AXES = ['pan', 'tilt'];
const S = {
  ws: null, uiUp: false, piUp: false, piHost: '', nextId: 1, pending: new Map(),
  schema: null, values: {}, presets: [], tele: null,
  held: new Set(), clickMode: null, fov: null,
};

// ── connection ────────────────────────────────────────────────────────────
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  S.ws = ws;
  ws.onopen = () => { S.uiUp = true; renderLinks(); };
  ws.onclose = () => {
    S.uiUp = false; S.piUp = false; renderLinks();
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    onMessage(m);
  };
}

function send(type, payload = {}, { quiet = false } = {}) {
  if (!S.ws || S.ws.readyState !== WebSocket.OPEN) {
    if (!quiet) log('error', `not connected — ${type} not sent`);
    return Promise.reject(new Error('not connected'));
  }
  const id = S.nextId++;
  S.ws.send(JSON.stringify({ type, id, ...payload }));
  return new Promise((resolve, reject) => {
    S.pending.set(id, { resolve, reject, type, quiet });
    setTimeout(() => { if (S.pending.delete(id)) reject(new Error('timeout')); }, 5000);
  });
}

function sendRaw(type, payload = {}) {   // fire-and-forget, no ack
  if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify({ type, ...payload }));
}

setInterval(() => sendRaw('heartbeat'), 100);

function onMessage(m) {
  switch (m.type) {
    case 'telemetry': S.tele = m; renderTelemetry(); break;
    case 'event': log(m.level, m.msg, m.t); break;
    case 'ack': { const p = S.pending.get(m.id); if (p) { S.pending.delete(m.id); p.resolve(m.result); } break; }
    case 'error': {
      const p = S.pending.get(m.id);
      if (p) { S.pending.delete(m.id); p.reject(new Error(m.msg)); }
      log('error', m.cmd ? `${m.cmd}: ${m.msg}` : m.msg);
      break;
    }
    case 'hello':
      S.values = m.config; S.presets = m.presets || [];
      for (const e of m.load_errors || []) log('warn', `config: ${e}`);
      loadSchema();
      break;
    case 'config': Object.assign(S.values, m.values); onConfigChanged(Object.keys(m.values), true); break;
    case 'presets': S.presets = m.presets; renderPresets(); break;
    case 'link':
      S.piUp = m.pi; S.piHost = m.host;
      if (!m.pi) { S.tele = null; renderTelemetry(); }
      renderLinks(m.error);
      if (m.pi) startVideo();
      break;
  }
}

function renderLinks(error) {
  const ui = $('#link-ui'), pi = $('#link-pi');
  ui.className = 'pill ' + (S.uiUp ? 'up' : 'down');
  pi.className = 'pill ' + (S.piUp ? 'up' : 'down');
  pi.textContent = S.piHost ? `Pi ${S.piHost}` : 'Pi';
  const b = $('#banner');
  if (!S.uiUp) { b.hidden = false; b.textContent = 'Laptop app not reachable — is turret_ui.py running?'; }
  else if (!S.piUp) {
    b.hidden = false;
    b.textContent = `Pi not connected (${S.piHost || '?'})${error ? ': ' + error : ''} — retrying…`;
  } else b.hidden = true;
}

// ── video ───────────────────────────────────────────────────────────────────
let videoRetry = null;
function startVideo() {
  const img = $('#cam');
  clearTimeout(videoRetry);
  img.src = `/video.mjpg?t=${Date.now()}`;
}
$('#cam').addEventListener('error', () => {
  clearTimeout(videoRetry);
  videoRetry = setTimeout(() => { if (S.piUp) startVideo(); }, 2000);
});

function videoPoint(ev) {
  const r = $('#video').getBoundingClientRect();
  return { x: ((ev.clientX - r.left) / r.width) * 2 - 1, y: ((ev.clientY - r.top) / r.height) * 2 - 1 };
}

$('#video').addEventListener('click', (ev) => {
  const p = videoPoint(ev);
  if (S.clickMode === 'crosshair') {
    setConfig({ 'camera.crosshair_x': round3(p.x), 'camera.crosshair_y': round3(p.y) });
    setClickMode(null);
    return;
  }
  if (S.clickMode === 'fov') { fovPick(p); return; }
  const cx = S.values['camera.crosshair_x'] || 0, cy = S.values['camera.crosshair_y'] || 0;
  send('aim_offset', { x: round3(p.x - cx), y: round3(p.y - cy) }).catch(() => {});
});
const round3 = (x) => Math.round(x * 1000) / 1000;

function setClickMode(mode, text) {
  S.clickMode = mode;
  const m = $('#click-mode');
  m.hidden = !mode;
  m.textContent = text || '';
}

function renderCrosshair() {
  const x = S.values['camera.crosshair_x'] || 0, y = S.values['camera.crosshair_y'] || 0;
  const c = $('#crosshair');
  c.style.left = `${50 + x * 50}%`;
  c.style.top = `${50 + y * 50}%`;
  $('#cal-cross-val').textContent = `x ${fmt(x, 3)}  y ${fmt(y, 3)}`;
}

// ── keyboard ────────────────────────────────────────────────────────────────
const KEY_DIRS = {
  KeyW: ['tilt', 1], ArrowUp: ['tilt', 1], KeyS: ['tilt', -1], ArrowDown: ['tilt', -1],
  KeyA: ['pan', -1], ArrowLeft: ['pan', -1], KeyD: ['pan', 1], ArrowRight: ['pan', 1],
};
let mods = { shift: false, alt: false };
const typing = (t) => t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA');

function speedMode() { return mods.shift ? 'fast' : mods.alt ? 'fine' : 'normal'; }

const lastMove = { pan: '0', tilt: '0' };
function updateMoves(force = false) {
  // Only send what changed, so a key on one axis never stops a goto on the other.
  for (const axis of AIM_AXES) {
    let dir = 0;
    for (const code of S.held) {
      const k = KEY_DIRS[code];
      if (k && k[0] === axis) dir += k[1];
    }
    dir = Math.sign(dir);
    const speed = speedMode();
    const sig = dir === 0 ? '0' : `${dir}/${speed}`;
    if (!force && sig === lastMove[axis]) continue;
    lastMove[axis] = sig;
    sendRaw('move', { axis, dir, speed });
  }
}

function releaseAll() {
  const hadMoves = [...S.held].some((c) => KEY_DIRS[c]);
  const hadFire = S.held.has('Space');
  S.held.clear();
  mods = { shift: false, alt: false };
  if (hadMoves) updateMoves();
  if (hadFire) sendRaw('fire_stop');
  setFireActive(false);
}

window.addEventListener('keydown', (ev) => {
  if (ev.code === 'Escape') { ev.preventDefault(); estop(); return; }
  if (typing(ev.target)) return;
  const modChanged = (ev.key === 'Shift' && !mods.shift) || (ev.key === 'Alt' && !mods.alt);
  mods = { shift: ev.shiftKey, alt: ev.altKey };
  if (ev.key === 'Alt') ev.preventDefault();
  if (modChanged && [...S.held].some((c) => KEY_DIRS[c])) updateMoves();
  if (ev.repeat) { if (KEY_DIRS[ev.code] || ev.code === 'Space') ev.preventDefault(); return; }

  if (KEY_DIRS[ev.code]) {
    ev.preventDefault();
    S.held.add(ev.code);
    updateMoves();
  } else if (ev.code === 'Space') {
    ev.preventDefault();
    S.held.add('Space');
    fireDown();
  } else if (ev.code === 'KeyF') toggleSpin();
  else if (ev.code === 'KeyR') toggleArm();
  else if (ev.code === 'Digit1') setConfig({ 'fire.mode': 'single' });
  else if (ev.code === 'Digit2') setConfig({ 'fire.mode': 'burst' });
  else if (ev.code === 'Digit3') setConfig({ 'fire.mode': 'auto' });
});

window.addEventListener('keyup', (ev) => {
  const modChanged = (ev.key === 'Shift' && mods.shift) || (ev.key === 'Alt' && mods.alt);
  mods = { shift: ev.shiftKey, alt: ev.altKey };
  if (KEY_DIRS[ev.code] && S.held.delete(ev.code)) updateMoves();
  else if (ev.code === 'Space' && S.held.delete('Space')) fireUp();
  else if (modChanged && [...S.held].some((c) => KEY_DIRS[c])) updateMoves();
});

window.addEventListener('blur', releaseAll);
document.addEventListener('visibilitychange', () => { if (document.hidden) releaseAll(); });

// ── actions ─────────────────────────────────────────────────────────────────
function estop() {
  if (S.tele && S.tele.state === 'ESTOP') return;
  releaseAll();
  sendRaw('estop', { reason: 'E-STOP pressed' });
}

$('#estop').addEventListener('click', () => {
  if (S.tele && S.tele.state === 'ESTOP') send('reset').catch(() => {});
  else estop();
});

function toggleArm() {
  const st = S.tele && S.tele.state;
  send(st === 'ARMED' || st === 'ARMING' ? 'disarm' : 'arm').catch(() => {});
}
function toggleSpin() {
  send('spin', { on: !(S.tele && S.tele.flywheels.on) }).catch(() => {});
}
function fireDown() { setFireActive(true); send('fire_start').catch(() => setFireActive(false)); }
function fireUp() { setFireActive(false); sendRaw('fire_stop'); }
function setFireActive(on) { $('#btn-fire').classList.toggle('active', on); }

$('#btn-arm').addEventListener('click', toggleArm);
$('#btn-spin').addEventListener('click', toggleSpin);
$('#btn-reload').addEventListener('click', () => send('reload').catch(() => {}));
const fireBtn = $('#btn-fire');
fireBtn.addEventListener('pointerdown', (ev) => { fireBtn.setPointerCapture(ev.pointerId); fireDown(); });
fireBtn.addEventListener('pointerup', fireUp);
fireBtn.addEventListener('pointercancel', fireUp);
for (const b of document.querySelectorAll('#mode button')) {
  b.setAttribute('role', 'radio');
  b.addEventListener('click', () => setConfig({ 'fire.mode': b.dataset.mode }));
}

// Quick sliders for frequently tuned parameters; they edit the same config
// values as the Config tab and stay in sync with it (and with other UIs).
const SLIDERS = {};
function slider(parent, key, label, { min, max, step, digits = 0, unit = '' }) {
  const input = el('input', { type: 'range', min, max, step, 'aria-label': label });
  const out = el('output');
  let timer = null;
  const show = () => { out.textContent = fmt(input.value, digits) + unit; };
  input.addEventListener('input', () => {
    show();
    clearTimeout(timer);
    timer = setTimeout(() => setConfig({ [key]: Number(input.value) }), 120);
  });
  SLIDERS[key] = { input, show };
  $(parent).append(el('label', { class: 'slider' }, el('span', {}, label), input, out));
}
slider('#fly-sliders', 'flywheel.throttle', 'Throttle', { min: 0, max: 100, step: 1, unit: '%' });
slider('#fly-sliders', 'flywheel.trim', 'Trim L/R', { min: -20, max: 20, step: 0.5, digits: 1, unit: '%' });
slider('#speed-sliders', 'pan.speed', 'Pan', { min: 5, max: 180, step: 1, unit: '°/s' });
slider('#speed-sliders', 'tilt.speed', 'Tilt', { min: 5, max: 180, step: 1, unit: '°/s' });
slider('#speed-sliders', 'loader.push_speed', 'Loader push', { min: 30, max: 2160, step: 10, unit: '°/s' });
slider('#speed-sliders', 'loader.return_speed', 'Loader return', { min: 30, max: 2160, step: 10, unit: '°/s' });

function syncSliders() {
  for (const [key, { input, show }] of Object.entries(SLIDERS)) {
    if (document.activeElement === input || S.values[key] === undefined) continue;
    input.value = S.values[key];
    show();
  }
  const thr = SLIDERS['flywheel.throttle'].input;
  thr.max = S.values['flywheel.max_throttle'];
  // Fastest possible cycle with the current loader settings (ramp up + down each way).
  const v = S.values, a = v['loader.accel'], stroke = v['loader.stroke'];
  const leg = (sp) => stroke / sp + sp / a;
  const cycle = leg(v['loader.push_speed']) + leg(v['loader.return_speed']) + v['loader.front_dwell'] / 1000;
  const rate = Math.min(1 / cycle, v['loader.rate']);
  $('#rate-info').textContent = `Loader cycle ≈ ${fmt(cycle * 1000, 0)} ms → up to ${fmt(rate, 1)} darts/s`
    + (1 / cycle > v['loader.rate'] ? ` (capped by fire rate ${v['loader.rate']})` : '');
}

// ── config ──────────────────────────────────────────────────────────────────
async function api(method, path, body) {
  const r = await fetch(path, {
    method, headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

async function setConfig(changes) {
  try {
    await api('PATCH', '/api/config', changes);
  } catch (e) { log('error', `config: ${e.message}`); }
}

async function loadSchema() {
  try {
    S.schema = await api('GET', '/api/schema');
  } catch (e) { log('error', `schema: ${e.message}`); return; }
  renderConfig();
  renderPresets();
  onConfigChanged(Object.keys(S.values));
}

function paramInput(p) {
  const v = S.values[p.key];
  let input;
  if (p.type === 'bool') {
    input = el('input', { type: 'checkbox', id: 'p-' + p.key });
    input.checked = !!v;
  } else if (p.type === 'enum') {
    input = el('select', { id: 'p-' + p.key }, p.choices.map((c) => el('option', { value: c }, String(c))));
    input.value = String(v);
  } else {
    input = el('input', { type: 'number', id: 'p-' + p.key, min: p.min, max: p.max, step: p.step || 'any' });
    input.value = v;
  }
  input.addEventListener('change', () => {
    let val = p.type === 'bool' ? input.checked : input.value;
    if (p.type === 'enum') val = p.choices.find((c) => String(c) === input.value);
    setConfig({ [p.key]: val });
  });
  return input;
}

function renderConfig() {
  const root = $('#config-groups');
  root.replaceChildren();
  const byGroup = {};
  for (const p of S.schema.params) (byGroup[p.group] ||= []).push(p);
  for (const [group, label] of Object.entries(S.schema.groups)) {
    const params = byGroup[group] || [];
    const box = el('div', { class: 'params' });
    for (const p of params) {
      const tag = p.apply === 'restart' ? 'restart turretd' : p.apply === 'camera' ? 'restarts camera' : '';
      const row = el('div', { class: 'param', 'data-key': p.key },
        el('label', { for: 'p-' + p.key }, p.label, tag && el('span', { class: 'tag' }, tag)),
        el('div', { class: 'ctl' }, paramInput(p), p.unit && el('span', { class: 'unit' }, p.unit)),
        el('button', { class: 'revert', title: `Reset to default (${p.default})`,
          onclick: () => setConfig({ [p.key]: p.default }) }, '↺'),
        p.help && el('div', { class: 'help' }, p.help));
      box.append(row);
    }
    root.append(el('details', { class: 'group', open: group === 'flywheel' },
      el('summary', {}, label), box));
  }
}

function syncParamRow(key, flash) {
  const row = document.querySelector(`.param[data-key="${CSS.escape(key)}"]`);
  if (!row || !S.schema) return;
  const p = S.schema.params.find((q) => q.key === key);
  const input = row.querySelector('input, select');
  const v = S.values[key];
  if (input !== document.activeElement) {
    if (p.type === 'bool') input.checked = !!v; else input.value = String(v);
  }
  row.classList.toggle('changed', v !== p.default);
  if (flash) { row.classList.remove('flash'); void row.offsetWidth; row.classList.add('flash'); }
}

function onConfigChanged(keys, flash = false) {
  for (const k of keys) syncParamRow(k, flash);
  const v = S.values;
  syncSliders();
  for (const b of document.querySelectorAll('#mode button')) b.setAttribute('aria-checked', b.dataset.mode === v['fire.mode']);
  renderCrosshair();
  for (const b of document.querySelectorAll('#aim-mode button')) b.setAttribute('aria-checked', b.dataset.mode === v['aim.mode']);
  $('#dist-wrap').hidden = v['aim.mode'] !== 'follow';
  if (document.activeElement !== $('#dist')) $('#dist').value = v['aim.target_distance'];
  $('#dist-val').textContent = `${fmt(v['aim.target_distance'], 1)} m`;
  const res = String(v['camera.resolution'] || '1280x720').split('x').map(Number);
  $('#video').style.aspectRatio = `${res[0]} / ${res[1]}`;
}

function renderPresets() {
  const sel = $('#preset-select');
  sel.replaceChildren(...(S.presets.length ? S.presets.map((n) => el('option', { value: n }, n))
    : [el('option', { value: '' }, 'no presets yet')]));
}
$('#preset-save').addEventListener('click', async () => {
  const name = $('#preset-name').value.trim();
  if (!name) return;
  try { await api('POST', `/api/presets/${encodeURIComponent(name)}`); log('info', `preset "${name}" saved`); $('#preset-name').value = ''; }
  catch (e) { log('error', `preset: ${e.message}`); }
});
$('#preset-load').addEventListener('click', async () => {
  const name = $('#preset-select').value;
  if (name) try { await api('POST', `/api/presets/${encodeURIComponent(name)}/load`); } catch (e) { log('error', `preset: ${e.message}`); }
});
$('#preset-delete').addEventListener('click', async () => {
  const name = $('#preset-select').value;
  if (name && confirm(`Delete preset "${name}"?`)) {
    try { await api('DELETE', `/api/presets/${encodeURIComponent(name)}`); } catch (e) { log('error', `preset: ${e.message}`); }
  }
});

// ── axes (control, calibrate, diagnostics) ─────────────────────────────────
function axisCard(axis, kind) {
  const c = el('div', { class: 'axis', 'data-axis': axis },
    el('div', { class: 'name' }, axis, el('span', { class: 'badge ref' })),
    el('div', { class: 'pos' }, '—'),
    el('div', { class: 'meta' }));
  const btns = el('div', { class: 'btns' });
  const cmd = (type, extra) => () => send(type, { axis, ...extra }).catch(() => {});
  if (kind === 'control' && AIM_AXES.includes(axis)) {
    const [neg, pos] = axis === 'pan' ? ['◀', '▶'] : ['▼', '▲'];
    btns.append(
      el('button', { title: 'Jog (step from config)', onclick: cmd('jog', { dir: -1 }) }, neg),
      el('button', { title: 'Jog', onclick: cmd('jog', { dir: 1 }) }, pos),
      el('button', { onclick: cmd('goto', { deg: 0 }) }, 'Go to 0'));
  } else if (kind === 'calibrate') {
    const [neg, pos] = axis === 'pan' ? ['◀', '▶'] : ['▼', '▲'];
    btns.append(
      el('button', { onclick: cmd('jog', { dir: -1, speed: 'fine' }) }, neg + ' fine'),
      el('button', { onclick: cmd('jog', { dir: 1, speed: 'fine' }) }, pos + ' fine'),
      el('button', { onclick: cmd('jog', { dir: -1 }) }, neg),
      el('button', { onclick: cmd('jog', { dir: 1 }) }, pos),
      el('button', { onclick: cmd('set_zero') }, 'Set zero here'),
      el('button', { onclick: cmd('set_limit', { which: 'min' }) }, 'Set min here'),
      el('button', { onclick: cmd('set_limit', { which: 'max' }) }, 'Set max here'));
  } else if (kind === 'diag') {
    for (const d of [-90, -10, -1, 1, 10, 90]) {
      btns.append(el('button', { onclick: cmd('raw_jog', { deg: d }) }, (d > 0 ? '+' : '') + d + '°'));
    }
    const en = el('button', { class: 'en', onclick: () => {
      const on = !(S.tele && S.tele.axes[axis].enabled);
      send('enable', { axis, on }).catch(() => {});
    } }, 'Enable');
    btns.append(en);
  }
  c.append(btns);
  return c;
}

$('#axes').append(...['pan', 'tilt', 'loader'].map((a) => axisCard(a, 'control')));
$('#cal-axes').append(...AIM_AXES.map((a) => axisCard(a, 'calibrate')));
$('#diag-axes').append(...['pan', 'tilt', 'loader'].map((a) => axisCard(a, 'diag')));
$('#cal-loader-jog').append(...[-90, -10, -1, 1, 10, 90].map((d) =>
  el('button', { onclick: () => send('raw_jog', { axis: 'loader', deg: d }).catch(() => {}) },
    (d > 0 ? '+' : '') + d + '°')));
$('#cal-rest').addEventListener('click', () => send('loader_set_rest').catch(() => {}));
$('#cal-front').addEventListener('click', () => send('loader_set_stroke').catch(() => {}));
$('#cal-ready').addEventListener('click', () => send('loader_set_ready').catch(() => {}));
$('#cal-test').addEventListener('click', () => send('loader_test').catch(() => {}));
$('#cal-cross').addEventListener('click', () => setClickMode('crosshair', 'Click where the dart hit'));
$('#cal-cross-reset').addEventListener('click', () => setConfig({ 'camera.crosshair_x': 0, 'camera.crosshair_y': 0 }));

function renderAxisCards(t) {
  for (const card of document.querySelectorAll('.axis')) {
    const axis = card.dataset.axis;
    const a = t && t.axes[axis];
    const pos = $('.pos', card), meta = $('.meta', card), ref = $('.ref', card);
    if (!a) { pos.textContent = '—'; meta.textContent = ''; ref.textContent = ''; continue; }
    pos.textContent = `${fmt(a.pos, 1)}°`;
    ref.textContent = a.referenced ? (axis === 'loader' ? 'at rest = 0' : 'referenced') : 'not referenced';
    ref.className = 'badge ref ' + (a.referenced ? 'ok' : 'warn');
    const lim = a.limits ? `limits ${fmt(a.limits[0], 1)}…${fmt(a.limits[1], 1)}°` : 'no limits';
    meta.textContent = `${a.enabled ? 'enabled' : 'disabled'} · ${a.moving ? `moving ${fmt(a.vel, 0)}°/s` : 'idle'} · ${axis === 'loader' ? `stroke ${fmt(S.values['loader.stroke'], 0)}°` : lim} · ${a.steps} µst`;
    const en = $('.en', card);
    if (en) en.textContent = a.enabled ? 'Disable' : 'Enable';
  }
}

// ── click-aim mode and tracker status ───────────────────────────────────────
for (const b of document.querySelectorAll('#aim-mode button')) {
  b.addEventListener('click', () => setConfig({ 'aim.mode': b.dataset.mode }));
}
$('#dist').addEventListener('input', () => { $('#dist-val').textContent = `${fmt(Number($('#dist').value), 1)} m`; });
$('#dist').addEventListener('change', () => setConfig({ 'aim.target_distance': Number($('#dist').value) }));
$('#follow-stop').addEventListener('click', () => send('aim_cancel').catch(() => {}));

let lastAimMsg = '';
function renderAim(a) {
  const hud = $('#hud-track'), mark = $('#track-mark');
  if (!a) {
    hud.hidden = true; mark.hidden = true; $('#follow-stop').hidden = true; $('#lock').hidden = true;
    $('#btn-fire').classList.remove('on-target');
    return;
  }
  const recent = a.t && Date.now() / 1000 - a.t < 8;
  hud.hidden = !(a.busy || (recent && a.msg));
  hud.textContent = a.msg;
  hud.className = `hud hud-br ${a.state}`;
  mark.hidden = !(a.point && (a.busy || (recent && a.state === 'locked')));
  if (a.point) {
    mark.style.left = `${50 + a.point[0] * 50}%`;
    mark.style.top = `${50 + a.point[1] * 50}%`;
  }
  mark.classList.toggle('busy', !!a.busy && a.busy !== 'follow');
  const following = !!a.following;
  mark.classList.toggle('follow', following);
  mark.classList.toggle('on', following && !!a.on_target);
  if (following) mark.hidden = !a.point;
  $('#follow-stop').hidden = !following;
  $('#lock').hidden = !following;
  $('#lock').classList.toggle('on', following && !!a.on_target);
  $('#btn-fire').classList.toggle('on-target', following && !!a.on_target);
  if (a.msg && a.msg !== lastAimMsg && !a.busy && a.state !== 'idle') {
    if (a.state === 'failed' || a.state === 'done') $('#fov-info').textContent = a.msg;
  }
  lastAimMsg = a.msg;
}

$('#fov-auto').addEventListener('click', () => {
  send('fov_calibrate', { axis: $('#fov-axis').value })
    .then(() => { $('#fov-info').textContent = 'Calibrating… keep the scene still.'; })
    .catch((e) => { $('#fov-info').textContent = e.message; });
});

// ── field of view wizard ────────────────────────────────────────────────────
// Pinhole camera: a point at fraction u of the half-frame sits atan(u·tan(fov/2))
// off-axis. Find the smallest fov for which points a and b are `deg` apart.
function solveFov(a, b, deg) {
  const sep = (fov) => {
    const k = Math.tan((fov * Math.PI) / 360);
    return (Math.abs(Math.atan(a * k) - Math.atan(b * k)) * 180) / Math.PI;
  };
  for (let fov = 10; fov <= 170; fov += 0.05) if (sep(fov) >= deg) return fov;
  return null;
}

$('#fov-start').addEventListener('click', () => {
  S.fov = { axis: $('#fov-axis').value, deg: Number($('#fov-deg').value) || 10, a: null };
  setClickMode('fov', 'Click a small, sharp feature');
  $('#fov-info').textContent = 'Step 1: click a feature in the video.';
});

async function fovPick(p) {
  const f = S.fov;
  if (!f) return;
  if (f.a === null) {
    f.a = p;
    setClickMode(null);
    $('#fov-info').textContent = `Turning ${f.axis} by ${f.deg}°…`;
    try { await send('jog', { axis: f.axis, dir: 1, deg: f.deg }); } catch { S.fov = null; return; }
    setTimeout(() => {
      setClickMode('fov', 'Click the same feature again');
      $('#fov-info').textContent = 'Step 2: click the same feature again.';
    }, 1500);
    return;
  }
  setClickMode(null);
  const [a, b] = f.axis === 'pan' ? [f.a.x, p.x] : [f.a.y, p.y];
  S.fov = null;
  if (Math.abs(b - a) < 0.02) { $('#fov-info').textContent = 'The feature barely moved — try a larger angle.'; return; }
  const fov = Math.round(solveFov(a, b, f.deg) * 10) / 10;
  if (!fov) { $('#fov-info').textContent = 'No field of view fits — check gear ratio and microsteps.'; return; }
  const key = f.axis === 'pan' ? 'camera.hfov' : 'camera.vfov';
  await setConfig({ [key]: fov });
  $('#fov-info').textContent = `${f.axis === 'pan' ? 'Horizontal' : 'Vertical'} field of view = ${fov}° (saved).`;
  send('jog', { axis: f.axis, dir: -1, deg: f.deg }).catch(() => {});
}

// ── telemetry rendering ─────────────────────────────────────────────────────
function renderTelemetry() {
  const t = S.tele;
  const stateEl = $('#state');
  stateEl.textContent = t ? (t.state === 'ESTOP' ? 'E-STOP' : t.state) : 'NO LINK';
  stateEl.className = 'state ' + (t ? t.state : '');
  const es = $('#estop');
  const latched = t && t.state === 'ESTOP';
  es.textContent = latched ? 'RESET E-STOP' : 'E-STOP';
  es.classList.toggle('reset', !!latched);

  renderAxisCards(t);
  renderAim(t && t.aim);
  if (!t) { $('#hud-aim').textContent = ''; $('#hud-fire').textContent = ''; return; }

  const sys = t.system || {};
  const temp = sys.cpu_temp;
  $('#sys').replaceChildren(
    el('span', { class: temp > 75 ? 'hot' : '' }, temp != null ? `CPU ${fmt(temp, 0)}°C` : ''),
    sys.throttled && sys.throttled.length ? el('span', { class: 'hot' }, ` ⚠ ${sys.throttled[0]}`) : '',
    t.heartbeat_age_ms != null ? ` · hb ${t.heartbeat_age_ms} ms` : '');

  const armed = t.state === 'ARMED' || t.state === 'ARMING';
  const arm = $('#btn-arm');
  arm.textContent = t.state === 'ARMING' ? 'ARMING…' : armed ? 'DISARM' : 'ARM';
  arm.classList.toggle('on', armed);
  arm.disabled = latched;
  const fw = t.flywheels;
  const spin = $('#btn-spin');
  spin.textContent = fw.on ? (fw.spun_up ? 'Flywheels ready' : 'Spinning up…') : 'Flywheels off';
  spin.classList.toggle('on', fw.on);
  spin.disabled = t.state !== 'ARMED';
  for (const [side, v] of [['l', fw.left], ['r', fw.right]]) {
    const bar = $('#bar-' + side);
    bar.style.width = `${v}%`;
    bar.classList.toggle('ready', fw.spun_up);
    $('#val-' + side).textContent = `${fmt(v, 0)}%`;
  }
  const fireBtn = $('#btn-fire');
  fireBtn.disabled = !t.fire.action;
  fireBtn.textContent = t.fire.action === 'load' ? 'LOAD' : 'FIRE';
  fireBtn.classList.toggle('load', t.fire.action === 'load');
  if (!t.fire.firing && !S.held.has('Space')) setFireActive(false);
  $('#ammo').textContent = t.fire.ammo == null ? '∞' : `${t.fire.ammo} / ${t.fire.magazine}`;
  $('#shots').textContent = t.fire.shots;

  const p = t.axes.pan, ti = t.axes.tilt;
  $('#hud-aim').textContent = `PAN  ${fmt(p.pos, 1).padStart(6)}°${p.referenced ? '' : ' ?'}\nTILT ${fmt(ti.pos, 1).padStart(6)}°${ti.referenced ? '' : ' ?'}`;
  $('#hud-fire').textContent = `${t.state === 'ESTOP' ? 'E-STOP' : t.state}\n${t.fire.mode.toUpperCase()} · ${t.fire.ammo == null ? '∞' : t.fire.ammo}${t.fire.firing ? ' · FIRING' : ''}`;

  const cam = t.camera;
  const msg = $('#cam-msg');
  if (cam && cam.error) { msg.hidden = false; msg.textContent = cam.error; } else msg.hidden = true;
  $('#hud-cam').textContent = cam ? `${cam.size[0]}×${cam.size[1]} · ${fmt(cam.fps, 0)} fps` : 'no camera';

  if (!$('#tab-diag').hidden) renderDiag(t);
  $('#cal-loader-info').textContent = `Loader at ${fmt(t.axes.loader.pos, 1)}° · stroke ${fmt(S.values['loader.stroke'], 0)}°`
    + ` · ready ${S.values['loader.ready'] > 0 ? fmt(S.values['loader.ready'], 0) + '°' : 'off'}`;
}

function renderDiag(t) {
  const kv = [
    ['motion worker', `${t.motion.alive ? 'alive' : 'DEAD'} (${t.motion.backend})`],
    ['loop max gap', `${t.motion.loop_max_gap_us} µs`],
    ['steps total', t.motion.steps_total],
    ['camera', t.camera ? `${t.camera.backend} · ${t.camera.frames} frames · ${t.camera.error || 'ok'}` : 'off'],
    ['CPU temp', t.system.cpu_temp != null ? `${t.system.cpu_temp} °C` : 'n/a'],
    ['throttling', t.system.throttled ? (t.system.throttled.join(', ') || 'none') : 'n/a'],
    ['load', t.system.load],
    ['heartbeat age', `${t.heartbeat_age_ms} ms`],
    ['ESC pulses', `L ${fmt(t.flywheels.left, 1)}% · R ${fmt(t.flywheels.right, 1)}%`],
    ['restart needed', t.restart_needed.length ? t.restart_needed.join(', ') : 'no'],
  ];
  $('#diag-sys').replaceChildren(...kv.flatMap(([k, v]) => [el('dt', {}, k), el('dd', {}, String(v))]));
  if ($('#diag-raw').closest('details').open) $('#diag-raw').textContent = JSON.stringify(t, null, 1);
}

// ── tabs & log ──────────────────────────────────────────────────────────────
for (const b of document.querySelectorAll('.tabs button')) {
  b.addEventListener('click', () => {
    for (const o of document.querySelectorAll('.tabs button')) o.setAttribute('aria-selected', o === b);
    for (const tab of document.querySelectorAll('.tab')) tab.hidden = tab.id !== 'tab-' + b.dataset.tab;
  });
}

function log(level, msg, t) {
  const box = $('#log');
  const time = new Date(t ? t * 1000 : Date.now()).toLocaleTimeString();
  const line = el('div', { class: level }, el('span', { class: 't' }, time), msg);
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
  box.append(line);
  while (box.childElementCount > 300) box.firstChild.remove();
  if (atBottom) box.scrollTop = box.scrollHeight;
}

renderLinks();
renderTelemetry();
connect();
