/* lawrag console.
 *
 * Vanilla ES2020, no build step, no dependencies. That is a deployment
 * constraint rather than a stylistic one: the target workstation is air-gapped,
 * so anything npm would have fetched has to be carried across by hand, and a
 * console that cannot be repaired on the machine it runs on is not much of a
 * console.
 *
 * Structure: a hash router (§router) picks one of five views (§views), each of
 * which renders into #view. Shared pieces — the question composer, status
 * chips, the SVG charts — live in §components. There is no client-side state
 * beyond the current route and a small cache of /api/stats.
 */
'use strict';

/* ============================================================== utilities */

/** h('div.card', {onclick}, ...children) — enough of a builder to stay honest. */
function h(sel, attrs, ...kids) {
  const [tag, ...classes] = String(sel).split('.');
  const node = document.createElement(tag || 'div');
  if (classes.length) node.className = classes.join(' ');
  // Arrays and nodes in the second slot are children, not attributes — the
  // alternative is remembering to write h('tbody', {}, rows) everywhere, and
  // the one time it is forgotten the rows silently vanish.
  const isAttrs = attrs !== null && typeof attrs === 'object'
    && !Array.isArray(attrs) && !(attrs instanceof Node);
  if (!isAttrs) { if (attrs !== undefined) kids.unshift(attrs); attrs = null; }
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className += (node.className ? ' ' : '') + v;
    else if (k === 'html') node.innerHTML = v;
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (k in node && k !== 'list' && k !== 'type' && k !== 'form') node[k] = v;
    else node.setAttribute(k, v);
  }
  add(node, kids);
  return node;
}

function add(node, kids) {
  for (const kid of kids.flat(4)) {
    if (kid === null || kid === undefined || kid === false) continue;
    node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
}

function icon(id, cls) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  if (cls) svg.setAttribute('class', cls);
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', '#' + id);
  svg.append(use);
  return svg;
}

const svgEl = (tag, attrs) => {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined) continue;
    node.setAttribute(k, v);
  }
  return node;
};

const fmt = new Intl.NumberFormat('ru-RU');
const n = (v) => (v === null || v === undefined ? '—' : fmt.format(v));

function ms(v) {
  if (v === null || v === undefined) return '—';
  return v < 1000 ? `${v} мс` : `${(v / 1000).toFixed(v < 10000 ? 2 : 1)} с`;
}

function isoDay(d) { return d.toISOString().slice(0, 10); }

function shortDate(s) {
  if (!s) return '—';
  if (s.startsWith('9999')) return 'бессрочно';
  const d = new Date(s);
  if (Number.isNaN(+d)) return s;
  return d.toLocaleDateString('ru-RU', { day: '2-digit', month: 'short', year: 'numeric' });
}

function whenAgo(iso) {
  const d = new Date(iso);
  if (Number.isNaN(+d)) return iso || '—';
  const secs = (Date.now() - d) / 1000;
  if (secs < 90) return 'только что';
  if (secs < 3600) return `${Math.round(secs / 60)} мин назад`;
  if (secs < 86400) return `${Math.round(secs / 3600)} ч назад`;
  return d.toLocaleString('ru-RU', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
}

/** Human-readable act codes are the exception, not the rule; keep the raw id
 *  visible everywhere so an operator can always grep the database for it. */
const ACT_KIND = {
  constitution: 'Конституция', constitutional_law: 'Конституционный закон',
  code: 'Кодекс', law: 'Закон', presidential_decree: 'Указ Президента',
  government_resolution: 'Постановление Правительства', ministerial_order: 'Приказ',
};

/* ------------------------------------------------------------------- api */

class ApiError extends Error {
  constructor(status, detail) { super(detail); this.status = status; }
}

async function api(path, { method = 'GET', body } = {}) {
  const res = await fetch('/api' + path, {
    method,
    headers: body ? { 'content-type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) throw new ApiError(res.status, (data && (data.detail || data.message)) || res.statusText);
  return data;
}

/** The answering endpoints are not under /api — they are the CLI-mirroring
 *  surface and their paths are part of the project's contract. */
async function core(path, { method = 'GET', body } = {}) {
  const res = await fetch(path, {
    method,
    headers: body ? { 'content-type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) throw new ApiError(res.status, (data && (data.detail || data.message)) || res.statusText);
  return data;
}

/* =========================================================== components */

function chip(label, tone, { mono = false, title = '' } = {}) {
  return h('span.chip' + (tone ? '.' + tone : '') + (mono ? '.mono' : ''),
    { title }, h('i'), label);
}

function eyebrow(text) { return h('div.eyebrow', text); }

/** "koap/st571/p7" -> "п. 7". The article node id is the prefix, so anything
 *  after it is the part of the article that actually matched. */
function pointLabel(src) {
  const node = String(src.node_id || '');
  const article = String(src.article_node_id || '');
  if (!article || node === article || !node.startsWith(article + '/')) return '';
  return node.slice(article.length + 1)
    .replace(/^p(\d+)/, 'п. $1')
    .replace(/^sp(\d+)/, 'подп. $1');
}

function loading(label = 'Загрузка…') {
  return h('div.loading', h('span.spinner'), label);
}

function empty(title, body) {
  return h('div.empty', h('h3', title), body ? h('div', body) : null);
}

/** Two different 503s reach this function — the database is down, or the
 *  answering stack would not load — and they need different remedies, so the
 *  heading is chosen from what the server actually said rather than from the
 *  status code alone. */
function errorBox(err) {
  const detail = String(err.message || err);
  const status = err instanceof ApiError ? err.status : 0;
  const isDb = status === 503 && detail.startsWith('database unavailable');
  const isStack = detail.startsWith('answering stack unavailable');

  let heading = 'Ошибка запроса. ';
  let remedy = null;
  if (isDb) {
    heading = 'База данных недоступна. ';
    remedy = h('div', { style: 'margin-top:6px' },
      'Проверьте доступность сервера и значение ', h('span.mono', 'LAWRAG_DB_DSN'),
      '.');
  } else if (isStack) {
    heading = 'Модели не загружены. ';
    remedy = h('div', { style: 'margin-top:6px' },
      'В окружении, из которого запущен сервер, нет нужного пакета. ',
      'Установите стек весов — ',
      h('span.mono', 'lawrag[models]'), ', затем перезапустите сервер.');
  }
  return h('div.banner.bad', icon('i-warn'),
    h('div', h('strong', heading), detail, remedy));
}

/** In force / superseded / repealed — the three states a legal UI must never
 *  blur together. Repeal wins over validity: an article whose body is "утратила
 *  силу" has an open validity range and is still not law. */
function validityChip(src) {
  if (src.repealed) return chip('исключена', 'bad', { title: 'Статья утратила силу' });
  if (src.is_current) return chip('действует', 'ok', { title: 'Действующая редакция' });
  return chip('прежняя редакция', 'warn', { title: `Действовала до ${shortDate(src.valid_to)}` });
}

/** The verifier's summary is written for a log line ("2/2 citations verified");
 *  the console is in Russian, so translate it rather than leaking the internal
 *  wording into the interface. */
function verdictText(verdict) {
  const v = String(verdict || '');
  if (!v) return 'проверка не выполнялась';
  if (v === 'no_citations') return 'ни одной ссылки';
  if (v.startsWith('not_verified')) return 'не проверено: ' + v.split(':').slice(1).join(':').trim();
  const m = v.match(/^(\d+)\/(\d+)/);
  if (m) return `ссылки подтверждены ${m[1]}/${m[2]}`;
  return v;
}

/** Abstention reasons are internal identifiers, and one of them arrives with a
 *  raw Windows socket error inside it. Translated here so the strip reads as a
 *  diagnosis; the original string stays on the element's title attribute so
 *  nothing is actually hidden from whoever has to fix it. */
const ABSTAIN_TEXT = {
  no_matching_provisions: 'в корпусе нет норм по этому вопросу',
  no_model_loaded: 'генератор не настроен',
};
const _REFUSED = /10061|refused|ConnectError|ConnectTimeout|Connection ?(refused|error)|Max retries/i;

function abstainText(reason) {
  const r = String(reason || '');
  if (!r) return 'без указания причины';
  if (ABSTAIN_TEXT[r]) return ABSTAIN_TEXT[r];
  if (r.startsWith('generation_failed')) {
    return _REFUSED.test(r) ? 'генератор не отвечает' : 'сбой генерации';
  }
  return r;
}

/** True when an abstention was caused by nothing listening on the generator's
 *  port — the one failure with a one-line remedy. */
function isGeneratorDown(reason) {
  const r = String(reason || '');
  return r.startsWith('generation_failed') && _REFUSED.test(r);
}

function confidenceChip(conf) {
  const map = { high: ['ok', 'высокая уверенность'], medium: ['warn', 'средняя уверенность'], low: ['bad', 'низкая уверенность'] };
  const [tone, label] = map[conf] || ['warn', conf || 'н/д'];
  return chip(label, tone);
}

/* -------------------------------------------------------------- composer */

/** The question box, shared by Ask and Search so the two views cannot drift
 *  apart in what they let you specify. */
function composer({ placeholder, submitLabel, examples = [], onSubmit, state }) {
  const ta = h('textarea', {
    placeholder, rows: 2, value: state.question || '',
    oninput: () => { ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight, 260) + 'px'; },
    onkeydown: (e) => {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); fire(); }
    },
  });

  // Pre-filled with today rather than left blank: the server would default to
  // today anyway, and a visible date is the difference between "as the law
  // stands" and "as the law stood", which is the whole point of the system.
  const dateInput = h('input', { type: 'date', value: state.as_of || isoDay(new Date()), max: isoDay(new Date()) });

  let lang = state.lang || '';
  const langSeg = h('div.seg');
  for (const [value, label] of [['', 'Авто'], ['ru', 'RU'], ['kk', 'KK']]) {
    const b = h('button', {
      type: 'button', class: value === lang ? 'on' : '',
      onclick: () => {
        lang = value;
        [...langSeg.children].forEach((c) => c.classList.toggle('on', c === b));
      },
    }, label);
    langSeg.append(b);
  }

  const submit = h('button.primary', { type: 'button', onclick: () => fire() },
    submitLabel, h('span.kbd', '⌘↵'));

  function fire() {
    const q = ta.value.trim();
    if (q.length < 3) { ta.focus(); return; }
    state.question = q; state.as_of = dateInput.value; state.lang = lang;
    onSubmit({ question: q, as_of: dateInput.value || null, lang: lang || null });
  }

  const box = h('div.composer',
    ta,
    h('div.controls',
      h('label.inline-field', 'на дату', dateInput),
      langSeg,
      h('div.spacer'),
      submit));

  const wrap = h('div', box);
  if (examples.length) {
    wrap.append(h('div.examples', examples.map((q) =>
      h('button', { type: 'button', onclick: () => { ta.value = q; ta.dispatchEvent(new Event('input')); fire(); } }, q))));
  }
  requestAnimationFrame(() => { ta.dispatchEvent(new Event('input')); ta.focus(); });
  return { node: wrap, setBusy: (b) => { submit.disabled = b; submit.firstChild.textContent = b ? 'Выполняется…' : submitLabel; } };
}

/* ---------------------------------------------------------------- charts */

let tipEl = null;
function tip(html, evt) {
  if (!tipEl) { tipEl = h('div.tooltip'); document.body.append(tipEl); }
  tipEl.innerHTML = html;
  tipEl.style.display = 'block';
  const pad = 14;
  const r = tipEl.getBoundingClientRect();
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
  tipEl.style.left = x + 'px'; tipEl.style.top = y + 'px';
}
function untip() { if (tipEl) tipEl.style.display = 'none'; }

/** Round an axis maximum up to 1/2/5 × 10^k so ticks read as numbers a person
 *  would choose, not as whatever the largest sample happened to be. */
function niceMax(v) {
  if (!(v > 0)) return 1;
  const mag = 10 ** Math.floor(Math.log10(v));
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (v <= step * mag) return step * mag;
  }
  return 10 * mag;
}

function figure(title, desc, ...body) {
  return h('figure.chart',
    h('figcaption', h('div.cap-t', title), desc ? h('div.cap-d', desc) : null),
    ...body);
}

/** Daily volume. One series, so no legend — the caption names it. */
function volumeChart(daily) {
  const W = 560, H = 170, ML = 34, MR = 8, MT = 8, MB = 22;
  const max = niceMax(Math.max(1, ...daily.map((d) => d.n)));
  const iw = W - ML - MR, ih = H - MT - MB;
  const step = iw / Math.max(daily.length, 1);
  const bw = Math.max(3, Math.min(26, step - 4));
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });

  const ticks = [0, Math.round(max / 2), max].filter((v, i, a) => a.indexOf(v) === i);
  for (const t of ticks) {
    const y = MT + ih - (t / max) * ih;
    svg.append(svgEl('line', { class: 'gridline', x1: ML, x2: W - MR, y1: y, y2: y }));
    const lbl = svgEl('text', { class: 'tick', x: ML - 7, y: y + 3.5, 'text-anchor': 'end' });
    lbl.textContent = fmt.format(t); svg.append(lbl);
  }

  daily.forEach((d, i) => {
    const bh = Math.max(d.n > 0 ? 2 : 0, (d.n / max) * ih);
    const x = ML + i * step + (step - bw) / 2;
    const rect = svgEl('rect', {
      class: 'bar', x, y: MT + ih - bh, width: bw, height: bh, rx: 2,
    });
    rect.addEventListener('mousemove', (e) => tip(
      `<b>${d.day}</b><br>${d.n} запрос(ов)<br>отказов: ${d.abstained} · без проверки: ${d.unverified}`, e));
    rect.addEventListener('mouseleave', untip);
    svg.append(rect);
    if (daily.length <= 16 || i % Math.ceil(daily.length / 8) === 0) {
      const lbl = svgEl('text', { class: 'tick', x: x + bw / 2, y: H - 6, 'text-anchor': 'middle' });
      lbl.textContent = d.day.slice(5).replace('-', '.');
      svg.append(lbl);
    }
  });
  svg.append(svgEl('line', { class: 'axisline', x1: ML, x2: W - MR, y1: MT + ih, y2: MT + ih }));
  return svg;
}

/** p50 and p95 latency. Same unit, one axis — never two scales. */
function latencyChart(daily) {
  const W = 560, H = 170, ML = 42, MR = 40, MT = 10, MB = 22;
  const vals = daily.flatMap((d) => [d.p50, d.p95]);
  const max = niceMax(Math.max(1, ...vals));
  const iw = W - ML - MR, ih = H - MT - MB;
  const X = (i) => ML + (daily.length === 1 ? iw / 2 : (i / (daily.length - 1)) * iw);
  const Y = (v) => MT + ih - (v / max) * ih;
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });

  for (const t of [0, max / 2, max]) {
    const y = Y(t);
    svg.append(svgEl('line', { class: 'gridline', x1: ML, x2: W - MR, y1: y, y2: y }));
    const lbl = svgEl('text', { class: 'tick', x: ML - 7, y: y + 3.5, 'text-anchor': 'end' });
    lbl.textContent = t >= 1000 ? `${+(t / 1000).toFixed(1)} с` : `${Math.round(t)} мс`;
    svg.append(lbl);
  }

  const series = [
    { key: 'p50', label: 'медиана', color: 'var(--series-1)' },
    { key: 'p95', label: 'p95', color: 'var(--series-2)' },
  ];
  for (const s of series) {
    const d = daily.map((row, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(row[s.key]).toFixed(1)}`).join(' ');
    svg.append(svgEl('path', { d, fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    daily.forEach((row, i) => {
      const c = svgEl('circle', { cx: X(i), cy: Y(row[s.key]), r: 3.2, fill: s.color, stroke: 'var(--surface)', 'stroke-width': 2 });
      const hit = svgEl('circle', { cx: X(i), cy: Y(row[s.key]), r: 10, fill: 'transparent' });
      hit.addEventListener('mousemove', (e) => tip(`<b>${row.day}</b><br>${s.label}: ${row[s.key]} мс`, e));
      hit.addEventListener('mouseleave', untip);
      svg.append(c, hit);
    });
    // Direct label at the right end: identity never rests on colour alone.
    if (daily.length) {
      const last = daily[daily.length - 1];
      const t = svgEl('text', { class: 'dl', x: X(daily.length - 1) + 8, y: Y(last[s.key]) + 3.5, fill: s.color });
      t.textContent = s.label; svg.append(t);
    }
  }
  svg.append(svgEl('line', { class: 'axisline', x1: ML, x2: W - MR, y1: MT + ih, y2: MT + ih }));

  const legend = h('div.legend', series.map((s) =>
    h('span.it', h('span.sw.line', { style: `background:${s.color}` }), s.label)));
  return h('div', svg, legend);
}

/** Verification outcomes. Status palette, never the series palette — and every
 *  segment is labelled, because these five states must not be told apart by
 *  colour alone. */
const VERDICT_TONE = {
  'fully verified':      { color: 'var(--st-good)',     label: 'ссылки подтверждены' },
  'partly verified':     { color: 'var(--st-warning)',  label: 'подтверждены частично' },
  'verification failed': { color: 'var(--st-critical)', label: 'проверка не пройдена' },
  'no citations':        { color: 'var(--st-serious)',  label: 'без ссылок' },
  'abstained':           { color: 'var(--ink-3)',       label: 'отказ от ответа' },
};

function verdictBar(verdicts) {
  const total = verdicts.reduce((a, v) => a + v.n, 0);
  if (!total) return empty('Пока нет запросов', 'Здесь появится разбивка по результатам проверки ссылок.');
  const bar = h('div.stack');
  const legend = h('div.legend');
  for (const v of verdicts) {
    const meta = VERDICT_TONE[v.bucket] || { color: 'var(--ink-3)', label: v.bucket };
    const pct = (v.n / total) * 100;
    const seg = h('div.seg', { style: `background:${meta.color};flex:${v.n}`, title: `${meta.label}: ${v.n}` });
    seg.addEventListener('mousemove', (e) => tip(`<b>${meta.label}</b><br>${v.n} из ${total} · ${pct.toFixed(1)}%`, e));
    seg.addEventListener('mouseleave', untip);
    bar.append(seg);
    legend.append(h('span.it',
      h('span.sw', { style: `background:${meta.color}` }),
      `${meta.label} — `, h('span.num', String(v.n))));
  }
  return h('div', bar, legend);
}

/* ============================================================== views */

const state = {
  ask: { question: '', as_of: '', lang: '', result: null, error: null },
  search: { question: '', as_of: '', lang: '', result: null, error: null, packet: false },
  corpus: { acts: null, act: null, toc: null, node: null, lang: 'ru', as_of: '' },
  ingest: { files: [], job: null, jobs: [], opts: { verify: true, reset: false, build_index: true } },
  monitor: { stats: null, metrics: null, queries: null, filter: 'all', days: 14 },
};

let statsCache = null;

/* ------------------------------------------------------------------ ask */

const ASK_EXAMPLES = [
  'Какой срок давности по административным правонарушениям?',
  'Что считается крупным ущербом в Уголовном кодексе?',
  'Права работника при сокращении штата',
];

function viewAsk(root) {
  const out = h('div', { style: 'margin-top:26px' });
  const c = composer({
    placeholder: 'Задайте вопрос по Конституции или кодексам РК…',
    submitLabel: 'Спросить',
    examples: state.ask.result ? [] : ASK_EXAMPLES,
    state: state.ask,
    onSubmit: async (req) => {
      c.setBusy(true);
      out.replaceChildren(loading('Поиск норм, генерация и проверка ссылок…'));
      try {
        state.ask.result = await core('/ask', { method: 'POST', body: req });
        state.ask.error = null;
      } catch (err) {
        state.ask.result = null; state.ask.error = err;
      }
      c.setBusy(false);
      renderAnswer(out);
    },
  });
  root.append(c.node, out);
  renderAnswer(out);
}

function renderAnswer(out) {
  out.replaceChildren();
  if (state.ask.error) { out.append(errorBox(state.ask.error)); return; }
  const a = state.ask.result;
  if (!a) {
    out.append(empty('Ответ будет содержать только текст закона',
      'Модель не формулирует норму своими словами: она цитирует найденные статьи, а каждая ссылка проверяется по базе — на существование, на попадание в выдачу, на дату и на дословность цитаты.'));
    return;
  }

  const failed = !a.verified || a.abstained;
  const strip = h('div.verdict-strip' + (a.abstained ? '.warn' : failed ? '.bad' : ''));
  if (a.abstained) {
    strip.append(chip('отказ от ответа', 'warn'),
      h('span.dim', { title: a.abstain_reason || '' }, abstainText(a.abstain_reason)));
  } else {
    strip.append(chip(verdictText(a.verification), a.verified ? 'ok' : 'bad'));
    strip.append(h('span.sep'), confidenceChip(a.confidence));
  }
  strip.append(h('span.sep'), h('span.dim', 'на дату'), h('span.num', shortDate(a.as_of)));
  strip.append(h('span.sep'), h('span.dim', 'язык'), h('span.num', (a.lang || '').toUpperCase()));
  strip.append(h('span.sep'), h('span.dim', ms(a.timings_ms && a.timings_ms.total)));
  if (a.escalate) strip.append(h('span.sep'), chip('передать юристу', 'bad'));
  if (a.uses_superseded_text) strip.append(chip('есть прежние редакции', 'warn'));
  out.append(strip);

  if (isGeneratorDown(a.abstain_reason)) {
    // Retrieval clearly worked — the sources below prove it — so the banner says
    // exactly which half failed, rather than letting an empty answer imply the
    // corpus came up short.
    out.append(h('div.banner.bad', { style: 'margin-top:14px;margin-bottom:0' },
      icon('i-warn'),
      h('div',
        h('strong', 'Генератор не запущен. '),
        'Нормы найдены и приведены ниже, но сформулировать ответ не удалось: ',
        'на порту генератора никто не слушает.',
        h('div', { style: 'margin-top:7px' },
          'Запустите llama-server: ', h('span.mono', 'scripts\\serve_llm.ps1'),
          ' (если бинарника нет в PATH — сначала ',
          h('span.mono', '$env:LLAMA_SERVER'), '), затем повторите вопрос.'),
        h('div', { style: 'margin-top:5px' },
          h('span.dim', { title: a.abstain_reason }, a.abstain_reason)))));
  }

  if (a.verification_problems && a.verification_problems.length) {
    out.append(h('div.banner.bad', { style: 'margin-top:14px;margin-bottom:0' },
      icon('i-warn'),
      h('div',
        h('strong', 'Проверка ссылок не пройдена. '),
        'Вывод заменён перечнем норм.',
        h('ul', { style: 'margin:7px 0 0;padding-left:17px' },
          a.verification_problems.map((p) => h('li', p))))));
  }

  out.append(h('div.answer-prose', a.text || '—'));

  if (a.citations && a.citations.length) {
    out.append(h('hr.rule'), eyebrow(`Ссылки — ${a.citations.length}`));
    const list = h('div', { style: 'margin-top:10px' });
    a.citations.forEach((cit, i) => {
      const pin = [cit.act_code, cit.article && `ст. ${cit.article}`, cit.point && `п. ${cit.point}`]
        .filter(Boolean).join(' · ');
      list.append(h('div.citation',
        h('div.idx', `[${i + 1}]`),
        h('div',
          h('div.head',
            h('span.pin', pin || 'без привязки'),
            cit.source ? chip(`источник ${cit.source}`, null, { mono: true }) : null),
          cit.quote ? h('blockquote', `«${cit.quote}»`) : h('div.dim', 'цитата не приведена'))));
    });
    out.append(list);
  }

  if (a.sources && a.sources.length) {
    out.append(h('hr.rule'), eyebrow(`Источники в контексте — ${a.sources.length}`));
    const list = h('div', { style: 'margin-top:10px' });
    for (const s of a.sources) list.append(sourceRow(s));
    out.append(list);
  }

  if (a.disclaimer) out.append(h('div.disclaimer', a.disclaimer));
}

/** One packet source, with its text fetched only when opened: the packet
 *  payload carries provenance, not bodies. */
function sourceRow(s) {
  const body = h('div.full', h('div.loading', h('span.spinner'), 'Загрузка текста…'));
  const det = h('details.source',
    h('summary',
      h('span.n', `[${s.index}]`),
      h('div.h',
        h('div.t', s.heading || s.node_id,
          // Three sources from one article differ only by their point, and a
          // repeated heading with nothing to tell them apart reads like a bug.
          pointLabel(s) ? h('span.dim', { style: 'font-weight:400' }, ` · ${pointLabel(s)}`) : null),
        h('div.p', `${s.node_id}${s.act_code ? ' · ' + s.act_code : ''} · ${(s.lang || '').toUpperCase()} · ${shortDate(s.valid_from)} — ${shortDate(s.valid_to)}`)),
      h('div.meta',
        (s.provenance || []).map((p) => chip(p, null, { mono: true })),
        validityChip(s))),
    body);

  let loaded = false;
  det.addEventListener('toggle', async () => {
    if (!det.open || loaded) return;
    loaded = true;
    try {
      const data = await api('/nodes/' + s.article_node_id + (s.valid_from ? `?as_of=${s.valid_from}` : ''));
      const pick = data.texts.find((t) => t.lang === s.lang && t.valid_from === s.valid_from)
        || data.texts.find((t) => t.in_force_on_as_of) || data.texts[0];
      body.replaceChildren(pick.body || h('span.dim', 'Текст пуст.'));
    } catch (err) {
      body.replaceChildren(h('div.dim', `Текст не загружен: ${err.message}`));
    }
  });
  return det;
}

/* --------------------------------------------------------------- search */

function viewSearch(root) {
  const out = h('div', { style: 'margin-top:26px' });
  const c = composer({
    placeholder: 'Запрос к поиску — без генерации ответа…',
    submitLabel: 'Найти',
    state: state.search,
    onSubmit: async (req) => {
      c.setBusy(true);
      out.replaceChildren(loading('Плотный + лексический поиск, слияние рангов…'));
      try {
        state.search.result = await core('/search', { method: 'POST', body: { ...req, include_packet: true } });
        state.search.error = null;
      } catch (err) { state.search.result = null; state.search.error = err; }
      c.setBusy(false);
      renderSearch(out);
    },
  });
  root.append(c.node, out);
  renderSearch(out);
}

function renderSearch(out) {
  out.replaceChildren();
  if (state.search.error) { out.append(errorBox(state.search.error)); return; }
  const r = state.search.result;
  if (!r) {
    out.append(empty('Поиск без модели',
      'Самый быстрый способ понять, почему ответ получился таким: видно, какая ветвь — векторная, лексическая или триграммная — дала каждый фрагмент, и с каким весом.'));
    return;
  }

  const t = r.timings_ms || {};
  const meta = h('div.verdict-strip',
    chip(`${r.hits.length} фрагментов`, null),
    h('span.sep'), h('span.dim', 'на дату'), h('span.num', shortDate(r.as_of)),
    h('span.sep'), h('span.dim', 'язык'), h('span.num', (r.lang || '').toUpperCase()),
    r.act_hints && r.act_hints.length
      ? [h('span.sep'), h('span.dim', 'акты'), ...r.act_hints.map((a) => chip(a, null, { mono: true }))]
      : null,
    h('span.sep'),
    ...Object.entries(t).map(([k, v]) => h('span.dim', { style: 'font-size:11.5px' }, `${k} ${v}мс`)));
  out.append(meta);

  if (r.packet_truncated) {
    out.append(h('div.banner', { style: 'margin-top:14px;margin-bottom:0' }, icon('i-warn'),
      h('div', h('strong', 'Контекст обрезан по бюджету токенов. '),
        `Отброшено источников: ${r.dropped_sources}. Пакет: ${n(r.packet_tokens)} токенов.`)));
  }

  const maxFused = Math.max(...r.hits.map((x) => x.fused_score || 0), 1e-9);
  const list = h('div', { style: 'margin-top:18px' });
  r.hits.forEach((hit, i) => {
    const scores = h('div.scores',
      scorebar('fused', hit.fused_score, hit.fused_score / maxFused),
      hit.rerank_score !== null && hit.rerank_score !== undefined ? scorebar('rerank', hit.rerank_score, null) : null,
      scorebar('dense', hit.dense_score, hit.dense_score),
      scorebar('lexical', hit.lexical_score, Math.min(1, hit.lexical_score)));
    const row = h('div.hit',
      h('div.top',
        h('span.rank', String(i + 1).padStart(2, '0')),
        h('span.title', hit.heading || hit.node_id),
        validityChip(hit),
        ...(hit.provenance || []).map((p) => chip(p, null, { mono: true }))),
      h('div.path', `${hit.act_title || hit.act_code} · ${hit.hierarchy_path || '—'} · `, h('span.mono', hit.chunk_id)),
      h('div.body', hit.text || ''));
    // Statutory text is long by nature; clamped by default so eight hits stay
    // comparable on one screen, expandable because the wording is the evidence.
    const more = h('button.more', { type: 'button', onclick: () => {
      row.classList.toggle('open');
      more.textContent = row.classList.contains('open') ? 'свернуть' : 'показать статью полностью';
    } }, 'показать статью полностью');
    row.append(more, scores);
    list.append(row);
  });
  out.append(list);

  if (r.packet) {
    const pre = h('pre.packet', r.packet);
    pre.hidden = true;
    const btn = h('button.ghost', {
      type: 'button',
      onclick: () => { pre.hidden = !pre.hidden; btn.firstChild.textContent = pre.hidden ? 'Показать контекст модели' : 'Скрыть контекст модели'; },
    }, 'Показать контекст модели');
    out.append(h('hr.rule'),
      h('div', { style: 'display:flex;align-items:center;gap:12px;margin-bottom:12px' },
        eyebrow(`Контекст — ${n(r.packet_tokens)} токенов`), btn),
      pre);
  }
}

function scorebar(label, value, ratio) {
  const bar = ratio === null || ratio === undefined ? null
    : h('u', h('b', { style: `width:${Math.max(2, Math.min(100, ratio * 100))}%` }));
  return h('span.scorebar', h('span.k', label), (value ?? 0).toFixed(4), bar);
}

/* --------------------------------------------------------------- corpus */

async function viewCorpus(root) {
  const left = h('div.card', { style: 'overflow:hidden' });
  const right = h('div');
  root.append(h('div.split.wide-left', left, right));

  left.append(loading());
  try {
    state.corpus.acts = await api('/acts');
  } catch (err) { left.replaceChildren(errorBox(err)); return; }

  const acts = state.corpus.acts;
  left.replaceChildren(h('header', h('div.eyebrow', `Акты — ${acts.length}`),
    h('span.dim', { style: 'font-size:11.5px' }, 'по юридической силе')));
  if (!acts.length) {
    left.append(h('div.body', h('div.dim', 'Корпус пуст. Импортируйте выпуск во вкладке «Импорт».')));
    right.append(empty('Нет данных', 'Сначала импортируйте корпус.'));
    return;
  }

  const body = h('div');
  left.append(body);
  for (const act of acts) {
    const btn = h('button.act-item', {
      type: 'button',
      onclick: () => { [...body.children].forEach((c) => c.classList.remove('on')); btn.classList.add('on'); openAct(act, right); },
    },
      h('div.t', act.title_ru || act.code),
      h('div.m',
        h('span', act.code),
        h('span', `${n(act.articles)} ст.`),
        h('span', `${n(act.chunks)} фрагм.`),
        h('span', (act.langs || []).join('/'))));
    body.append(btn);
  }
  body.firstChild.classList.add('on');
  openAct(acts[0], right);
}

async function openAct(act, right) {
  state.corpus.act = act;
  const head = h('div', { style: 'margin-bottom:18px' },
    eyebrow(`${ACT_KIND[act.kind] || act.kind} · ранг ${act.norm_rank}`),
    h('h2', { style: 'margin-top:6px' }, act.title_ru || act.code),
    h('div.dim', { style: 'margin-top:6px;font-size:12.5px' },
      [act.official_id, act.adopted_on ? `принят ${shortDate(act.adopted_on)}` : null,
        act.repealed ? `${act.repealed} исключённых фрагментов` : null].filter(Boolean).join(' · ')));

  const langSeg = h('div.seg');
  for (const l of ['ru', 'kk']) {
    const b = h('button', { type: 'button', class: state.corpus.lang === l ? 'on' : '', onclick: () => { state.corpus.lang = l; openAct(act, right); } }, l.toUpperCase());
    langSeg.append(b);
  }
  const dateInput = h('input', {
    type: 'date', value: state.corpus.as_of || isoDay(new Date()), max: isoDay(new Date()),
    onchange: () => { state.corpus.as_of = dateInput.value; openAct(act, right); },
  });

  const tocBox = h('div', loading('Структура акта…'));
  const detail = h('div');
  right.replaceChildren(head,
    h('div', { style: 'display:flex;gap:12px;align-items:center;margin-bottom:18px' },
      h('span.eyebrow', 'редакция на'), dateInput, langSeg),
    h('div.split', h('div', tocBox), h('div', detail)));

  detail.append(empty('Выберите статью', 'Справа появится текст с историей редакций и перекрёстными ссылками.'));

  let data;
  try {
    const qs = new URLSearchParams({ lang: state.corpus.lang });
    if (state.corpus.as_of) qs.set('as_of', state.corpus.as_of);
    data = await api(`/acts/${encodeURIComponent(act.code)}/toc?${qs}`);
  } catch (err) { tocBox.replaceChildren(errorBox(err)); return; }

  const toc = h('div.toc');
  const articles = [];
  const walk = (nodes, depth) => {
    for (const node of nodes) {
      if (node.kind === 'article') {
        const btn = h('button.article', {
          type: 'button',
          onclick: () => {
            articles.forEach((b) => b.classList.remove('on'));
            btn.classList.add('on');
            openNode(node.id, detail);
          },
        },
          h('span.no', node.number ? `ст. ${node.number}` : '—'),
          h('span' + (node.heading ? '.h' : '.h.absent'), node.heading || 'нет текста на эту дату'));
        articles.push(btn);
        toc.append(btn);
      } else if (node.kind !== 'act') {
        toc.append(h('div.chapter', h('div.lbl', node.heading || `${node.kind} ${node.number || ''}`)));
      }
      if (node.children && node.children.length) walk(node.children, depth + 1);
    }
  };
  walk(data.tree, 0);
  tocBox.replaceChildren(
    h('div', { style: 'display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px' },
      eyebrow(`Оглавление — ${articles.length} ст.`),
      h('span.dim', { style: 'font-size:11.5px' }, shortDate(data.as_of))),
    toc);
  if (!articles.length) tocBox.append(h('div.dim', { style: 'padding:16px 0' }, 'На эту дату структура пуста.'));
}

async function openNode(nodeId, detail) {
  detail.replaceChildren(loading());
  let data;
  try { data = await api('/nodes/' + nodeId + (state.corpus.as_of ? `?as_of=${state.corpus.as_of}` : '')); }
  catch (err) { detail.replaceChildren(errorBox(err)); return; }

  const lang = state.corpus.lang;
  const mine = data.texts.filter((t) => t.lang === lang);
  const texts = mine.length ? mine : data.texts;

  const box = h('div',
    h('div', { style: 'display:flex;align-items:baseline;gap:10px;margin-bottom:16px' },
      h('h3', nodeId),
      h('span.dim', { style: 'font-size:11.5px' }, `${texts.length} редакц.`)));

  texts.forEach((t) => {
    const cls = t.in_force_on_as_of ? 'current' : 'past';
    box.append(h('div.redaction.' + cls,
      h('div.h',
        t.in_force_on_as_of ? chip('в силе на выбранную дату', 'ok') : chip('прежняя редакция', 'warn'),
        chip(t.lang.toUpperCase(), null, { mono: true }),
        h('span.dim', { style: 'font-size:11.5px' },
          `${shortDate(t.valid_from)} — ${shortDate(t.valid_to)}`),
        t.release_id ? chip(t.release_id, null, { mono: true, title: 'Выпуск корпуса' }) : null),
      t.heading ? h('div', { style: 'font-weight:600;margin-bottom:7px' }, t.heading) : null,
      h('div.txt', t.body || '—')));
  });

  if (data.references && data.references.length) {
    box.append(h('hr.rule'), eyebrow(`Ссылается на — ${data.references.length}`),
      h('div', { style: 'margin-top:10px;display:flex;flex-wrap:wrap;gap:6px' },
        data.references.map((r) => r.dst_node_id
          ? chip(r.dst_node_id, null, { mono: true, title: r.raw })
          : chip(r.raw.slice(0, 40), 'warn', { mono: true, title: 'Ссылка не разрешена — акт отсутствует в корпусе' }))));
  }
  if (data.cited_by && data.cited_by.length) {
    box.append(h('hr.rule'), eyebrow(`На неё ссылаются — ${data.cited_by.length}`),
      h('div', { style: 'margin-top:10px;display:flex;flex-wrap:wrap;gap:6px' },
        data.cited_by.map((id) => chip(id, null, { mono: true }))));
  }
  detail.replaceChildren(box);
}

/* --------------------------------------------------------------- ingest */

function viewIngest(root) {
  const s = state.ingest;

  const input = h('input', {
    type: 'file', multiple: true, hidden: true,
    accept: '.txt,.json,.html,.htm',
    onchange: () => readFiles([...input.files]),
  });

  const drop = h('div.drop',
    h('h3', 'Перетащите файлы выпуска'),
    h('p', h('span.mono', '*.act.txt'), ' · ', h('span.mono', 'documents.json'),
      ' + HTML adilet · ', h('span.mono', 'manifest.json')),
    h('button', { type: 'button', onclick: () => input.click() }, 'Выбрать файлы'),
    input);

  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault(); drop.classList.remove('over');
    readFiles([...e.dataTransfer.files]);
  });

  const files = h('div.filelist');
  const opts = h('div', { style: 'margin-top:8px' });
  const runBtn = h('button.primary', { type: 'button', disabled: true, onclick: run }, 'Импортировать');
  const jobBox = h('div', { style: 'margin-top:26px' });

  function readFiles(list) {
    const jobs = list.map((f) => new Promise((res) => {
      const r = new FileReader();
      r.onload = () => res({ name: f.name, content: String(r.result), size: f.size });
      r.onerror = () => res(null);
      r.readAsText(f, 'utf-8');
    }));
    Promise.all(jobs).then((read) => {
      s.files = read.filter(Boolean);
      renderFiles();
    });
  }

  function renderFiles() {
    files.replaceChildren();
    if (!s.files.length) { runBtn.disabled = true; return; }
    runBtn.disabled = false;
    files.append(eyebrow(`Готово к импорту — ${s.files.length} файл(ов)`));
    const box = h('div', { style: 'margin-top:10px' });
    for (const f of s.files) {
      box.append(h('div.filerow',
        h('span.nm', f.name),
        f.name.endsWith('.act.txt') ? chip('источник акта', 'ok')
          : f.name === 'documents.json' ? chip('манифест adilet', 'ok')
          : f.name === 'manifest.json' ? chip('манифест выпуска', 'ok')
          : chip('вспомогательный', null),
        h('span.sz', `${(f.size / 1024).toFixed(1)} КБ`)));
    }
    files.append(box);
  }

  const optDefs = [
    ['verify', 'Проверять манифест выпуска', 'Отклонить пакет, если контрольные суммы не сходятся. Ничего не будет импортировано.'],
    ['build_index', 'Строить векторный индекс (HNSW)', 'Отключите для быстрой проверки парсинга — поиск по векторам будет недоступен до пересборки.'],
    ['reset', 'Сбросить схему перед импортом', 'Удаляет ВЕСЬ корпус, включая журнал запросов. Используйте только для полной переустановки.'],
  ];
  for (const [key, title, desc] of optDefs) {
    const cb = h('input', { type: 'checkbox', checked: s.opts[key], onchange: () => { s.opts[key] = cb.checked; runBtn.className = s.opts.reset ? 'danger' : 'primary'; runBtn.firstChild.textContent = s.opts.reset ? 'Сбросить и импортировать' : 'Импортировать'; } });
    opts.append(h('label.opt', cb, h('div', h('div.t', title), h('div.d', desc))));
  }

  async function run() {
    runBtn.disabled = true;
    jobBox.replaceChildren(loading('Файлы передаются на сервер…'));
    try {
      const job = await api('/ingest', {
        method: 'POST',
        body: {
          files: s.files.map((f) => ({ name: f.name, content: f.content })),
          verify: s.opts.verify, reset: s.opts.reset, build_index: s.opts.build_index,
        },
      });
      s.job = job;
      pollJob(job.id, jobBox, () => { runBtn.disabled = false; statsCache = null; refreshRuntime(); });
    } catch (err) {
      jobBox.replaceChildren(errorBox(err));
      runBtn.disabled = false;
    }
  }

  root.append(
    h('div.split.even',
      h('div',
        drop, files,
        h('hr.rule'),
        eyebrow('Параметры импорта'), opts,
        h('div', { style: 'margin-top:16px' }, runBtn),
        jobBox),
      h('div',
        h('div.card',
          h('header', h('div.eyebrow', 'Как это работает')),
          h('div.body', { style: 'font-size:12.5px;line-height:1.65;color:var(--ink-2)' },
            h('p', { style: 'margin-top:0' }, 'Файлы сохраняются в ', h('span.mono', 'data/uploads/<job>'),
              ' и импортируются тем же кодом, что и ', h('span.mono', 'lawrag ingest'), ' — парсинг, разбиение на фрагменты, извлечение ссылок и определений, эмбеддинги.'),
            h('p', 'Импорт выполняется по одному за раз: сброс схемы во время параллельного импорта повредил бы корпус.'),
            h('p', { style: 'margin-bottom:0' }, 'Пакет без ', h('span.mono', 'manifest.json'),
              ' будет импортирован, но манифест посчитается локально — это не то же самое, что подпись, сделанная на подключённой стороне.'))),
        h('div', { style: 'margin-top:18px' }, jobHistory()))));

  renderFiles();
}

function jobHistory() {
  const box = h('div.card', h('header', h('div.eyebrow', 'История импортов')), h('div.body', loading()));
  api('/ingest/jobs').then((jobs) => {
    const body = box.lastChild;
    if (!jobs.length) { body.replaceChildren(h('div.dim', { style: 'font-size:12.5px' }, 'В этом процессе импортов ещё не было.')); return; }
    body.replaceChildren(h('table.grid',
      h('thead', h('tr', h('th', 'Задача'), h('th', 'Файлы'), h('th', 'Статус'), h('th.r', 'Начало'))),
      h('tbody', jobs.map((j) => h('tr',
        h('td', h('span.mono', j.id)),
        h('td', String(j.files.length)),
        h('td', j.status === 'done' ? chip('готово', 'ok') : j.status === 'failed' ? chip('ошибка', 'bad') : chip('выполняется', 'warn')),
        h('td.r.num', whenAgo(j.started_at)))))));
  }).catch(() => { box.lastChild.replaceChildren(h('div.dim', 'Недоступно.')); });
  return box;
}

function pollJob(jobId, box, done) {
  let stopped = false;
  const tick = async () => {
    if (stopped) return;
    let job;
    try { job = await api('/ingest/jobs/' + jobId); }
    catch (err) { box.replaceChildren(errorBox(err)); done(); return; }

    const tone = job.status === 'done' ? 'ok' : job.status === 'failed' ? 'bad' : 'warn';
    box.replaceChildren(
      h('div', { style: 'display:flex;align-items:center;gap:10px;margin-bottom:12px' },
        eyebrow('Импорт'), h('span.mono', job.id),
        chip(job.status === 'running' ? 'выполняется' : job.status === 'done' ? 'готово' : 'ошибка', tone),
        job.status === 'running' ? h('span.spinner') : null),
      job.totals && Object.keys(job.totals).length
        ? h('div.stat-row', { style: 'margin-bottom:12px' },
          Object.entries(job.totals).map(([k, v]) =>
            h('div.stat', h('div.k', k), h('div.v', typeof v === 'number' ? n(v) : String(v).slice(0, 12)))))
        : null,
      h('pre.joblog', job.log.join('\n')));
    const log = box.querySelector('.joblog');
    if (log) log.scrollTop = log.scrollHeight;

    if (job.status === 'running') setTimeout(tick, 1200);
    else { stopped = true; done(); }
  };
  tick();
}

/* -------------------------------------------------------------- monitor */

async function viewMonitor(root) {
  root.append(loading('Опрос базы…'));
  let stats, metrics;
  try {
    [stats, metrics] = await Promise.all([api('/stats'), api(`/metrics?days=${state.monitor.days}`).catch(() => null)]);
  } catch (err) { root.replaceChildren(errorBox(err)); return; }
  state.monitor.stats = stats;
  statsCache = stats;
  refreshRuntime();

  root.replaceChildren();

  // Everything below distinguishes "we know this is zero" from "we cannot know".
  const live = stats.database.reachable;

  if (!live) {
    root.append(h('div.banner.bad', icon('i-warn'),
      h('div', h('strong', 'База данных недоступна. '), stats.database.error || '',
        h('div', { style: 'margin-top:6px' }, 'Запросы не выполняются. Проверьте ', h('span.mono', 'LAWRAG_DB_DSN'), '.'))));
  }
  const rt = stats.runtime;
  const missing = rt.missing_modules || [];
  // Two distinct failures, reported separately: configured for placeholders
  // (answers, but not real ones) and configured for production but unable to
  // load (does not answer at all). One banner for both would be a lie either way.
  if (missing.length || rt.answering_error) {
    root.append(h('div.banner.bad', icon('i-warn'),
      h('div',
        h('strong', 'Модели не загружаются. '),
        missing.length
          ? [
            'Конфигурация требует ',
            ...missing.map((m, i) => [i ? ', ' : '', h('span.mono', m.backend),
              ' (', h('span.mono', m.module), ')']),
            ', но этих пакетов нет в окружении, из которого запущен сервер.',
          ]
          : rt.answering_error,
        h('div', { style: 'margin-top:8px;display:flex;align-items:center;gap:8px;flex-wrap:wrap' },
          h('span.dim', 'интерпретатор'), h('span.mono', rt.python || '—')),
        h('div', { style: 'margin-top:7px' },
          'Установите стек весов: ',
          h('span.mono', 'lawrag[models]'), ' и перезапустите сервер.'))));
  } else if (rt.llm && rt.llm.reachable === false) {
    root.append(h('div.banner.bad', icon('i-warn'),
      h('div',
        h('strong', 'Генератор не отвечает. '),
        'Поиск работает, но ответы формулировать нечем: на ',
        h('span.mono', rt.llm.base_url || rt.llm_base_url || '—'),
        ' нет соединения.',
        h('div', { style: 'margin-top:7px' },
          'Запустите llama-server: ', h('span.mono', 'scripts\\serve_llm.ps1'),
          ' (профиль B по умолчанию). Если ', h('span.mono', 'llama-server'),
          ' не в PATH — сначала ', h('span.mono', '$env:LLAMA_SERVER = "…\\llama-server.exe"'), '.'),
        rt.llm.error ? h('div', { style: 'margin-top:5px' },
          h('span.dim', rt.llm.error)) : null)));
  } else {
    // Driven off the flags themselves rather than off configured_for_production,
    // so an older payload without that field cannot produce a banner naming no
    // backend at all.
    const placeholders = [
      rt.placeholder_embedding ? `эмбеддинги «${rt.embedding_backend}»` : null,
      rt.placeholder_llm ? `генератор «${rt.llm_backend}»` : null,
    ].filter(Boolean);
    if (placeholders.length) {
      root.append(h('div.banner', icon('i-warn'),
        h('div',
          h('strong', 'Система работает на заглушках: '),
          placeholders.join(' и '),
          '. Результаты поиска и ответы не отражают реальное качество системы.')));
    }
  }

  // Undated corpus: every point-in-time feature in this system is inert, and
  // that is invisible from anywhere else in the interface — the date control
  // still works, it just cannot change any answer.
  const dating = stats.dating || {};
  if (live && dating.total && dating.undated / dating.total > 0.5) {
    const pct = Math.round((dating.undated / dating.total) * 100);
    root.append(h('div.banner.bad', icon('i-warn'),
      h('div',
        h('strong', 'Даты редакций не импортированы. '),
        `${pct}% текстов (${n(dating.undated)} из ${n(dating.total)}) стоят на `,
        h('span.mono', '1900-01-01'),
        ' — это запасное значение парсера, а не дата принятия. Разных дат начала: ',
        h('span.num', n(dating.distinct_starts)),
        `, закрытых редакций: ${n(dating.closed)}.`,
        h('div', { style: 'margin-top:7px' },
          'Поиск на дату и история редакций для такого корпуса ничего не меняют: ',
          'любая статья выглядит действующей с 1900 года и ни разу не изменённой. ',
          h('span.mono', 'documents.json'), ' должен содержать ',
          h('span.mono', 'valid_from'), ' для каждого документа.'))));
  }

  const c = stats.counts || {};
  const cov = stats.coverage || {};
  const q = stats.queries || {};
  // With the database down every number below is unknown, and a tile reading
  // "индекс полный" under a "database unavailable" banner is worse than blank.
  const detail = (text) => (live ? text : 'нет данных');
  root.append(h('div.stat-row',
    statTile('Акты', c.act, detail(`${n(c.node)} узлов`)),
    statTile('Фрагменты', c.chunk, detail(`${n(cov.current)} действующих · ${n(cov.superseded)} прежних`)),
    statTile('Эмбеддинги', c.chunk_embedding,
      detail(stats.unembedded_chunks ? `${n(stats.unembedded_chunks)} без вектора` : 'индекс полный'),
      live && stats.unembedded_chunks > 0),
    statTile('Ссылки', c.xref,
      detail(`${n((stats.unresolved || []).reduce((a, r) => a + r.n, 0))} неразрешённых`)),
    statTile('Определения', c.term, detail('глоссарий')),
    statTile('Редакции', dating.distinct_starts,
      detail(dating.undated ? `${n(dating.undated)} без даты` : 'все датированы'),
      live && dating.total > 0 && dating.undated > 0),
    statTile('Запросы', q.total, detail(`${n(q.last_24h)} за сутки · ${ms(q.avg_latency_ms)} средн.`))));

  root.append(h('hr.rule'));

  if (metrics) {
    const daily = metrics.daily || [];
    root.append(h('div', { style: 'display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:16px' },
      eyebrow(`Журнал запросов — ${metrics.days} дней`),
      daysSelector()));
    root.append(h('div.chart-grid',
      figure('Запросов в день', 'Каждый ответ пишется в query_log и воспроизводим по нему.',
        daily.length ? volumeChart(daily) : empty('Нет запросов за период')),
      figure('Задержка ответа', 'Медиана и 95-й перцентиль, включая генерацию и проверку ссылок.',
        daily.length ? latencyChart(daily) : empty('Нет данных о задержке')),
      figure('Результат проверки ссылок', 'Ответ с непройденной проверкой заменяется перечнем норм.',
        verdictBar(metrics.verdicts || [])),
      figure('Чаще всего цитируется', 'По ссылкам, которые прошли в итоговый ответ.',
        topArticles(metrics.top_articles || []))));
    root.append(h('hr.rule'));
  }

  root.append(h('div.split.rail',
    h('div', queryLog()),
    h('div',
      unresolvedCard(stats.unresolved || [], live),
      h('div', { style: 'margin-top:18px' }, releasesCard(stats.releases || [], live)),
      h('div', { style: 'margin-top:18px' }, runtimeCard(rt)))));
}

function daysSelector() {
  const seg = h('div.seg');
  for (const d of [7, 14, 30, 90]) {
    seg.append(h('button', {
      type: 'button', class: state.monitor.days === d ? 'on' : '',
      onclick: () => { state.monitor.days = d; render(); },
    }, `${d} дн`));
  }
  return seg;
}

function statTile(label, value, detail, alert) {
  return h('div.stat' + (alert ? '.alert' : ''),
    h('div.k', label), h('div.v', n(value)), detail ? h('div.d', detail) : null);
}

function topArticles(rows) {
  if (!rows.length) return empty('Пока нет цитирований');
  const max = Math.max(...rows.map((r) => r.n));
  return h('div.stackbars', rows.slice(0, 8).map((r) => h('div',
    h('div', { style: 'display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px' },
      h('span.mono', `${r.act_code || '?'} · ст. ${r.article}`),
      h('span.num.dim', String(r.n))),
    h('div', { style: 'height:6px;background:var(--surface-3);border-radius:2px;overflow:hidden' },
      h('div', { style: `height:100%;width:${(r.n / max) * 100}%;background:var(--accent);border-radius:2px` })))));
}

function queryLog() {
  const box = h('div.card');
  const tbody = h('tbody');
  const filters = h('div.seg');
  const search = h('input', {
    type: 'search', placeholder: 'поиск по вопросу…', style: 'width:190px',
    oninput: () => { clearTimeout(search._t); search._t = setTimeout(load, 280); },
  });

  for (const [key, label] of [['all', 'Все'], ['abstained', 'Отказы'], ['unverified', 'Без проверки'], ['slow', 'Медленные']]) {
    filters.append(h('button', {
      type: 'button', class: state.monitor.filter === key ? 'on' : '',
      onclick: () => {
        state.monitor.filter = key;
        [...filters.children].forEach((b, i) => b.classList.toggle('on', ['all', 'abstained', 'unverified', 'slow'][i] === key));
        load();
      },
    }, label));
  }

  const count = h('span.dim', { style: 'font-size:11.5px' }, '');
  box.append(
    h('header',
      h('div', { style: 'display:flex;align-items:baseline;gap:10px' },
        h('div.eyebrow', 'Журнал запросов'), count),
      h('div', { style: 'display:flex;gap:8px' }, search, filters)),
    h('div.scroll-x',
      h('table.grid',
        h('thead', h('tr',
          h('th', 'Когда'), h('th', 'Вопрос'), h('th', 'Проверка'),
          h('th.r', 'Ссылок'), h('th.r', 'Задержка'))),
        tbody)));

  async function load() {
    tbody.replaceChildren(h('tr', h('td', { colspan: 5 }, loading())));
    try {
      const data = await api(`/queries?limit=20&only=${state.monitor.filter}&q=${encodeURIComponent(search.value)}`);
      count.textContent = data.total > data.rows.length
        ? `${data.rows.length} из ${fmt.format(data.total)}` : `${data.total}`;
      if (!data.rows.length) {
        tbody.replaceChildren(h('tr', h('td', { colspan: 5 }, h('div.dim', { style: 'padding:18px 0' }, 'Ничего не найдено.'))));
        return;
      }
      tbody.replaceChildren(...data.rows.map((row) => h('tr.clickable', { onclick: () => openQuery(row.id) },
        h('td.num.dim', { style: 'white-space:nowrap' }, whenAgo(row.asked_at)),
        h('td', h('div.trunc', row.question)),
        h('td', verdictChip(row)),
        h('td.r.num', `${row.n_cited}/${row.n_retrieved}`),
        h('td.r.num', ms(row.latency_ms)))));
    } catch (err) {
      tbody.replaceChildren(h('tr', h('td', { colspan: 5 }, errorBox(err))));
    }
  }
  load();
  return box;
}

function verdictChip(row) {
  if (row.abstained) return chip('отказ', 'warn');
  const v = row.verdict || '';
  if (v === 'no_citations' || v.startsWith('not_verified')) return chip('без ссылок', 'bad');
  const m = v.match(/^(\d+)\/(\d+)/);
  if (m) return chip(`${m[1]}/${m[2]}`, m[1] === m[2] ? 'ok' : 'bad', { title: verdictText(v) });
  return chip(v || '—', null);
}

async function openQuery(id) {
  const bg = h('div.drawer-bg', { onclick: close });
  const panel = h('div.drawer', loading());
  document.body.append(bg, panel);
  function close() { bg.remove(); panel.remove(); document.removeEventListener('keydown', esc); }
  function esc(e) { if (e.key === 'Escape') close(); }
  document.addEventListener('keydown', esc);

  let row;
  try { row = await api('/queries/' + id); }
  catch (err) { panel.replaceChildren(errorBox(err)); return; }

  panel.replaceChildren(
    h('header',
      h('div', eyebrow(`Запрос #${row.id} · ${whenAgo(row.asked_at)}`),
        h('h2', { style: 'margin-top:7px' }, row.question)),
      h('button.ghost', { type: 'button', onclick: close }, icon('i-close'))),
    h('div.verdict-strip' + (row.abstained ? '.warn' : ''),
      verdictChip(row), h('span.sep'),
      h('span.dim', 'на дату'), h('span.num', shortDate(row.as_of)),
      h('span.sep'), h('span.num', (row.lang || '').toUpperCase()),
      h('span.sep'), h('span.num', ms(row.latency_ms)),
      row.release_id ? [h('span.sep'), chip(row.release_id.slice(0, 12), null, { mono: true })] : null),
    h('div.answer-prose', { style: 'font-size:15.5px' }, row.answer || '—'),
    h('hr.rule'),
    eyebrow(`Процитировано — ${(row.cited || []).length}`),
    h('div', { style: 'margin-top:10px' }, (row.cited || []).length
      ? (row.cited || []).map((c, i) => h('div.citation',
        h('div.idx', `[${i + 1}]`),
        h('div',
          h('div.head', h('span.pin', [c.act_code, c.article && `ст. ${c.article}`, c.point && `п. ${c.point}`].filter(Boolean).join(' · ') || '—')),
          c.quote ? h('blockquote', `«${c.quote}»`) : null)))
      : h('div.dim', 'Ни одной ссылки — ответ был понижен до перечня норм.')),
    h('hr.rule'),
    eyebrow(`Найдено фрагментов — ${(row.retrieved || []).length}`),
    h('div', { style: 'margin-top:10px;display:flex;flex-wrap:wrap;gap:6px' },
      (row.retrieved || []).map((cid) => chip(cid, null, { mono: true }))));
}

function unresolvedCard(rows, live = true) {
  const box = h('div.card',
    h('header', h('div.eyebrow', 'Неразрешённые ссылки'),
      h('span.dim', { style: 'font-size:11.5px' }, 'акты вне корпуса')));
  if (!rows.length) {
    box.append(h('div.body', h('div.dim', { style: 'font-size:12.5px' },
      live ? 'Все перекрёстные ссылки разрешены — либо корпус пуст.' : 'Нет данных.')));
    return box;
  }
  box.append(h('table.grid',
    h('thead', h('tr', h('th', 'Акт'), h('th.r', 'Ссылок'))),
    h('tbody', rows.map((r) => h('tr',
      h('td', h('span.mono', r.dst_act_code || '— без кода —')),
      h('td.r.num', n(r.n)))))));
  return box;
}

function releasesCard(rows, live = true) {
  const box = h('div.card', h('header', h('div.eyebrow', 'Выпуски корпуса')));
  if (!rows.length) {
    box.append(h('div.body', h('div.dim', { style: 'font-size:12.5px' },
      live ? 'Ни один выпуск не импортирован.' : 'Нет данных.')));
    return box;
  }
  box.append(h('table.grid',
    h('thead', h('tr', h('th', 'Выпуск'), h('th.r', 'Импортирован'))),
    h('tbody', rows.map((r) => h('tr',
      h('td',
        h('div', h('span.mono', { title: r.manifest_sha256 }, r.id || '—')),
        h('div.dim', { style: 'font-size:11px;margin-top:2px' }, r.source || '—')),
      h('td.r.num', whenAgo(r.imported_at)))))));
  return box;
}

function runtimeCard(rt) {
  const missingFor = (component) =>
    (rt.missing_modules || []).some((m) => m.component === component);
  const rows = [
    ['эмбеддинги', rt.embedding_backend, rt.placeholder_embedding || missingFor('embedding')],
    ['реранкер', rt.rerank_backend, missingFor('rerank')],
    ['генератор', rt.llm_backend === 'echo' ? 'echo' : `${rt.llm_backend} · ${rt.llm_model}`,
      rt.placeholder_llm || missingFor('llm') || (rt.llm && rt.llm.reachable === false)],
    ['порт генератора',
      (rt.llm && rt.llm.base_url) || rt.llm_base_url || '—',
      rt.llm && rt.llm.reachable === false],
    ['top-k', String(rt.final_k), false],
    ['бюджет контекста', `${n(rt.max_packet_tokens)} т.`, false],
    ['кросс-язык', rt.cross_language ? 'вкл' : 'выкл', false],
    ['база', rt.dsn, false],
    ['интерпретатор', rt.python || '—', false],
    // What the environment changed out from under config/default.yaml.
    ['переопределено средой',
      (rt.env_overrides && rt.env_overrides.length) ? rt.env_overrides.join(', ') : '—',
      false],
  ];
  // A definition list, not a table: half these values are long identifiers and
  // a right-aligned column breaks a DSN mid-word.
  return h('div.card',
    h('header', h('div.eyebrow', 'Конфигурация'),
      rt.production_ready ? chip('боевой режим', 'ok')
        : (rt.missing_modules || []).length || rt.answering_error
          ? chip('модели не загружаются', 'bad')
          : rt.llm && rt.llm.reachable === false
            ? chip('генератор не отвечает', 'bad')
            : chip('заглушки', 'warn')),
    h('div.deflist', rows.map(([k, v, warn]) => h('div.defrow',
      h('div.k', k),
      h('div.v' + (warn ? '.warn' : ''), v)))));
}

/* ============================================================== router */

const VIEWS = {
  ask: { title: 'Вопрос', eyebrow: 'Поиск с проверкой ссылок', icon: 'i-ask', nav: 'Вопрос', key: '1', mount: viewAsk },
  search: { title: 'Поиск', eyebrow: 'Извлечение без генерации', icon: 'i-search', nav: 'Поиск', key: '2', mount: viewSearch },
  corpus: { title: 'Корпус', eyebrow: 'Акты, статьи, редакции', icon: 'i-corpus', nav: 'Корпус', key: '3', mount: viewCorpus },
  ingest: { title: 'Импорт', eyebrow: 'Загрузка выпуска корпуса', icon: 'i-ingest', nav: 'Импорт', key: '4', mount: viewIngest },
  monitor: { title: 'Состояние', eyebrow: 'Корпус, задержки, журнал', icon: 'i-monitor', nav: 'Состояние', key: '5', mount: viewMonitor },
};

function currentRoute() {
  const key = (location.hash || '').replace(/^#\/?/, '').split('/')[0];
  return VIEWS[key] ? key : 'ask';
}

function buildNav() {
  const nav = document.getElementById('nav');
  nav.replaceChildren(...Object.entries(VIEWS).map(([key, v]) =>
    h('a', { href: '#/' + key, dataset: { route: key } },
      icon(v.icon), v.nav, h('span.key', v.key))));
}

function render() {
  const key = currentRoute();
  const view = VIEWS[key];
  document.querySelectorAll('#nav a').forEach((a) => a.classList.toggle('on', a.dataset.route === key));
  document.getElementById('page-title').textContent = view.title;
  document.getElementById('page-eyebrow').textContent = view.eyebrow;
  document.getElementById('page-actions').replaceChildren();
  document.title = `lawrag — ${view.title.toLowerCase()}`;
  const root = document.getElementById('view');
  root.className = 'view' + (key === 'monitor' || key === 'corpus' ? ' wide' : '');
  root.replaceChildren();
  untip();
  try { view.mount(root); }
  catch (err) { root.replaceChildren(errorBox(err)); }
}

/* -------------------------------------------------------------- chrome */

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('lawrag.theme', theme); } catch { /* private mode */ }
  const btn = document.getElementById('theme-toggle');
  btn.replaceChildren(icon(theme === 'dark' ? 'i-sun' : 'i-moon'),
    h('span', theme === 'dark' ? 'Светлая тема' : 'Тёмная тема'));
}

async function refreshRuntime() {
  const box = document.getElementById('runtime');
  try {
    const s = statsCache || (statsCache = await api('/stats'));
    const rt = s.runtime;
    const broken = (rt.missing_modules || []).length > 0 || !!rt.answering_error;
    const llmDown = !!(rt.llm && rt.llm.reachable === false);
    const label = !s.database.reachable ? 'база недоступна'
      : broken ? 'модели не загружаются'
      : llmDown ? 'генератор не отвечает'
      : rt.production_ready ? 'боевой режим' : 'заглушки';
    const tone = !s.database.reachable || broken || llmDown
      ? 'bad' : rt.production_ready ? 'ok' : 'warn';
    box.replaceChildren(
      h('div.row', h('span.eyebrow', 'состояние'), chip(label, tone)),
      h('div.row', h('span', 'фрагментов'), h('b', n((s.counts || {}).chunk))),
      h('div.row', h('span', 'актов'), h('b', n((s.counts || {}).act))));
  } catch (err) {
    box.replaceChildren(h('div.row', h('span.eyebrow', 'состояние'), chip('сервер недоступен', 'bad')));
  }
}

function boot() {
  let theme = 'light';
  try { theme = localStorage.getItem('lawrag.theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); } catch { /* ignore */ }
  applyTheme(theme);
  document.getElementById('theme-toggle').addEventListener('click', () => {
    applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
    if (currentRoute() === 'monitor') render();
  });

  buildNav();
  addEventListener('hashchange', render);
  addEventListener('keydown', (e) => {
    if (e.target.matches('input, textarea, select')) return;
    const entry = Object.entries(VIEWS).find(([, v]) => v.key === e.key);
    if (entry) location.hash = '#/' + entry[0];
  });
  render();
  refreshRuntime();
}

document.addEventListener('DOMContentLoaded', boot);
