// The layout checker (plan: Phase 5, session 1): runs inside a rendered page and lists layout problems a person
// would notice: overlapping things, text that's hard to read, things cut off or off the screen, near-miss
// alignment, uneven spacing, buttons too small to hit and text too small to read. The result is written into
// a hidden report element (id __layout_report) as JSON, so a headless browser's --dump-dom can read it.
(function () {
  const MIN_CONTRAST = 4.5, MIN_CONTRAST_LARGE = 3, MIN_TARGET = 24, MIN_FONT = 11, NEAR = 3, TOL = 0.5;
  const vw = document.documentElement.clientWidth, vh = Math.max(document.documentElement.clientHeight, 1);
  const problems = [];
  const visible = el => {
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    return s.display !== 'none' && s.visibility !== 'hidden' && +s.opacity > 0.05 && r.width > 0 && r.height > 0;
  };
  const name = el => {
    let n = el.tagName.toLowerCase();
    if (el.id) return n + '#' + el.id;
    if (el.classList.length) n += '.' + [...el.classList].slice(0, 2).join('.');
    const t = (el.innerText || el.value || el.alt || '').trim().replace(/\s+/g, ' ').slice(0, 30);
    return t ? `${n} "${t}"` : n;
  };
  const box = el => { const r = el.getBoundingClientRect(); return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) }; };
  const add = (kind, el, message, extra) => problems.push({ kind, element: name(el), box: box(el), message, ...(extra || {}) });

  // the things a person looks at: text blocks, controls, pictures
  const all = [...document.body.querySelectorAll('*')].filter(el => el.id !== '__layout_report' && visible(el));
  const ownText = el => [...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim());
  const isControl = el => /^(button|input|select|textarea|a)$/i.test(el.tagName) || el.getAttribute('role') === 'button' || el.onclick;
  const items = all.filter(el => ownText(el) || isControl(el) || /^(img|canvas|svg|video)$/i.test(el.tagName));

  // 1. overlaps between things that aren't nested in each other (and aren't meant to float, like a fixed overlay)
  const floats = el => { for (let e = el; e && e !== document.body; e = e.parentElement) { const p = getComputedStyle(e).position; if (p === 'fixed' || p === 'sticky') return true; } return false; };
  const rects = items.map(el => [el, el.getBoundingClientRect()]);
  for (let i = 0; i < rects.length && problems.length < 200; i++) for (let j = i + 1; j < rects.length; j++) {
    const [a, ra] = rects[i], [b, rb] = rects[j];
    if (a.contains(b) || b.contains(a) || floats(a) !== floats(b)) continue;
    const ox = Math.min(ra.right, rb.right) - Math.max(ra.left, rb.left), oy = Math.min(ra.bottom, rb.bottom) - Math.max(ra.top, rb.top);
    if (ox > NEAR && oy > NEAR && ox * oy > 0.15 * Math.min(ra.width * ra.height, rb.width * rb.height))
      add('overlap', a, `overlaps ${name(b)} by ${Math.round(ox)}x${Math.round(oy)} px`, { other: name(b) });
  }

  // 2. contrast of text against what's really behind it
  const rgba = c => { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return [0, 0, 0, 0]; const v = m[1].split(/[ ,/]+/).filter(Boolean).map(Number); return [v[0], v[1], v[2], v.length > 3 ? v[3] : 1]; };
  const lum = ([r, g, b]) => { const f = c => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  const background = el => {
    let col = [255, 255, 255], layers = [];
    for (let e = el; e; e = e.parentElement) { const c = rgba(getComputedStyle(e).backgroundColor); if (c[3] > 0) layers.push(c); if (c[3] >= 1) break; if (getComputedStyle(e).backgroundImage !== 'none') return null; }
    for (const c of layers.reverse()) col = col.map((v, k) => v * (1 - c[3]) + c[k] * c[3]);
    return col;
  };
  for (const el of all.filter(ownText)) {
    const s = getComputedStyle(el), fg = rgba(s.color), bg = background(el);
    if (!bg || fg[3] === 0) continue;
    const f = fg.slice(0, 3).map((v, k) => v * fg[3] + bg[k] * (1 - fg[3]));
    const [l1, l2] = [lum(f), lum(bg)].sort((a, b) => b - a), ratio = (l1 + 0.05) / (l2 + 0.05);
    const px = parseFloat(s.fontSize), large = px >= 24 || (px >= 18.66 && +s.fontWeight >= 700);
    if (ratio < (large ? MIN_CONTRAST_LARGE : MIN_CONTRAST)) add('contrast', el, `text contrast ${ratio.toFixed(2)}:1, needs ${large ? 3 : 4.5}:1`, { ratio: +ratio.toFixed(2) });
    if (px < MIN_FONT) add('small text', el, `text is ${px}px, below ${MIN_FONT}px`);
  }

  // 3. cut off: off the screen sideways, or text that doesn't fit its box
  for (const el of items) {
    const r = el.getBoundingClientRect();
    if (r.right > vw + TOL || r.left < -TOL) add('off screen', el, `reaches ${Math.round(r.left < 0 ? r.left : r.right - vw)} px past the ${r.left < 0 ? 'left' : 'right'} edge`);
    const s = getComputedStyle(el);
    if ((s.overflow === 'hidden' || s.overflowX === 'hidden' || s.textOverflow === 'ellipsis') && el.scrollWidth > el.clientWidth + 1) add('cut off', el, 'its text doesn\'t fit and is cut off');
  }
  if (document.documentElement.scrollWidth > vw + 1) problems.push({ kind: 'off screen', element: 'page', message: `the page is ${document.documentElement.scrollWidth - vw} px wider than the screen (sideways scrolling)` });

  // 4. near-miss alignment and uneven spacing among siblings laid out in a column or a row
  const groups = new Map();
  // (inline pieces of text, like a logo and a name in one line, are spaced by the text itself: not checked)
  for (const el of items) { if (getComputedStyle(el).display === 'inline') continue; const p = el.parentElement; if (!groups.has(p)) groups.set(p, []); groups.get(p).push(el); }
  for (const [p, kids] of groups) {
    if (kids.length < 3) continue;
    const rs = kids.map(k => k.getBoundingClientRect());
    const column = rs.every((r, i) => i === 0 || r.top >= rs[i - 1].bottom - NEAR);
    const row = rs.every((r, i) => i === 0 || r.left >= rs[i - 1].right - NEAR);
    if (column) {
      const lefts = rs.map(r => r.left), base = lefts.slice().sort((a, b) => a - b)[Math.floor(lefts.length / 2)];
      kids.forEach((k, i) => { const d = Math.abs(lefts[i] - base); if (d > TOL && d <= NEAR) add('alignment', k, `left edge is ${d.toFixed(1)} px off from its neighbours`); });
      const gaps = rs.slice(1).map((r, i) => r.top - rs[i].bottom);
      if (gaps.length >= 2) { const lo = Math.min(...gaps), hi = Math.max(...gaps); if (hi - lo > 6 && hi - lo < 40) add('spacing', p, `gaps between items vary from ${Math.round(lo)} to ${Math.round(hi)} px`); }
    } else if (row) {
      const tops = rs.map(r => r.top), base = tops.slice().sort((a, b) => a - b)[Math.floor(tops.length / 2)];
      kids.forEach((k, i) => { const d = Math.abs(tops[i] - base); if (d > TOL && d <= NEAR) add('alignment', k, `top edge is ${d.toFixed(1)} px off from its neighbours`); });
      const gaps = rs.slice(1).map((r, i) => r.left - rs[i].right);
      if (gaps.length >= 2) { const lo = Math.min(...gaps), hi = Math.max(...gaps); if (hi - lo > 6 && hi - lo < 40) add('spacing', p, `gaps between items vary from ${Math.round(lo)} to ${Math.round(hi)} px`); }
    }
  }

  // 5. controls too small to hit
  for (const el of items.filter(isControl)) {
    const r = el.getBoundingClientRect();
    if ((r.width < MIN_TARGET || r.height < MIN_TARGET) && !(el.tagName === 'A' && ownText(el) && r.height >= 16))
      add('small target', el, `${Math.round(r.width)}x${Math.round(r.height)} px; at least ${MIN_TARGET}x${MIN_TARGET} is easy to hit`);
  }

  const weights = { overlap: 3, contrast: 2, 'off screen': 3, 'cut off': 2, alignment: 0.5, spacing: 0.5, 'small target': 1, 'small text': 1 };
  const penalty = problems.reduce((s, p) => s + (weights[p.kind] || 1), 0);
  const counts = {}; problems.forEach(p => counts[p.kind] = (counts[p.kind] || 0) + 1);
  const out = document.createElement('pre'); out.id = '__layout_report'; out.style.display = 'none';
  out.textContent = JSON.stringify({ viewport: { w: vw, h: vh }, checked: items.length, counts, score: Math.max(0, +(1 - penalty / Math.max(10, items.length)).toFixed(3)), problems: problems.slice(0, 100) });
  document.body.appendChild(out);
})();
