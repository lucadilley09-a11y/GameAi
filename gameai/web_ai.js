// The Game AI model in plain JavaScript: same math as gameai/model.py, with a KV cache
// so each new token only costs one step. Works in browsers and Node.
function createAI(manifest, buffer) {
  const f32 = new Float32Array(buffer), W = {};
  for (const t of manifest.tensors) W[t.name] = f32.subarray(t.offset, t.offset + t.size);
  const c = manifest.config, d = c.d_model, H = c.n_head, KV = c.n_kv_head, hd = d / H, rep = H / KV, L = c.n_layer;
  const V = c.vocab_size, hid = W['blocks.0.mlp.gate.weight'].length / d, tk = manifest.tokenizer;
  const inv = Float32Array.from({ length: hd / 2 }, (_, i) => 1 / Math.pow(10000, (2 * i) / hd));
  const stoi = new Map(tk.itos.map((s, i) => [s, i]));
  const mv = (w, rows, cols, x) => { const o = new Float32Array(rows); for (let r = 0; r < rows; r++) { let s = 0; const b = r * cols; for (let k = 0; k < cols; k++) s += w[b + k] * x[k]; o[r] = s; } return o; };
  const rms = (x, w) => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; const k = 1 / Math.sqrt(s / x.length + 1e-6); const o = new Float32Array(x.length); for (let i = 0; i < x.length; i++) o[i] = x[i] * k * w[i]; return o; };
  const rope = (v, heads, p) => { for (let h = 0; h < heads; h++) for (let i = 0; i < hd / 2; i++) { const a = p * inv[i], co = Math.cos(a), si = Math.sin(a), j = h * hd + 2 * i, x1 = v[j], x2 = v[j + 1]; v[j] = x1 * co - x2 * si; v[j + 1] = x1 * si + x2 * co; } };
  let cacheK, cacheV, pos;
  const reset = () => { cacheK = Array.from({ length: L }, () => []); cacheV = Array.from({ length: L }, () => []); pos = 0; };
  function push(t) {
    let x = Float32Array.from(W['emb.weight'].subarray(t * d, (t + 1) * d));
    for (let l = 0; l < L; l++) {
      const p = `blocks.${l}.`, h = rms(x, W[p + 'n1.weight']);
      const q = mv(W[p + 'attn.q.weight'], H * hd, d, h), k = mv(W[p + 'attn.k.weight'], KV * hd, d, h), v = mv(W[p + 'attn.v.weight'], KV * hd, d, h);
      rope(q, H, pos); rope(k, KV, pos); cacheK[l].push(k); cacheV[l].push(v);
      const att = new Float32Array(H * hd), n = cacheK[l].length, sc = new Float32Array(n), scale = 1 / Math.sqrt(hd);
      for (let hh = 0; hh < H; hh++) {
        const kv = Math.floor(hh / rep) * hd, qo = hh * hd; let m = -Infinity;
        for (let j = 0; j < n; j++) { let s = 0; const kk = cacheK[l][j]; for (let i = 0; i < hd; i++) s += q[qo + i] * kk[kv + i]; sc[j] = s * scale; if (sc[j] > m) m = sc[j]; }
        let z = 0; for (let j = 0; j < n; j++) { sc[j] = Math.exp(sc[j] - m); z += sc[j]; }
        for (let j = 0; j < n; j++) { const w = sc[j] / z, vv = cacheV[l][j]; for (let i = 0; i < hd; i++) att[qo + i] += w * vv[kv + i]; }
      }
      const o = mv(W[p + 'attn.o.weight'], d, H * hd, att); for (let i = 0; i < d; i++) x[i] += o[i];
      const h2 = rms(x, W[p + 'n2.weight']), g = mv(W[p + 'mlp.gate.weight'], hid, d, h2), u = mv(W[p + 'mlp.up.weight'], hid, d, h2);
      for (let i = 0; i < hid; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
      const dn = mv(W[p + 'mlp.down.weight'], d, hid, g); for (let i = 0; i < d; i++) x[i] += dn[i];
    }
    pos++;
    return mv(W['head.weight'], V, d, rms(x, W['norm.weight']));
  }
  function encode(text) {
    const t = text.toLowerCase().split(/\s+/).filter(Boolean).join(' ').slice(0, tk.max_prompt);
    return [tk.bos, ...[...t].map(ch => stoi.has(ch) ? stoi.get(ch) : tk.unk), tk.sep];
  }
  // grammar-constrained decoding: slot i may only use field i's values
  const cons = tk.constraints || {};
  const allowedVals = (f, chosen, opts) => { const c = cons[f]; if (c && chosen[c[0]] in c[1]) return c[1][chosen[c[0]]]; return opts; };
  // grammar-constrained decoding: slot i may only use field i's values, and rule parts must fit together
  function decode(prompt) {
    reset(); let logits; for (const t of encode(prompt)) logits = push(t);
    const out = {}, chosen = {};
    tk.field_names.forEach((f, slot) => {
      const ids = tk.field_ids[slot], vals = ids.map(i => tk.itos[i].split('=')[1]), ok = allowedVals(f, chosen, vals);
      let m = -Infinity; ids.forEach((i, j) => { if (ok.includes(vals[j])) m = Math.max(m, logits[i]); });
      let z = 0; const pr = ids.map((i, j) => { if (!ok.includes(vals[j])) return 0; const e = Math.exp(logits[i] - m); z += e; return e; });
      let b = 0; pr.forEach((v, j) => { if (v > pr[b]) b = j; });
      out[f] = { value: vals[b], conf: pr[b] / z }; chosen[f] = vals[b];
      logits = push(ids[b]);
    });
    return out;
  }
  const recipeFields = tk.recipe_fields || tk.field_names;
  return { decode, params: manifest.params, config: c, synonyms: manifest.synonyms, fields: recipeFields, allFields: tk.field_names,
           options: recipeFields.map(f => { const s = tk.field_names.indexOf(f); return tk.field_ids[s].map(i => tk.itos[i].split('=')[1]); }) };
}
if (typeof module !== 'undefined') module.exports = { createAI };
