// ForgeGPT in the browser: byte-level BPE tokenizer, sampling, and KV-cached generation
// with ONNX Runtime Web. A JavaScript port of src/forgegpt/tokenizer (Hugging Face
// `tokenizers` byte-level BPE) and src/forgegpt/onnx_infer.py; tests check it produces the
// same token ids as the Python versions.
//
// Works as an ES module in the browser and in Node (tests): `ort` is passed in, so the page
// can load ONNX Runtime Web however it likes.

export const EOT_ID = 0;

// --- tokenizer -----------------------------------------------------------------------

// GPT-2's reversible byte <-> unicode table: every byte maps to a printable character, so
// BPE can work on strings without ever seeing raw control bytes or spaces.
function bytesToUnicode() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const enc = new Map();
  const dec = new Map();
  bs.forEach((b, i) => {
    const ch = String.fromCodePoint(cs[i]);
    enc.set(b, ch);
    dec.set(ch, b);
  });
  return { enc, dec };
}

// The ByteLevel pre-tokenizer's split pattern (same as GPT-2).
const PRE_TOKENIZE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export class Tokenizer {
  constructor(json) {
    const model = json.model;
    this.vocab = new Map(Object.entries(model.vocab));
    this.idToToken = new Map();
    for (const [tok, id] of this.vocab) this.idToToken.set(id, tok);
    this.ranks = new Map();
    model.merges.forEach((m, i) => {
      const [a, b] = Array.isArray(m) ? m : m.split(" ");
      this.ranks.set(a + " " + b, i);
    });
    this.added = json.added_tokens.map((t) => ({ content: t.content, id: t.id, special: t.special }));
    for (const t of this.added) this.idToToken.set(t.id, t.content);
    this.special = new Set(this.added.filter((t) => t.special).map((t) => t.id));
    const { enc, dec } = bytesToUnicode();
    this.byteEnc = enc;
    this.byteDec = dec;
    this.utf8 = new TextEncoder();
    this.cache = new Map();
  }

  static async fromUrl(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`could not load tokenizer (${res.status})`);
    return new Tokenizer(await res.json());
  }

  // Merge the lowest-ranked adjacent pair until no known merge is left.
  bpe(word) {
    const cached = this.cache.get(word);
    if (cached) return cached;
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) break;
      const a = parts[best];
      const b = parts[best + 1];
      const merged = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) {
          merged.push(a + b);
          i++;
        } else {
          merged.push(parts[i]);
        }
      }
      parts = merged;
    }
    if (this.cache.size < 50000) this.cache.set(word, parts);
    return parts;
  }

  encodePlain(text, out) {
    for (const m of text.matchAll(PRE_TOKENIZE)) {
      let mapped = "";
      for (const byte of this.utf8.encode(m[0])) mapped += this.byteEnc.get(byte);
      for (const piece of this.bpe(mapped)) out.push(this.vocab.get(piece));
    }
  }

  // Added tokens ([phone], <|endoftext|>, ...) are matched in the raw text first.
  encode(text) {
    const ids = [];
    let rest = text;
    while (rest.length) {
      let hit = null;
      for (const t of this.added) {
        const at = rest.indexOf(t.content);
        if (at >= 0 && (hit === null || at < hit.at)) hit = { at, t };
      }
      if (!hit) {
        this.encodePlain(rest, ids);
        break;
      }
      if (hit.at > 0) this.encodePlain(rest.slice(0, hit.at), ids);
      ids.push(hit.t.id);
      rest = rest.slice(hit.at + hit.t.content.length);
    }
    return ids;
  }

  // Token strings with the byte-level marks turned back into text (for the token view).
  tokens(ids) {
    return ids.map((id) => this.decode([id]));
  }

  decode(ids) {
    const bytes = [];
    for (const id of ids) {
      if (this.special.has(id)) continue;
      const tok = this.idToToken.get(id) ?? "";
      for (const ch of tok) {
        const b = this.byteDec.get(ch);
        if (b !== undefined) bytes.push(b);
      }
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }
}

// --- sampling --------------------------------------------------------------------------

// Small seeded RNG (mulberry32) so a seed gives a repeatable completion.
export function makeRng(seed) {
  let a = seed === null || seed === undefined ? (Math.random() * 2 ** 32) >>> 0 : seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Same algorithm as forgegpt.onnx_infer.sample_next: temperature, then top-k, then top-p.
export function sampleNext(logits, { temperature = 0.8, topK = 50, topP = 0.95 }, rng) {
  const V = logits.length;
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  let idx = Array.from({ length: V }, (_, i) => i);
  idx.sort((x, y) => logits[y] - logits[x]);
  if (topK && topK < V) idx = idx.slice(0, topK);
  const maxL = logits[idx[0]];
  let probs = idx.map((i) => Math.exp((logits[i] - maxL) / temperature));
  let sum = probs.reduce((s, p) => s + p, 0);
  probs = probs.map((p) => p / sum);
  if (topP && topP < 1) {
    let cum = 0;
    let keep = 0;
    while (keep < probs.length && cum <= topP) {
      cum += probs[keep];
      keep++;
    }
    idx = idx.slice(0, keep);
    probs = probs.slice(0, keep);
    sum = probs.reduce((s, p) => s + p, 0);
    probs = probs.map((p) => p / sum);
  }
  let r = rng();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return idx[i];
  }
  return idx[idx.length - 1];
}

// --- generation ------------------------------------------------------------------------

export class Generator {
  // info: {n_layer, n_head, d_model, block_size} (from model/info.json)
  constructor(ort, session, info) {
    this.ort = ort;
    this.session = session;
    this.L = info.n_layer;
    this.H = info.n_head;
    this.D = info.d_model / info.n_head;
    this.block = info.block_size;
  }

  emptyPast() {
    return new this.ort.Tensor("float32", new Float32Array(0), [this.L, 2, 1, this.H, 0, this.D]);
  }

  async run(ids, past) {
    const input = new this.ort.Tensor(
      "int64",
      BigInt64Array.from(ids.map((i) => BigInt(i))),
      [1, ids.length],
    );
    const out = await this.session.run({ input_ids: input, past });
    const V = out.logits.dims[2];
    const last = out.logits.data.subarray((ids.length - 1) * V, ids.length * V);
    return { logits: last, present: out.present };
  }

  // Yields token ids one at a time. Prefill once, then one token per step with the cache.
  async *stream(promptIds, cfg) {
    const rng = makeRng(cfg.seed);
    let ids = (promptIds.length ? promptIds : [EOT_ID]).slice(-this.block);
    let { logits, present } = await this.run(ids, this.emptyPast());
    for (let step = 0; step < cfg.maxNewTokens; step++) {
      const next = sampleNext(logits, cfg, rng);
      if (next === EOT_ID && !cfg.ignoreEot) return;
      yield next;
      ids.push(next);
      if (ids.length > this.block) {
        // Context full: slide the window and rebuild the cache (positions must restart).
        ids = ids.slice(-this.block);
        ({ logits, present } = await this.run(ids, this.emptyPast()));
      } else {
        ({ logits, present } = await this.run([next], present));
      }
    }
  }
}
