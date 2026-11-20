// Byte-level BPE tokenizer: a line-for-line port of tokenizer/bpe.py for the browser.
// Python's Unicode classes are written out so both split text identically:
//   \s  -> Python str.isspace() characters
//   \w  -> letters, numbers and "_"   (Python: isalnum() or "_")
//   \d  -> decimal digits (Nd)

const S = "\\t\\n\\x0b\\x0c\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const PATTERN = new RegExp(
  "'(?:[sdmt]|ll|ve|re)" +                    // contractions
  "| ?[\\p{L}\\p{Nl}\\p{No}]{1,24}" +         // letters: \w minus digits and "_"
  "| ?\\p{Nd}{1,3}" +                         // digits in groups of up to 3
  `| ?(?:[^${S}\\p{L}\\p{N}_]|_){1,8}` +      // punctuation and symbols
  `|[${S}]{1,8}(?![^${S}])` +                 // whitespace, leaving the last space for the next word
  `|[${S}]{1,8}`,
  "gu"
);

export class BPETokenizer {
  constructor(merges, specialTokens) {
    this.ranks = new Map();                    // (left << 16 | right) -> merged id
    this.vocab = [];                           // id -> Uint8Array
    for (let i = 0; i < 256; i++) this.vocab.push(new Uint8Array([i]));
    for (const [a, b] of merges) {
      this.ranks.set(a * 65536 + b, this.vocab.length);
      const va = this.vocab[a], vb = this.vocab[b], m = new Uint8Array(va.length + vb.length);
      m.set(va); m.set(vb, va.length);
      this.vocab.push(m);
    }
    this.nBase = this.vocab.length;
    this.special = {};
    specialTokens.forEach((s, i) => { this.special[s] = this.nBase + i; });
    this.bosId = this.special["<|bos|>"];
    this.userId = this.special["<|user|>"];
    this.assistantId = this.special["<|assistant|>"];
    this.endId = this.special["<|end|>"];
    this.enc = new TextEncoder();
    this.dec = new TextDecoder("utf-8");
    this.cache = new Map();
  }

  static fromJSON(d) { return new BPETokenizer(d.merges, d.special_tokens); }

  bpe(chunk) {
    let ids = Array.from(this.enc.encode(chunk));
    while (ids.length >= 2) {
      let best = -1, bestId = Infinity;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < bestId) { bestId = r; best = i; }
      }
      if (best < 0) break;
      const a = ids[best], b = ids[best + 1], out = [];
      for (let i = 0; i < ids.length;) {
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { out.push(bestId); i += 2; }
        else { out.push(ids[i]); i += 1; }
      }
      ids = out;
    }
    return ids;
  }

  // plain text -> ids; special tokens are never parsed out of text
  encode(text) {
    const out = [];
    for (const m of text.matchAll(PATTERN)) {
      const chunk = m[0];
      let ids = this.cache.get(chunk);
      if (!ids) { ids = this.bpe(chunk); if (this.cache.size < 200000) this.cache.set(chunk, ids); }
      for (const x of ids) out.push(x);
    }
    return out;
  }

  // <|bos|> <|user|> ... <|end|> <|assistant|> ... <|end|> ... [<|assistant|>]
  encodeChat(messages, addGenerationPrompt = false) {
    const ids = [this.bosId];
    for (const m of messages) {
      ids.push(m.role === "user" ? this.userId : this.assistantId);
      for (const x of this.encode(m.content)) ids.push(x);
      ids.push(this.endId);
    }
    if (addGenerationPrompt) ids.push(this.assistantId);
    return ids;
  }

  decode(ids) {
    let n = 0;
    for (const i of ids) if (i < this.nBase) n += this.vocab[i].length;
    const buf = new Uint8Array(n);
    let o = 0;
    for (const i of ids) if (i < this.nBase) { buf.set(this.vocab[i], o); o += this.vocab[i].length; }
    return this.dec.decode(buf);
  }
}
