// Runs the model off the main thread: download (cached), tokenize, generate with a KV cache.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs";
import { BPETokenizer } from "./tokenizer.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const L = 12, H = 12, D = 64, CTX = 1024, VOCAB = 50004;   // ids 50004.. are padding rows
let session = null, tok = null, stopFlag = false;

async function fetchCached(url, onProgress) {
  let cache = null;
  try { cache = await caches.open("gpt2-124m-from-scratch-v1"); } catch (e) { /* no Cache API */ }
  if (cache) { const hit = await cache.match(url); if (hit) { onProgress(1, 1); return new Uint8Array(await hit.arrayBuffer()); } }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed (${res.status}) for ${url}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length; onProgress(got, total);
  }
  const buf = new Uint8Array(got);
  let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
  if (cache) { try { await cache.put(url, new Response(buf)); } catch (e) { /* quota: fine, just no cache */ } }
  return buf;
}

async function load(base) {
  const tj = await fetchCached(base + "tokenizer.json", () => {});
  tok = BPETokenizer.fromJSON(JSON.parse(new TextDecoder().decode(tj)));
  const model = await fetchCached(base + "onnx/model_int8.onnx", (got, total) => postMessage({ type: "progress", got, total }));
  postMessage({ type: "status", text: "Starting the model…" });
  session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  postMessage({ type: "ready", threads: ort.env.wasm.numThreads });
}

function sample(logits, seen, { temperature, topK = 50, topP = 0.95, repetitionPenalty = 1.1 }) {
  const x = new Float32Array(VOCAB);
  for (let i = 0; i < VOCAB; i++) x[i] = logits[i];
  for (const i of seen) x[i] = x[i] > 0 ? x[i] / repetitionPenalty : x[i] * repetitionPenalty;
  if (temperature <= 0) { let b = 0; for (let i = 1; i < VOCAB; i++) if (x[i] > x[b]) b = i; return b; }
  const idx = Array.from({ length: VOCAB }, (_, i) => i).sort((a, b) => x[b] - x[a]).slice(0, topK);
  const m = x[idx[0]];
  let p = idx.map(i => Math.exp((x[i] - m) / temperature));
  const z = p.reduce((a, b) => a + b, 0); p = p.map(v => v / z);
  let cum = 0, keep = 0;
  for (; keep < p.length; keep++) { if (cum > topP) break; cum += p[keep]; }   // always keeps the top token
  const z2 = p.slice(0, keep).reduce((a, b) => a + b, 0);
  let r = Math.random() * z2;
  for (let j = 0; j < keep; j++) { r -= p[j]; if (r <= 0) return idx[j]; }
  return idx[keep - 1];
}

async function generate({ messages, temperature, maxNewTokens }) {
  stopFlag = false;
  let msgs = messages.slice();
  let ids = tok.encodeChat(msgs, true);
  while (ids.length + maxNewTokens > CTX && msgs.length > 1) { msgs = msgs.slice(2); ids = tok.encodeChat(msgs, true); }
  if (ids.length + maxNewTokens > CTX) ids = [tok.bosId, ...ids.slice(-(CTX - maxNewTokens - 1))];

  let pk = new ort.Tensor("float32", new Float32Array(0), [L, 1, H, 0, D]);
  let pv = new ort.Tensor("float32", new Float32Array(0), [L, 1, H, 0, D]);
  let feed = ids;
  const seen = new Set(ids.filter(i => i < VOCAB));
  const out = [];
  let shown = "", t0 = performance.now();
  for (let n = 0; n < maxNewTokens && ids.length + out.length < CTX; n++) {
    const r = await session.run({
      input_ids: new ort.Tensor("int64", BigInt64Array.from(feed, BigInt), [1, feed.length]),
      past_k: pk, past_v: pv,
    });
    pk = r.present_k; pv = r.present_v;
    if (n === 0) t0 = performance.now();   // time decoding only, after the prompt is read
    const next = sample(r.logits.data, seen, { temperature });
    if (next >= tok.bosId || stopFlag) break;  // a special token ends the reply
    out.push(next); seen.add(next); feed = [next];
    const text = tok.decode(out);
    if (!text.endsWith("�")) {           // wait for the rest of a multi-byte character
      shown = text;
      const secs = (performance.now() - t0) / 1000;
      postMessage({ type: "token", text: shown, tps: n > 0 ? n / secs : 0 });
    }
  }
  postMessage({ type: "done", text: shown || tok.decode(out) });
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "load") await load(m.base);
    else if (m.type === "generate") await generate(m);
    else if (m.type === "stop") stopFlag = true;
  } catch (err) {
    postMessage({ type: "error", text: String(err && err.message || err) });
  }
};
