"""Shared pieces of the data pipeline.

Kept separate from the download code so it can be tested offline:
  - tokenizer loading (one tokenizer per worker process)
  - bounded, order-preserving parallel tokenization
  - ShardWriter: writes raw little-endian uint16 token shards, no header
  - conversation normalization for the chat fine-tune set
  - thin Hugging Face Hub helpers (list files / download one file)
"""
import collections
import gc
import importlib
import multiprocessing as mp
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_TOKENIZER_CLASS = "tokenizer.bpe:BPETokenizer"
SPECIAL_NAMES = ("<|bos|>", "<|user|>", "<|assistant|>", "<|end|>")
TOKEN_DTYPE = np.dtype("<u2")  # little-endian uint16


# ---------------------------------------------------------------- tokenizer

def load_tokenizer(path, class_spec=DEFAULT_TOKENIZER_CLASS):
    """Load a tokenizer given "module:Class" (Class.load(path) is called)."""
    mod_name, _, cls_name = class_spec.partition(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    tok = cls.load(path)
    missing = [s for s in SPECIAL_NAMES if s not in tok.special]
    if missing:
        raise ValueError(f"tokenizer {path} is missing special tokens {missing}")
    if tok.vocab_size > 65536:
        raise ValueError(f"vocab_size {tok.vocab_size} does not fit in uint16")
    return tok


_WORKER_TOK = None


def _worker_init(path, class_spec):
    global _WORKER_TOK
    _WORKER_TOK = load_tokenizer(path, class_spec)


def _to_u16(ids):
    a = np.asarray(ids, dtype=np.int64)
    if a.size and (a.min() < 0 or a.max() > 65535):
        raise ValueError("token id outside uint16 range")
    return a.astype(TOKEN_DTYPE)


def _encode_docs(texts):
    """Worker: encode a batch of plain-text docs -> (one uint16 array, n_docs).
    Every doc becomes <|bos|> + encode(text)."""
    tok = _WORKER_TOK
    bos = tok.special["<|bos|>"]
    out = []
    for t in texts:
        out.append(bos)
        out.extend(tok.encode(t))
    return _to_u16(out), len(texts)


def _encode_convs(convs):
    """Worker: encode a batch of normalized conversations -> list of uint16 arrays."""
    tok = _WORKER_TOK
    return [_to_u16(tok.encode_chat(m)) for m in convs]


def _batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def parallel_map_ordered(fn, batches, tokenizer_path, class_spec, workers, max_inflight=None):
    """Apply fn to every batch in a worker pool, yielding results in input order.

    Unlike Pool.imap, at most `max_inflight` batches are pulled from the input
    iterator ahead of the consumer, so a 2GB parquet file is never fully
    loaded into memory just because the reader is faster than the workers.
    workers <= 0 runs inline (handy for debugging)."""
    if workers <= 0:
        _worker_init(tokenizer_path, class_spec)
        for b in batches:
            yield fn(b)
        return
    max_inflight = max_inflight or workers * 4
    # fork is fastest on Linux (Kaggle); fall back to the platform default elsewhere.
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    # Move the parent's objects (e.g. a million loaded conversations) out of the
    # GC's reach so forked workers don't copy those pages when they collect.
    gc.collect()
    gc.freeze()
    try:
        with ctx.Pool(workers, initializer=_worker_init, initargs=(tokenizer_path, class_spec)) as pool:
            gc.unfreeze()
            pending = collections.deque()
            for b in batches:
                pending.append(pool.apply_async(fn, (b,)))
                if len(pending) >= max_inflight:
                    yield pending.popleft().get()
            while pending:
                yield pending.popleft().get()
    finally:
        gc.unfreeze()


def tokenize_documents(texts, tokenizer_path, class_spec=DEFAULT_TOKENIZER_CLASS,
                       workers=4, docs_per_task=256):
    """Yield (uint16 array, n_docs) for consecutive batches of documents, in order."""
    return parallel_map_ordered(_encode_docs, _batched(texts, docs_per_task),
                                tokenizer_path, class_spec, workers)


def tokenize_conversations(convs, tokenizer_path, class_spec=DEFAULT_TOKENIZER_CLASS,
                           workers=4, convs_per_task=64):
    """Yield one uint16 array per conversation, in input order."""
    for arrs in parallel_map_ordered(_encode_convs, _batched(convs, convs_per_task),
                                     tokenizer_path, class_spec, workers):
        yield from arrs


# ---------------------------------------------------------------- shard writer

def write_tokens(path, arr):
    """Atomically write a uint16 array as raw little-endian bytes (no header)."""
    tmp = path + ".tmp"
    np.ascontiguousarray(arr, dtype=TOKEN_DTYPE).tofile(tmp)
    os.replace(tmp, path)


def read_tokens(path):
    return np.fromfile(path, dtype=TOKEN_DTYPE)


class ShardWriter:
    """Concatenates token arrays and cuts them into shards.

    A full shard (shard_tokens) is only written once at least
    shard_tokens + min_shard_tokens tokens are buffered. So when close() is
    called after at least one shard went out, the leftover is between
    min_shard_tokens and shard_tokens + min_shard_tokens and becomes the last
    (slightly larger) shard: there is never a tiny tail shard. If no full
    shard was ever written, the leftover is written only if it has at least
    floor_tokens tokens, otherwise it is dropped with a warning.

    name_fn(k) gives the file name of the k-th shard written by this writer.
    """

    def __init__(self, out_dir, name_fn, shard_tokens=100_000_000,
                 min_shard_tokens=10_000_000, floor_tokens=500_000, log=print):
        if min_shard_tokens > shard_tokens:
            raise ValueError("min_shard_tokens must be <= shard_tokens")
        self.out_dir = out_dir
        self.name_fn = name_fn
        self.shard_tokens = shard_tokens
        self.min_shard_tokens = min_shard_tokens
        self.floor_tokens = floor_tokens
        self.log = log
        self.buf = []
        self.nbuf = 0
        self.written = []  # (path, n_tokens)
        self.dropped_tokens = 0
        os.makedirs(out_dir, exist_ok=True)

    def _emit(self, arr):
        path = os.path.join(self.out_dir, self.name_fn(len(self.written)))
        write_tokens(path, arr)
        self.written.append((path, int(arr.size)))
        self.log(f"  wrote {os.path.basename(path)}: {arr.size:,} tokens")

    def add(self, arr):
        if arr.size == 0:
            return
        self.buf.append(arr)
        self.nbuf += arr.size
        if self.nbuf >= self.shard_tokens + self.min_shard_tokens:
            data = np.concatenate(self.buf)
            while data.size >= self.shard_tokens + self.min_shard_tokens:
                self._emit(data[:self.shard_tokens])
                data = data[self.shard_tokens:]
            self.buf = [data.copy()]  # copy so the big concat can be freed
            self.nbuf = data.size

    def close(self):
        if self.nbuf:
            data = np.concatenate(self.buf)
            if self.written or data.size >= self.floor_tokens:
                self._emit(data)
            else:
                self.dropped_tokens += int(data.size)
                self.log(f"  WARNING: dropping {data.size:,} leftover tokens "
                         f"(< floor of {self.floor_tokens:,}, nothing to merge into)")
        self.buf, self.nbuf = [], 0
        return self.written

    @property
    def total_written(self):
        return sum(n for _, n in self.written)


# ---------------------------------------------------------------- chat normalization

_ROLE_MAP = {
    "user": "user", "human": "user", "prompter": "user",
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant",
    "system": "system",
}


def normalize_conversation(messages):
    """Normalize a list of {"role", "content"} dicts.

    Returns (messages, None) on success or (None, reason) if dropped.
    Rules: roles mapped to user/assistant/system; leading system messages are
    folded into the first user message as "<system>\\n\\n<user>"; a system
    message anywhere else drops the conversation; contents are stripped and
    must be non-empty; the conversation must start with user and strictly
    alternate; a trailing user turn (no answer) is removed; at least one
    user/assistant pair must remain."""
    if not messages:
        return None, "empty"
    system_parts = []
    out = []
    for m in messages:
        if not isinstance(m, dict):
            return None, "bad_message"
        role = _ROLE_MAP.get(str(m.get("role", "")).strip().lower())
        content = m.get("content")
        if role is None or not isinstance(content, str):
            return None, "bad_message"
        content = content.strip()
        if role == "system":
            if out:
                return None, "system_not_leading"
            if content:
                system_parts.append(content)
            continue
        if not content:
            return None, "empty_content"
        out.append({"role": role, "content": content})
    if not out:
        return None, "empty"
    if out[0]["role"] != "user":
        return None, "not_user_first"
    for i, m in enumerate(out):
        if m["role"] != ("user" if i % 2 == 0 else "assistant"):
            return None, "not_alternating"
    if out[-1]["role"] == "user":
        out.pop()
    if len(out) < 2:
        return None, "no_assistant_reply"
    if system_parts:
        out[0] = {"role": "user", "content": "\n\n".join(system_parts) + "\n\n" + out[0]["content"]}
    return out, None


def dolly_to_messages(row):
    instr = (row.get("instruction") or "").strip()
    ctx = (row.get("context") or "").strip()
    resp = row.get("response") or ""
    user = instr + ("\n\n" + ctx if ctx else "")
    return [{"role": "user", "content": user}, {"role": "assistant", "content": resp}]


def oasst_best_paths(rows, lang="en"):
    """Build one conversation per oasst1 message tree by following, from the
    root prompt, the best-ranked child at every step (lowest `rank`; 0 = best;
    children without a rank sort after ranked ones, ties broken by
    created_date then message_id). Messages that are deleted, failed review
    (review_result is False) or not in `lang` are removed together with their
    subtrees. The path is cut so it ends on an assistant message.

    rows: iterable of dicts with message_id, parent_id, role, text, lang,
    rank, deleted, review_result, created_date (oasst1 parquet columns)."""
    children = collections.defaultdict(list)
    roots = []
    for r in rows:
        if r.get("deleted") is True or r.get("review_result") is False:
            continue
        if lang and r.get("lang") != lang:
            continue
        if r.get("parent_id") is None:
            roots.append(r)
        else:
            children[r["parent_id"]].append(r)

    def key(r):
        rank = r.get("rank")
        return (rank is None, rank if rank is not None else 0,
                str(r.get("created_date") or ""), str(r["message_id"]))

    roots.sort(key=lambda r: str(r["message_id"]))
    convs = []
    for root in roots:
        path = [root]
        node = root
        while children.get(node["message_id"]):
            node = min(children[node["message_id"]], key=key)
            path.append(node)
        while path and path[-1].get("role") != "assistant":
            path.pop()
        if len(path) >= 2:
            convs.append([{"role": r["role"], "content": r["text"]} for r in path])
    return convs


# ---------------------------------------------------------------- hub helpers

def _hub():
    try:
        import huggingface_hub
    except ImportError as e:
        raise SystemExit("huggingface_hub is required for downloading "
                         "(pip install huggingface_hub)") from e
    return huggingface_hub


def hf_list_files(repo_id, prefix="", suffix="", revision=None):
    """Sorted list of files in a dataset repo whose path starts with prefix and ends with suffix."""
    hub = _hub()
    files = hub.HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
    sel = sorted(f for f in files if f.startswith(prefix) and f.endswith(suffix))
    if not sel:
        raise SystemExit(f"no files matching {prefix!r}*{suffix!r} in dataset {repo_id} "
                         f"(revision {revision or 'main'}). The repo layout may have changed; "
                         f"check https://huggingface.co/datasets/{repo_id}/tree/main")
    return sel


def hf_download(repo_id, filename, local_dir, revision=None):
    """Download one file of a dataset repo into local_dir and return its local path."""
    hub = _hub()
    os.makedirs(local_dir, exist_ok=True)
    return hub.hf_hub_download(repo_id, filename, repo_type="dataset",
                               revision=revision, local_dir=local_dir)


def parquet_column_batches(path, columns, batch_rows=2048):
    """Yield lists of row dicts for the given columns of a parquet file, batch by batch."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    have = set(pf.schema_arrow.names)
    missing = [c for c in columns if c not in have]
    if missing:
        raise SystemExit(f"{path}: expected columns {missing} not found; columns are {sorted(have)}")
    for rb in pf.iter_batches(batch_size=batch_rows, columns=columns):
        yield rb.to_pylist()


def fmt_tokens(n):
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)
