"""
Byte-level BPE tokenizer, written from scratch (standard library only).

Training
    1. Split text into chunks with a regex (words keep their leading space).
    2. Count how often each distinct chunk occurs. BPE then runs over the
       distinct chunks weighted by count, not over the raw text.
    3. Repeatedly merge the most frequent adjacent pair of tokens. Pair counts
       are updated incrementally and the best pair comes from a heap, so each
       merge only touches the chunks that contain that pair.

Encoding
    Split with the same regex, then apply merges in the order they were
    learned. Results are cached per chunk, so common words are a dict lookup.

Token ids
    0..255                       raw bytes
    256..256+num_merges-1        learned merges
    then the special tokens      <|bos|> <|user|> <|assistant|> <|end|>

Usage
    python -m tokenizer.bpe train --input sample.txt --vocab-size 50000 --out tokenizer.json
    python -m tokenizer.bpe demo  --tokenizer tokenizer.json --text "Hello there!"
"""

import argparse
import heapq
import json
import re
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

SPECIAL_TOKENS = ["<|bos|>", "<|user|>", "<|assistant|>", "<|end|>"]

# Chunking pattern. Every character matches some alternative, so joining the
# chunks always gives back the original text. Run lengths are capped so that
# one giant run (a line of ======, a wall of unspaced text) cannot become a
# single enormous chunk.
PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)"        # English contractions: 's 't 'll 've 're
    r"| ?[^\W\d_]{1,24}"           # letters (any script), optional leading space
    r"| ?\d{1,3}"                  # digits in groups of up to 3
    r"| ?(?:[^\s\w]|_){1,8}"       # punctuation and symbols
    r"|\s{1,8}(?!\S)"              # whitespace, leaving the last space for the next word
    r"|\s{1,8}"
)
_PAT = re.compile(PATTERN)


def _count_block(text):
    return Counter(_PAT.findall(text))


def _read_blocks(paths, block_bytes=8 << 20):
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                lines = f.readlines(block_bytes)
                if not lines:
                    break
                yield "".join(lines)


def count_chunks(paths, workers=1):
    """Chunk frequencies over one or more text files."""
    total = Counter()
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as pool:
            for i, c in enumerate(pool.imap_unordered(_count_block, _read_blocks(paths))):
                total.update(c)
                if i % 20 == 0:
                    print(f"  counted {i + 1} blocks, {len(total):,} distinct chunks, {time.time() - t0:.0f}s", flush=True)
    else:
        for i, block in enumerate(_read_blocks(paths)):
            total.update(_count_block(block))
            if i % 20 == 0:
                print(f"  counted {i + 1} blocks, {len(total):,} distinct chunks, {time.time() - t0:.0f}s", flush=True)
    return total


def learn_merges(chunk_counts, num_merges, min_chunk_freq=1, verbose=True):
    """
    Learn BPE merges from {chunk_string: count}.
    Returns a list of (left_id, right_id) pairs; merge i creates token id 256 + i.
    """
    words, freqs = [], []
    for chunk, n in chunk_counts.items():
        if n < min_chunk_freq:
            continue
        b = chunk.encode("utf-8")
        if len(b) < 2:
            continue
        words.append(list(b))
        freqs.append(n)

    pair_counts = defaultdict(int)     # pair -> weighted frequency
    pair_words = defaultdict(set)      # pair -> indices of words containing it
    for wi, w in enumerate(words):
        f = freqs[wi]
        for pair in zip(w, w[1:]):
            pair_counts[pair] += f
            pair_words[pair].add(wi)

    # Max-heap with lazy deletion: an entry is valid only if its count still
    # matches pair_counts. Every count change pushes a fresh entry.
    heap = [(-c, pair) for pair, c in pair_counts.items()]
    heapq.heapify(heap)

    merges = []
    t0 = time.time()
    while len(merges) < num_merges:
        best = None
        while heap:
            neg, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) == -neg and neg < 0:
                best = pair
                break
        if best is None:
            break   # nothing left to merge (corpus too small for this vocab size)

        a, b = best
        new_id = 256 + len(merges)
        merges.append(best)
        changed = set()

        for wi in list(pair_words[best]):
            w = words[wi]
            f = freqs[wi]
            old_pairs = list(zip(w, w[1:]))
            for p in old_pairs:
                pair_counts[p] -= f
            # rebuild the word with (a, b) replaced by new_id
            nw = []
            i, n = 0, len(w)
            while i < n:
                if i < n - 1 and w[i] == a and w[i + 1] == b:
                    nw.append(new_id)
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            words[wi] = nw
            new_pairs = list(zip(nw, nw[1:]))
            for p in new_pairs:
                pair_counts[p] += f
            old_set, new_set = set(old_pairs), set(new_pairs)
            for p in old_set - new_set:
                pair_words[p].discard(wi)
            for p in new_set - old_set:
                pair_words[p].add(wi)
            changed |= old_set
            changed |= new_set

        del pair_counts[best]
        del pair_words[best]
        changed.discard(best)
        for p in changed:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
            elif p in pair_counts:
                del pair_counts[p]
                pair_words.pop(p, None)

        if verbose and (len(merges) % 1000 == 0 or len(merges) == num_merges):
            print(f"  merge {len(merges):,}/{num_merges:,}  {time.time() - t0:.0f}s", flush=True)
    return merges


class BPETokenizer:
    def __init__(self, merges, special_tokens=None, cache_limit=2_000_000):
        self.merges = [tuple(m) for m in merges]
        self.special_tokens = list(special_tokens if special_tokens is not None else SPECIAL_TOKENS)

        self.vocab = [bytes([i]) for i in range(256)]          # id -> bytes
        self.ranks = {}                                        # (left, right) -> merged id
        for a, b in self.merges:
            self.ranks[(a, b)] = len(self.vocab)
            self.vocab.append(self.vocab[a] + self.vocab[b])

        self.n_base = len(self.vocab)
        self.special = {s: self.n_base + i for i, s in enumerate(self.special_tokens)}   # name -> id
        self._special_by_id = {i: s for s, i in self.special.items()}
        self.vocab_size = self.n_base + len(self.special_tokens)
        assert self.vocab_size <= 65536, "ids must fit in uint16"

        self._cache = {}
        self._cache_limit = cache_limit

    # ---- convenience ids -------------------------------------------------
    @property
    def bos_id(self):
        return self.special["<|bos|>"]

    @property
    def user_id(self):
        return self.special["<|user|>"]

    @property
    def assistant_id(self):
        return self.special["<|assistant|>"]

    @property
    def end_id(self):
        return self.special["<|end|>"]

    # ---- training --------------------------------------------------------
    @classmethod
    def train(cls, paths, vocab_size=50000, min_chunk_freq=2, workers=1, verbose=True):
        """vocab_size counts the 256 bytes plus merges; special tokens are added on top."""
        if isinstance(paths, str):
            paths = [paths]
        if verbose:
            print("counting chunks...", flush=True)
        counts = count_chunks(paths, workers=workers)
        if verbose:
            print(f"{len(counts):,} distinct chunks; learning {vocab_size - 256:,} merges...", flush=True)
        merges = learn_merges(counts, vocab_size - 256, min_chunk_freq=min_chunk_freq, verbose=verbose)
        if verbose and len(merges) < vocab_size - 256:
            print(f"note: corpus only supported {len(merges):,} merges")
        return cls(merges)

    # ---- encode / decode -------------------------------------------------
    def _bpe(self, chunk):
        ids = list(chunk.encode("utf-8"))
        ranks = self.ranks
        while len(ids) >= 2:
            # the earliest-learned merge present in the chunk goes first
            best_id = None
            for p in zip(ids, ids[1:]):
                r = ranks.get(p)
                if r is not None and (best_id is None or r < best_id):
                    best_id, best = r, p
            if best_id is None:
                break
            a, b = best
            out = []
            i, n = 0, len(ids)
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(best_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def encode(self, text):
        """Plain text -> token ids. Special tokens are never parsed out of the text."""
        out = []
        cache = self._cache
        for chunk in _PAT.findall(text):
            ids = cache.get(chunk)
            if ids is None:
                ids = self._bpe(chunk)
                if len(cache) < self._cache_limit:
                    cache[chunk] = ids
            out.extend(ids)
        return out

    def encode_chat(self, messages, add_generation_prompt=False):
        """
        messages: [{"role": "user" | "assistant", "content": str}, ...]
        Layout:   <|bos|> <|user|> ... <|end|> <|assistant|> ... <|end|> ...
        With add_generation_prompt=True a trailing <|assistant|> is added so the
        model writes the next reply.
        """
        ids = [self.bos_id]
        for m in messages:
            role = m["role"]
            if role == "user":
                ids.append(self.user_id)
            elif role == "assistant":
                ids.append(self.assistant_id)
            else:
                raise ValueError(f"unknown role: {role!r}")
            ids.extend(self.encode(m["content"]))
            ids.append(self.end_id)
        if add_generation_prompt:
            ids.append(self.assistant_id)
        return ids

    def decode(self, ids, skip_special=False):
        parts = []
        for i in ids:
            i = int(i)
            if i < self.n_base:
                parts.append(self.vocab[i])
            elif i in self._special_by_id:
                if not skip_special:
                    parts.append(self._special_by_id[i].encode("utf-8"))
            # ids past the vocab (the model's padding rows) decode to nothing
        return b"".join(parts).decode("utf-8", errors="replace")

    # ---- save / load -----------------------------------------------------
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "version": 1,
                "pattern": PATTERN,
                "merges": [list(m) for m in self.merges],
                "special_tokens": self.special_tokens,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("pattern") != PATTERN:
            raise ValueError("tokenizer file was made with a different chunking pattern")
        return cls(d["merges"], d["special_tokens"])


def main():
    ap = argparse.ArgumentParser(description="From-scratch byte-level BPE tokenizer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="learn a vocabulary from text files")
    t.add_argument("--input", nargs="+", required=True, help="UTF-8 text file(s)")
    t.add_argument("--vocab-size", type=int, default=50000,
                   help="bytes + merges (the 4 special tokens are added on top)")
    t.add_argument("--min-chunk-freq", type=int, default=2,
                   help="ignore chunks seen fewer times than this (saves time and memory)")
    t.add_argument("--workers", type=int, default=4, help="processes for the counting pass")
    t.add_argument("--out", default="tokenizer.json")

    d = sub.add_parser("demo", help="encode and decode a string")
    d.add_argument("--tokenizer", default="tokenizer.json")
    d.add_argument("--text", required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        t0 = time.time()
        tok = BPETokenizer.train(args.input, args.vocab_size, args.min_chunk_freq, args.workers)
        tok.save(args.out)
        print(f"saved {args.out}: vocab_size={tok.vocab_size:,} "
              f"({tok.n_base:,} BPE + {len(tok.special_tokens)} special), {time.time() - t0:.0f}s")
        print("special ids:", tok.special)
    else:
        tok = BPETokenizer.load(args.tokenizer)
        ids = tok.encode(args.text)
        print(ids)
        print([tok.decode([i]) for i in ids])
        assert tok.decode(ids) == args.text


if __name__ == "__main__":
    main()
