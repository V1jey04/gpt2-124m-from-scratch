"""Build the chat fine-tune token shards from open instruction/chat datasets.

Datasets (choose with --datasets):
  smoltalk  HuggingFaceTB/smoltalk, config "all", train split (data/all/train-*.parquet, column `messages`)
  dolly     databricks/databricks-dolly-15k (databricks-dolly-15k.jsonl: instruction/context/response)
  oasst1    OpenAssistant/oasst1 (data/*.parquet, train+validation), English only,
            one conversation per tree following the best-ranked reply (rank 0) at each step

Every conversation is normalized to alternating user/assistant messages
(system text is prepended to the first user message plus a blank line), then
shuffled with a fixed seed, encoded with tok.encode_chat(messages), filtered by
--max-tokens and written as raw uint16:
  val_00000.bin    first --val-frac of the shuffled, kept conversations
  train_00000.bin  ... the rest, ~--shard-tokens per shard
  meta.json        counts, drop reasons, token totals

Example:
  python data/prepare_chat.py --tokenizer tokenizer.json --out /kaggle/working/chat
"""
import argparse
import collections
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.common import (DEFAULT_TOKENIZER_CLASS, ShardWriter, dolly_to_messages, fmt_tokens,  # noqa: E402
                         hf_download, hf_list_files, normalize_conversation, oasst_best_paths,
                         parquet_column_batches, tokenize_conversations, write_tokens)

import numpy as np  # noqa: E402

ALL_DATASETS = ("smoltalk", "dolly", "oasst1")


# ---------------------------------------------------------------- download layer

def load_smoltalk(dl_dir, revision=None):
    repo = "HuggingFaceTB/smoltalk"
    convs = []
    for fname in hf_list_files(repo, "data/all/train-", ".parquet", revision):
        print(f"  smoltalk: {fname}", flush=True)
        local = hf_download(repo, fname, dl_dir, revision)
        for batch in parquet_column_batches(local, ["messages"]):
            convs.extend(row["messages"] for row in batch)
    return convs


def load_dolly(dl_dir, revision=None):
    repo = "databricks/databricks-dolly-15k"
    fname = "databricks-dolly-15k.jsonl"
    local = hf_download(repo, fname, dl_dir, revision)
    convs = []
    with open(local, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            for k in ("instruction", "context", "response"):
                if k not in row:
                    raise SystemExit(f"{fname} line {i + 1}: missing field {k!r}; fields are {sorted(row)}")
            convs.append(dolly_to_messages(row))
    return convs


OASST_COLUMNS = ["message_id", "parent_id", "role", "text", "lang", "rank",
                 "deleted", "review_result", "created_date", "message_tree_id"]


def load_oasst1(dl_dir, revision=None):
    repo = "OpenAssistant/oasst1"
    rows = []
    for fname in hf_list_files(repo, "data/", ".parquet", revision):  # train + validation
        print(f"  oasst1: {fname}", flush=True)
        local = hf_download(repo, fname, dl_dir, revision)
        for batch in parquet_column_batches(local, OASST_COLUMNS):
            rows.extend(batch)
    roles = {r["role"] for r in rows}
    if not roles <= {"prompter", "assistant"}:
        raise SystemExit(f"oasst1: unexpected roles {roles}")
    return oasst_best_paths(rows, lang="en")


LOADERS = {"smoltalk": load_smoltalk, "dolly": load_dolly, "oasst1": load_oasst1}


# ---------------------------------------------------------------- core

def build_chat(sources, out_dir, tokenizer_path, *, max_tokens=1024, val_frac=0.01, seed=1234,
               max_examples=None, shard_tokens=100_000_000, min_shard_tokens=10_000_000,
               floor_tokens=500_000, workers=4, class_spec=DEFAULT_TOKENIZER_CLASS, log=print):
    """sources: {dataset_name: [raw message list, ...]} -> shards in out_dir. Returns stats."""
    max_examples = max_examples or {}
    stats = {"datasets": {}, "max_tokens": max_tokens, "seed": seed, "val_frac": val_frac}
    pool = []  # (dataset, messages)
    for name in sorted(sources):
        drops = collections.Counter()
        kept = []
        for raw in sources[name]:
            msgs, reason = normalize_conversation(raw)
            if msgs is None:
                drops[reason] += 1
            else:
                kept.append(msgs)
        cap = max_examples.get(name)
        n_norm = len(kept)
        if cap is not None and len(kept) > cap:
            kept = random.Random(f"{seed}-{name}").sample(kept, cap)
        stats["datasets"][name] = {"raw": len(sources[name]), "normalized": n_norm,
                                   "after_cap": len(kept), "dropped": dict(drops),
                                   "too_long": 0, "kept": 0, "tokens": 0}
        log(f"{name}: {len(sources[name]):,} raw -> {n_norm:,} valid -> {len(kept):,} after cap; "
            f"dropped {dict(drops)}")
        pool.extend((name, m) for m in kept)

    random.Random(seed).shuffle(pool)
    log(f"tokenizing {len(pool):,} conversations with {workers} workers ...")
    t0 = time.time()
    encoded = []  # uint16 arrays in shuffled order
    for (name, _), arr in zip(pool, tokenize_conversations((m for _, m in pool), tokenizer_path,
                                                           class_spec, workers)):
        d = stats["datasets"][name]
        if arr.size > max_tokens:
            d["too_long"] += 1
            continue
        d["kept"] += 1
        d["tokens"] += int(arr.size)
        encoded.append(arr)
    dt = time.time() - t0
    total = sum(a.size for a in encoded)
    log(f"encoded in {dt:.0f}s ({total / max(dt, 1e-9):,.0f} tok/s); kept {len(encoded):,} "
        f"conversations, {fmt_tokens(total)} tokens")
    if not encoded:
        raise SystemExit("no conversations left after filtering")

    n_val = max(1, math.ceil(val_frac * len(encoded))) if val_frac > 0 else 0
    os.makedirs(out_dir, exist_ok=True)
    if n_val:
        val = np.concatenate(encoded[:n_val])
        write_tokens(os.path.join(out_dir, "val_00000.bin"), val)
        log(f"  wrote val_00000.bin: {val.size:,} tokens ({n_val:,} conversations)")
    writer = ShardWriter(out_dir, lambda k: f"train_{k:05d}.bin", shard_tokens,
                         min_shard_tokens, floor_tokens, log=log)
    for arr in encoded[n_val:]:
        writer.add(arr)
    writer.close()
    stats.update({
        "conversations": len(encoded), "val_conversations": n_val,
        "val_tokens": int(sum(a.size for a in encoded[:n_val])),
        "train_conversations": len(encoded) - n_val, "train_tokens": writer.total_written,
        "dropped_tokens": writer.dropped_tokens,
        "shards": [[os.path.basename(p), n] for p, n in writer.written],
    })
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def parse_caps(items):
    caps = {}
    for it in items or []:
        name, _, n = it.partition("=")
        if name not in ALL_DATASETS or not n.isdigit():
            raise SystemExit(f"bad --max-examples entry {it!r}; use e.g. smoltalk=300000")
        caps[name] = int(n)
    return caps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="smoltalk,dolly,oasst1",
                    help=f"comma separated subset of {','.join(ALL_DATASETS)}")
    ap.add_argument("--max-examples", nargs="*", metavar="NAME=N",
                    help="cap per dataset (random sample with the seed), e.g. smoltalk=300000")
    ap.add_argument("--max-tokens", type=int, default=1024, help="drop conversations longer than this")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--min-shard-tokens", type=int, default=10_000_000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--download-dir", default="/tmp/chat_dl")
    ap.add_argument("--tokenizer-class", default=DEFAULT_TOKENIZER_CLASS, help=argparse.SUPPRESS)
    args = ap.parse_args()

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    bad = [n for n in names if n not in ALL_DATASETS]
    if bad:
        raise SystemExit(f"unknown datasets {bad}; choose from {ALL_DATASETS}")
    sources = {}
    for n in names:
        print(f"loading {n} ...", flush=True)
        sources[n] = LOADERS[n](os.path.join(args.download_dir, n))
    stats = build_chat(sources, args.out, args.tokenizer, max_tokens=args.max_tokens,
                       val_frac=args.val_frac, seed=args.seed, max_examples=parse_caps(args.max_examples),
                       shard_tokens=args.shard_tokens, min_shard_tokens=args.min_shard_tokens,
                       workers=args.workers, class_spec=args.tokenizer_class)
    print("\n==== summary ====")
    for n, d in stats["datasets"].items():
        print(f"{n:9s} kept {d['kept']:,} conversations ({fmt_tokens(d['tokens'])} tokens); "
              f"too long {d['too_long']:,}; invalid {sum(d['dropped'].values()):,}")
    print(f"train: {stats['train_conversations']:,} conversations, {stats['train_tokens']:,} tokens "
          f"in {len(stats['shards'])} shards; val: {stats['val_conversations']:,} conversations, "
          f"{stats['val_tokens']:,} tokens -> {args.out}")


if __name__ == "__main__":
    main()
