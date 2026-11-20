"""Tokenize FineWeb-Edu sample-10BT into uint16 pretraining shards.

The dataset is 14 parquet files (sample/10BT/000_00000.parquet ... 013_00000.parquet
at the time of writing; the list is read from the Hub at runtime). Each file is
downloaded, tokenized and deleted on its own, so a run can cover any range of
files with --file-start/--file-end (end exclusive) and the job can be split over
several Kaggle sessions.

Output (raw little-endian uint16, no header):
  val_00000.bin          first ~--val-tokens tokens of file 0 (only when file 0 is processed)
  train_FFFPP.bin        FFF = parquet file index, PP = shard number within that file
  _done_FFF.json         marker written after a file is fully done; reruns skip it

Shard names only depend on the file index, so outputs of different sessions never
collide and sort in dataset order. Every document is <|bos|> + encode(text).

Example:
  python data/prepare_pretrain.py --tokenizer tokenizer.json --out /kaggle/working/pretrain --file-start 0 --file-end 5
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.common import (DEFAULT_TOKENIZER_CLASS, ShardWriter, fmt_tokens, hf_download,  # noqa: E402
                         hf_list_files, parquet_column_batches, tokenize_documents, write_tokens)

import numpy as np  # noqa: E402

REPO_ID = "HuggingFaceFW/fineweb-edu"
FILE_PREFIX = "sample/10BT/"
MAX_SHARDS_PER_FILE = 100  # PP is two digits


def done_marker(out_dir, file_idx):
    return os.path.join(out_dir, f"_done_{file_idx:03d}.json")


def shard_name(file_idx, k):
    if k >= MAX_SHARDS_PER_FILE:
        raise SystemExit(f"file {file_idx} needs more than {MAX_SHARDS_PER_FILE} shards; "
                         f"raise --shard-tokens")
    return f"train_{file_idx:03d}{k:02d}.bin"


def clean_texts(texts):
    """Light cleaning only: strip, skip empty / non-string."""
    for t in texts:
        if isinstance(t, str):
            t = t.strip()
            if t:
                yield t


def remove_partial_outputs(out_dir, file_idx):
    """Delete leftovers of an interrupted run on this file."""
    pats = [f"train_{file_idx:03d}[0-9][0-9].bin", f"train_{file_idx:03d}[0-9][0-9].bin.tmp"]
    if file_idx == 0:
        pats += ["val_00000.bin", "val_00000.bin.tmp"]
    for p in pats:
        for f in glob.glob(os.path.join(out_dir, p)):
            os.remove(f)


def process_file(texts, file_idx, out_dir, tokenizer_path, *, carve_val, val_tokens=10_000_000,
                 shard_tokens=100_000_000, min_shard_tokens=10_000_000, floor_tokens=500_000,
                 workers=4, docs_per_task=256, class_spec=DEFAULT_TOKENIZER_CLASS,
                 log_every=30.0, log=print):
    """Core: raw texts of one source file -> val (optional) + train shards.

    Returns a stats dict. Independent of where texts come from, so it is
    tested with synthetic documents."""
    remove_partial_outputs(out_dir, file_idx)
    writer = ShardWriter(out_dir, lambda k: shard_name(file_idx, k), shard_tokens,
                         min_shard_tokens, floor_tokens, log=log)
    val_parts, val_n, val_docs = [], 0, 0
    val_done = not carve_val
    docs = tokens = 0
    t0 = last = time.time()
    for arr, ndocs in tokenize_documents(clean_texts(texts), tokenizer_path, class_spec,
                                         workers, docs_per_task):
        docs += ndocs
        tokens += arr.size
        if not val_done:
            # val is cut at a batch (= document) boundary
            val_parts.append(arr)
            val_n += arr.size
            val_docs += ndocs
            if val_n >= val_tokens:
                write_tokens(os.path.join(out_dir, "val_00000.bin"), np.concatenate(val_parts))
                log(f"  wrote val_00000.bin: {val_n:,} tokens ({val_docs:,} docs)")
                val_parts, val_done = [], True
        else:
            writer.add(arr)
        now = time.time()
        if now - last >= log_every:
            last = now
            log(f"  file {file_idx:03d}: {docs:,} docs, {fmt_tokens(tokens)} tokens, "
                f"{tokens / (now - t0):,.0f} tok/s")
    if not val_done and val_parts:
        # the whole file was smaller than val_tokens: it all went to val
        write_tokens(os.path.join(out_dir, "val_00000.bin"), np.concatenate(val_parts))
        log(f"  WARNING: file {file_idx} only had {val_n:,} tokens, all used for val")
    writer.close()
    dt = time.time() - t0
    return {
        "file_index": file_idx,
        "docs": docs,
        "tokens": tokens,
        "val_tokens": val_n if carve_val else 0,
        "train_tokens": writer.total_written,
        "dropped_tokens": writer.dropped_tokens,
        "shards": [[os.path.basename(p), n] for p, n in writer.written],
        "seconds": round(dt, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    ap.add_argument("--out", required=True, help="output directory for .bin shards")
    ap.add_argument("--file-start", type=int, default=0, help="first parquet file index (inclusive)")
    ap.add_argument("--file-end", type=int, default=None, help="last parquet file index (exclusive); default: all")
    ap.add_argument("--download-dir", default="/tmp/fineweb_dl",
                    help="where parquet files are downloaded (deleted after use; not persisted)")
    ap.add_argument("--revision", default=None, help="dataset git revision (default: main)")
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--min-shard-tokens", type=int, default=10_000_000,
                    help="a file's final shard is never smaller than this (the tail is merged)")
    ap.add_argument("--val-tokens", type=int, default=10_000_000,
                    help="validation tokens carved from the start of file 0")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--docs-per-task", type=int, default=256)
    ap.add_argument("--stop-after-hours", type=float, default=11.0,
                    help="don't start another file if it would likely end after this many hours")
    ap.add_argument("--force", action="store_true", help="redo files that already have a done marker")
    ap.add_argument("--keep-download", action="store_true", help="don't delete parquet files after use")
    ap.add_argument("--tokenizer-class", default=DEFAULT_TOKENIZER_CLASS, help=argparse.SUPPRESS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = hf_list_files(REPO_ID, FILE_PREFIX, ".parquet", args.revision)
    end = len(files) if args.file_end is None else min(args.file_end, len(files))
    print(f"{REPO_ID}: {len(files)} parquet files under {FILE_PREFIX}; this run: [{args.file_start}, {end})")
    if args.file_start >= end:
        raise SystemExit("empty file range")

    t_run = time.time()
    longest = 0.0
    done_now, skipped, not_started = [], [], []
    for idx in range(args.file_start, end):
        fname = files[idx]
        if os.path.exists(done_marker(args.out, idx)) and not args.force:
            print(f"[{idx:03d}] {fname}: already done, skipping")
            skipped.append(idx)
            continue
        elapsed = time.time() - t_run
        if elapsed + 1.2 * longest > args.stop_after_hours * 3600:
            print(f"[{idx:03d}] stopping: {elapsed / 3600:.2f}h elapsed, last file took "
                  f"{longest / 3600:.2f}h, budget {args.stop_after_hours}h")
            not_started = list(range(idx, end))
            break
        t_file = time.time()
        print(f"[{idx:03d}] downloading {fname} ...", flush=True)
        local = hf_download(REPO_ID, fname, args.download_dir, args.revision)
        print(f"[{idx:03d}] downloaded in {time.time() - t_file:.0f}s, tokenizing with {args.workers} workers",
              flush=True)
        texts = (row["text"] for batch in parquet_column_batches(local, ["text"]) for row in batch)
        stats = process_file(texts, idx, args.out, args.tokenizer, carve_val=(idx == 0),
                             val_tokens=args.val_tokens, shard_tokens=args.shard_tokens,
                             min_shard_tokens=args.min_shard_tokens, workers=args.workers,
                             docs_per_task=args.docs_per_task, class_spec=args.tokenizer_class)
        stats["source_file"] = fname
        stats["revision"] = args.revision or "main"
        with open(done_marker(args.out, idx) + ".tmp", "w") as f:
            json.dump(stats, f, indent=1)
        os.replace(done_marker(args.out, idx) + ".tmp", done_marker(args.out, idx))
        if not args.keep_download:
            os.remove(local)
        took = time.time() - t_file
        longest = max(longest, took)
        done_now.append(idx)
        print(f"[{idx:03d}] done: {stats['docs']:,} docs, {fmt_tokens(stats['tokens'])} tokens, "
              f"{len(stats['shards'])} shards, {took / 60:.1f} min "
              f"({stats['tokens'] / max(stats['seconds'], 1e-9):,.0f} tok/s)", flush=True)

    # summary over everything in the output dir (this and earlier runs)
    all_stats = []
    for p in sorted(glob.glob(os.path.join(args.out, "_done_*.json"))):
        with open(p) as f:
            all_stats.append(json.load(f))
    tot_train = sum(s["train_tokens"] for s in all_stats)
    tot_val = sum(s["val_tokens"] for s in all_stats)
    n_shards = sum(len(s["shards"]) for s in all_stats)
    print("\n==== summary ====")
    print(f"files done this run : {done_now}")
    print(f"files skipped (done): {skipped}")
    if not_started:
        print(f"files NOT started   : {not_started}  (rerun with --file-start {not_started[0]})")
    print(f"output dir {args.out}: {len(all_stats)} files done in total, {n_shards} train shards, "
          f"{tot_train:,} train tokens ({tot_train * 2 / 1e9:.2f} GB), {tot_val:,} val tokens")
    print(f"wall time this run  : {(time.time() - t_run) / 3600:.2f}h")


if __name__ == "__main__":
    main()
