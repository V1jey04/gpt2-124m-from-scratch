"""Write a plain-text sample of FineWeb-Edu (sample-10BT) for tokenizer training.

Documents are stripped and written one after another, each followed by a blank
line. Stops once --max-bytes (UTF-8) would be exceeded. Deterministic: it reads
the first --num-files parquet files in sorted order and takes an equal byte
budget from each.

Example:
  python data/make_tokenizer_sample.py --out /kaggle/working/tok_sample.txt --max-bytes 1500000000
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.common import hf_download, hf_list_files, parquet_column_batches  # noqa: E402

REPO_ID = "HuggingFaceFW/fineweb-edu"
FILE_PREFIX = "sample/10BT/"


def write_sample(texts, f, max_bytes):
    """Core: write stripped non-empty docs separated by a blank line until max_bytes.
    Returns (docs_written, bytes_written)."""
    n = nbytes = 0
    for t in texts:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        b = (t + "\n\n").encode("utf-8")
        if nbytes + len(b) > max_bytes:
            break
        f.write(b)
        n += 1
        nbytes += len(b)
    return n, nbytes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output .txt path")
    ap.add_argument("--max-bytes", type=int, default=1_500_000_000)
    ap.add_argument("--num-files", type=int, default=1,
                    help="spread the sample over the first N parquet files (each ~2.15GB download)")
    ap.add_argument("--download-dir", default="/tmp/fineweb_dl")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--keep-download", action="store_true")
    args = ap.parse_args()

    files = hf_list_files(REPO_ID, FILE_PREFIX, ".parquet", args.revision)[:args.num_files]
    per_file = args.max_bytes // len(files)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    tot_docs = tot_bytes = 0
    with open(args.out, "wb") as f:
        for fname in files:
            print(f"downloading {fname} ...", flush=True)
            local = hf_download(REPO_ID, fname, args.download_dir, args.revision)
            texts = (row["text"] for batch in parquet_column_batches(local, ["text"]) for row in batch)
            n, nb = write_sample(texts, f, per_file)
            tot_docs += n
            tot_bytes += nb
            print(f"  {fname}: {n:,} docs, {nb / 1e9:.3f} GB", flush=True)
            if not args.keep_download:
                os.remove(local)
    print(f"wrote {args.out}: {tot_docs:,} docs, {tot_bytes / 1e9:.3f} GB in {time.time() - t0:.0f}s")
    if tot_bytes < 0.9 * args.max_bytes:
        print("NOTE: sample is smaller than requested; use --num-files to read more files")


if __name__ == "__main__":
    main()
