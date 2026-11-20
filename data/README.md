# Data pipeline

Everything here downloads open datasets from the Hugging Face Hub, tokenizes them
with our own `tokenizer/bpe.py` and writes raw `uint16` token files (little-endian,
no header) that `train.py` reads. No `transformers` or `tokenizers` anywhere;
`huggingface_hub`, `pyarrow` and `numpy` are only used to fetch and read files.

Run all commands from the project root. On Kaggle, turn **Internet on** in the
notebook settings. Setting an `HF_TOKEN` secret is optional but helps if the Hub
rate-limits anonymous downloads.

## Files

| file | what it does |
|---|---|
| `make_tokenizer_sample.py` | writes ~1.5 GB of FineWeb-Edu text for tokenizer training |
| `prepare_pretrain.py` | tokenizes FineWeb-Edu `sample-10BT` into pretraining shards, file by file |
| `prepare_chat.py` | builds the chat fine-tune set (SmolTalk, Dolly, OASST1) |
| `common.py` | shared code: worker pool, shard writer, chat cleanup, Hub helpers |
| `test_pipeline.py` | offline tests (`python data/test_pipeline.py`) |

## Steps in order

### 1. Text sample for the tokenizer

```
python data/make_tokenizer_sample.py --out /kaggle/working/tok_sample.txt --max-bytes 1500000000
python -m tokenizer.bpe train --input /kaggle/working/tok_sample.txt --vocab-size 50000 --out /kaggle/working/tokenizer.json
```

The first command downloads one 2.15 GB parquet file and writes documents separated
by blank lines. It should take a few minutes (estimate). Use `--num-files 2` to spread
the sample over two files. Save `tokenizer.json` as a private Kaggle Dataset because
every later step needs it.

### 2. Pretraining tokens (FineWeb-Edu sample-10BT)

The dataset is 14 parquet files (`sample/10BT/000_00000.parquet` to `013_00000.parquet`,
13 x 2.15 GB plus one 541 MB file). The script reads that list from the Hub when it
starts. It downloads one file, tokenizes it, deletes it and moves on to the next.

```
python data/prepare_pretrain.py --tokenizer /kaggle/input/<tokenizer-dataset>/tokenizer.json \
    --out /kaggle/working/pretrain --file-start 0 --file-end 5
```

Output:

- `val_00000.bin`: about the first 10M tokens of file 0. It is only written by the run that does file 0.
- `train_FFFPP.bin`: `FFF` is the parquet file number, `PP` the shard within that file, with about 100M tokens per shard.
  The last shard of a file absorbs the tail, so it can hold up to ~110M tokens.
- `_done_FFF.json`: a marker with stats, written when a file is finished.

Every document is `<|bos|>` followed by its tokens, with documents concatenated.

**Splitting across Kaggle sessions.** One session cannot hold all of it, because
`/kaggle/working` keeps only about 20 GB and 10B tokens is about 20 GB. Use three sessions:

| session | command flags | rough output (estimate) |
|---|---|---|
| 1 | `--file-start 0 --file-end 5` | ~3.7B tokens, ~7.5 GB, includes val |
| 2 | `--file-start 5 --file-end 10` | ~3.7B tokens, ~7.5 GB |
| 3 | `--file-start 10 --file-end 14` | ~2.6B tokens, ~5 GB |

For each session:

1. Create a CPU notebook with Internet on and attach the tokenizer dataset.
2. Run the command with that session's range. Use "Save Version" -> "Save & Run All",
   so the run continues with the browser closed (12 h limit).
3. When it finishes, open the notebook's Output tab and choose "New Dataset" to save
   `/kaggle/working/pretrain` as a private dataset, e.g. `fineweb-edu-tok-part1`.

Shard names depend only on the file number, so the three datasets never share a
file name, and sorting all `train_*.bin` names together gives dataset order. In
`train.py`, attach all three datasets and glob `train_*.bin` across them.

**Reruns.** Files that already have a `_done_FFF.json` in `--out` are skipped. An
interrupted file is redone from scratch, and its partial shards are deleted first.
The script will not start a new file once it looks like that file would end past
`--stop-after-hours` (default 11). It then prints the `--file-start` to use next.
Pass `--revision <commit sha>` to pin the dataset version so reruns are bit-identical.

**Speed (estimate).** The tokenizer is pure Python, so this step is CPU-bound. Our
pipeline hit ~1.4M tokens/s on a 2-core test machine, but on Python source code,
not FineWeb-Edu, with a small 16k vocab. Kaggle's 4 older cores may be faster or
slower than that. As a guess, expect 0.5 to 2 hours of tokenizing per 2.15 GB file.
The first file of a session is the slowest while the per-worker cache warms up.
The progress lines print tokens/s, so check the first few minutes. If it is much
slower, use smaller ranges per session.

### 3. Chat fine-tune set

```
python data/prepare_chat.py --tokenizer /kaggle/input/<tokenizer-dataset>/tokenizer.json \
    --out /kaggle/working/chat
```

- SmolTalk (`HuggingFaceTB/smoltalk`, config `all`, train split only, about 1.04M conversations, 2 GB download)
- Dolly (`databricks/databricks-dolly-15k`, 15k rows). The instruction and the
  `context` (when present) are joined with a blank line.
- OASST1 (`OpenAssistant/oasst1`). English trees only, with one conversation per tree.
  It follows the best-ranked reply (rank 0) at each turn and skips deleted or
  review-failed messages.

Conversations become alternating user/assistant turns. A system message is put at
the start of the first user message, followed by a blank line. A conversation is
dropped if it does not start with the user, does not alternate, has an empty turn,
or is longer than `--max-tokens` (default 1024) once encoded. A last user turn
with no reply is cut off. Everything is shuffled with `--seed` (default 1234), 1%
goes to `val_00000.bin` and the rest goes to `train_00000.bin` and onward. Each
conversation is `tok.encode_chat(messages)`. `meta.json` lists counts, drop reasons
and tokens per dataset.

Useful flags: `--datasets smoltalk,dolly`, `--max-examples smoltalk=300000 oasst1=5000`.
This takes the first download of ~2.2 GB and several GB of RAM for SmolTalk. The total
token count isn't known until it runs (`meta.json` has it), but expect a few hundred
million tokens, i.e. under 1 GB of output (estimate). Save `/kaggle/working/chat` as
another private dataset.

## Tests

```
python data/test_pipeline.py
```

The tests run offline with a fake byte-level tokenizer and fake local parquet/jsonl
files in place of the Hub. They also run the real `tokenizer/bpe.py`, trained on a
tiny synthetic corpus.
