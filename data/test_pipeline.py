"""Offline tests for the data pipeline (no network).

Run from the project root:  python data/test_pipeline.py
Uses a byte-level FakeTokenizer implementing the BPETokenizer interface, and the
real tokenizer.bpe.BPETokenizer (trained on a tiny synthetic corpus) if present.
The Hub layer is replaced with local parquet/jsonl files written by the tests.
"""
import glob
import json
import os
import random
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from data import common  # noqa: E402
from data import prepare_chat, prepare_pretrain, make_tokenizer_sample  # noqa: E402
from data.common import (ShardWriter, normalize_conversation, oasst_best_paths,  # noqa: E402
                         read_tokens, dolly_to_messages)

FAKE = "data.test_pipeline:FakeTokenizer"
BOS, USER, ASSIST, END = 256, 257, 258, 259


class FakeTokenizer:
    """Bytes are ids 0..255, specials 256..259."""
    special = {"<|bos|>": BOS, "<|user|>": USER, "<|assistant|>": ASSIST, "<|end|>": END}
    vocab_size = 260

    @classmethod
    def load(cls, path):
        return cls()

    def encode(self, text):
        return list(text.encode("utf-8"))

    def encode_chat(self, messages, add_generation_prompt=False):
        ids = [BOS]
        for m in messages:
            ids.append(USER if m["role"] == "user" else ASSIST)
            ids.extend(self.encode(m["content"]))
            ids.append(END)
        if add_generation_prompt:
            ids.append(ASSIST)
        return ids

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8")


def rand_docs(n, seed, min_len=20, max_len=400):
    rng = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta", "über", "naïve", "数据", "x", "42", "the", "of"]
    docs = []
    for _ in range(n):
        k = rng.randint(min_len // 5, max_len // 5)
        docs.append(" ".join(rng.choice(words) for _ in range(k)))
    return docs


def expected_stream(docs):
    out = []
    for d in docs:
        d = d.strip() if isinstance(d, str) else ""
        if d:
            out.append(BOS)
            out.extend(d.encode("utf-8"))
    return np.array(out, dtype=np.uint16)


def quiet(*a, **k):
    pass


TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ---------------------------------------------------------------- shard writer

@test
def shard_writer_sizes_and_roundtrip():
    tmp = tempfile.mkdtemp()
    try:
        for total in [0, 50, 99, 100, 999, 1000, 1099, 1100, 1101, 2500, 10_345]:
            d = os.path.join(tmp, str(total))
            w = ShardWriter(d, lambda k: f"train_{k:05d}.bin", shard_tokens=1000,
                            min_shard_tokens=100, floor_tokens=60, log=quiet)
            data = np.arange(total, dtype=np.uint16)
            rng = random.Random(total)
            i = 0
            while i < total:  # feed in random-sized pieces
                j = min(total, i + rng.randint(1, 700))
                w.add(data[i:j])
                i = j
            written = w.close()
            files = sorted(glob.glob(os.path.join(d, "train_*.bin")))
            assert [p for p, _ in written] == files
            got = np.concatenate([read_tokens(f) for f in files]) if files else np.zeros(0, np.uint16)
            sizes = [read_tokens(f).size for f in files]
            if total < 60:
                assert files == [] and w.dropped_tokens == total, (total, sizes)
                continue
            assert np.array_equal(got, data), total
            # all shards but the last are exactly shard_tokens; the last is in [min, shard+min)
            assert all(s == 1000 for s in sizes[:-1]), (total, sizes)
            if len(sizes) > 1:
                assert 100 <= sizes[-1] < 1100, (total, sizes)
            assert not glob.glob(os.path.join(d, "*.tmp"))
        # file format: raw little-endian uint16, no header
        p = os.path.join(tmp, "fmt.bin")
        common.write_tokens(p, np.array([1, 258, 65535], dtype=np.uint16))
        assert open(p, "rb").read() == b"\x01\x00\x02\x01\xff\xff"
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------- pretrain core

def run_file(docs, idx, out, carve_val, workers=2, cls=FAKE, tok_path="unused", **kw):
    params = dict(val_tokens=3000, shard_tokens=5000, min_shard_tokens=1000, floor_tokens=200,
                  workers=workers, docs_per_task=7, class_spec=cls, log=quiet)
    params.update(kw)
    return prepare_pretrain.process_file(iter(docs), idx, out, tok_path, carve_val=carve_val, **params)


@test
def pretrain_single_file_val_and_bos():
    tmp = tempfile.mkdtemp()
    try:
        docs = rand_docs(300, 1) + ["", "   \n ", None] + rand_docs(5, 2)
        st = run_file(docs, 0, tmp, carve_val=True)
        exp = expected_stream(docs)
        val = read_tokens(os.path.join(tmp, "val_00000.bin"))
        shards = sorted(glob.glob(os.path.join(tmp, "train_*.bin")))
        assert [os.path.basename(s) for s in shards][:2] == ["train_00000.bin", "train_00001.bin"]
        train = np.concatenate([read_tokens(s) for s in shards])
        assert np.array_equal(np.concatenate([val, train]), exp)
        assert st["docs"] == 305 and st["tokens"] == exp.size
        assert st["val_tokens"] == val.size >= 3000
        assert st["train_tokens"] == train.size
        # val and every shard boundary that follows val start with <|bos|>
        assert val[0] == BOS and train[0] == BOS
        assert (exp == BOS).sum() == 305
        # bos is followed by document text, never by another bos (no empty docs)
        pos = np.nonzero(exp == BOS)[0]
        assert np.all(exp[np.minimum(pos + 1, exp.size - 1)] != BOS)
        sizes = [read_tokens(s).size for s in shards]
        assert all(s == 5000 for s in sizes[:-1]) and 1000 <= sizes[-1] < 6000, sizes
    finally:
        shutil.rmtree(tmp)


@test
def pretrain_workers_deterministic():
    a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        docs = rand_docs(400, 3)
        run_file(docs, 4, a, carve_val=False, workers=0)
        run_file(docs, 4, b, carve_val=False, workers=3)
        fa = sorted(os.listdir(a))
        assert fa == sorted(os.listdir(b)) and fa[0] == "train_00400.bin"
        for f in fa:
            assert open(os.path.join(a, f), "rb").read() == open(os.path.join(b, f), "rb").read()
    finally:
        shutil.rmtree(a)
        shutil.rmtree(b)


@test
def pretrain_two_sessions_naming_and_order():
    s1, s2, merged = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        file_docs = [rand_docs(250, 10 + i) for i in range(4)]
        for i in (0, 1):
            run_file(file_docs[i], i, s1, carve_val=(i == 0))
        for i in (2, 3):
            run_file(file_docs[i], i, s2, carve_val=(i == 0))
        n1 = set(os.listdir(s1))
        n2 = set(os.listdir(s2))
        assert not (n1 & n2), n1 & n2
        assert "val_00000.bin" in n1 and "val_00000.bin" not in n2
        for d in (s1, s2):
            for f in os.listdir(d):
                shutil.copy(os.path.join(d, f), merged)
        trains = sorted(glob.glob(os.path.join(merged, "train_*.bin")))
        got = np.concatenate([read_tokens(f) for f in trains])
        val = read_tokens(os.path.join(merged, "val_00000.bin"))
        exp = np.concatenate([expected_stream(d) for d in file_docs])
        assert np.array_equal(np.concatenate([val, got]), exp)
        names = [os.path.basename(f) for f in trains]
        assert all(len(n) == len("train_00000.bin") for n in names)
        assert [n[6:9] for n in names] == sorted(n[6:9] for n in names)
    finally:
        for d in (s1, s2, merged):
            shutil.rmtree(d)


@test
def pretrain_tiny_tail_and_tiny_file():
    tmp = tempfile.mkdtemp()
    try:
        # just over one shard: the tail must be merged, so there is exactly one shard
        docs = rand_docs(40, 5, 200, 300)
        exp = expected_stream(docs)
        n = exp.size
        st = run_file(docs, 7, tmp, carve_val=False, shard_tokens=n - 50, min_shard_tokens=500)
        assert [s[0] for s in st["shards"]] == ["train_00700.bin"] and st["shards"][0][1] == n
        # file smaller than floor: dropped with no tiny shard
        st = run_file(rand_docs(1, 6, 20, 40), 8, tmp, carve_val=False)
        assert st["shards"] == [] and st["dropped_tokens"] > 0
        assert not glob.glob(os.path.join(tmp, "train_008*.bin"))
    finally:
        shutil.rmtree(tmp)


@test
def pretrain_rerun_cleans_partial_outputs():
    tmp = tempfile.mkdtemp()
    try:
        for junk in ("train_00105.bin", "train_00199.bin.tmp"):
            open(os.path.join(tmp, junk), "wb").write(b"\x00\x00")
        open(os.path.join(tmp, "train_00200.bin"), "wb").write(b"keep")
        st = run_file(rand_docs(30, 7), 1, tmp, carve_val=False)
        names = sorted(os.listdir(tmp))
        assert "train_00199.bin.tmp" not in names and "train_00200.bin" in names
        assert set(n for n in names if n.startswith("train_001")) == {s[0] for s in st["shards"]}
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------- pretrain main() with fake hub

def write_parquet(path, columns):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table(columns), path)


class FakeHub:
    def __init__(self, root):
        self.root = root
        self.downloads = []

    def list_files(self, repo_id, prefix="", suffix="", revision=None):
        out = []
        for dp, _, fs in os.walk(os.path.join(self.root, repo_id)):
            for f in fs:
                rel = os.path.relpath(os.path.join(dp, f), os.path.join(self.root, repo_id))
                if rel.startswith(prefix) and rel.endswith(suffix):
                    out.append(rel)
        return sorted(out)

    def download(self, repo_id, filename, local_dir, revision=None):
        self.downloads.append(filename)
        dst = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(self.root, repo_id, filename), dst)
        return dst


def patch_hub(module, hub):
    module.hf_list_files = hub.list_files
    module.hf_download = hub.download


def run_main(module, argv):
    old = sys.argv
    sys.argv = [module.__file__] + argv
    try:
        module.main()
    finally:
        sys.argv = old


@test
def pretrain_main_resume_with_fake_hub():
    tmp = tempfile.mkdtemp()
    try:
        repo = os.path.join(tmp, "hub", "HuggingFaceFW/fineweb-edu", "sample", "10BT")
        os.makedirs(repo)
        os.makedirs(os.path.join(tmp, "hub", "HuggingFaceFW/fineweb-edu", "sample", "100BT"))
        file_docs = []
        for i in range(3):
            docs = rand_docs(200, 100 + i)
            file_docs.append(docs)
            write_parquet(os.path.join(repo, f"{i:03d}_00000.parquet"),
                          {"text": docs, "id": [str(j) for j in range(len(docs))]})
        write_parquet(os.path.join(tmp, "hub", "HuggingFaceFW/fineweb-edu", "sample", "100BT",
                                   "000_00000.parquet"), {"text": ["WRONG"] * 3})
        hub = FakeHub(os.path.join(tmp, "hub"))
        patch_hub(prepare_pretrain, hub)
        out, dl = os.path.join(tmp, "out"), os.path.join(tmp, "dl")
        common_args = ["--tokenizer", "x", "--out", out, "--download-dir", dl, "--tokenizer-class", FAKE,
                       "--shard-tokens", "6000", "--min-shard-tokens", "1000", "--val-tokens", "2000",
                       "--workers", "2"]
        real_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            run_main(prepare_pretrain, common_args + ["--file-start", "0", "--file-end", "2"])
            # simulate a crash in file 2: leftover partial shard
            open(os.path.join(out, "train_00203.bin"), "wb").write(b"\x01\x00")
            run_main(prepare_pretrain, common_args + ["--file-start", "0"])
        finally:
            sys.stdout.close()
            sys.stdout = real_stdout
        assert hub.downloads == ["sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet",
                                 "sample/10BT/002_00000.parquet"], hub.downloads  # 0,1 skipped on rerun
        assert not glob.glob(os.path.join(dl, "**", "*.parquet"), recursive=True)  # deleted after use
        assert sorted(glob.glob(os.path.join(out, "_done_*.json"))) == \
            [os.path.join(out, f"_done_{i:03d}.json") for i in range(3)]
        trains = sorted(glob.glob(os.path.join(out, "train_*.bin")))
        got = np.concatenate([read_tokens(os.path.join(out, "val_00000.bin"))] +
                             [read_tokens(f) for f in trains])
        assert np.array_equal(got, np.concatenate([expected_stream(d) for d in file_docs]))
        meta = json.load(open(os.path.join(out, "_done_002.json")))
        assert meta["source_file"] == "sample/10BT/002_00000.parquet"
        assert sum(n for _, n in meta["shards"]) == meta["train_tokens"]
    finally:
        shutil.rmtree(tmp)


@test
def tokenizer_sample_writer():
    import io
    f = io.BytesIO()
    n, nb = make_tokenizer_sample.write_sample(["  a b \n", "", None, "c\n\nd", "e" * 100], f, 20)
    assert (n, nb) == (2, len(b"a b\n\nc\n\nd\n\n")) and f.getvalue() == b"a b\n\nc\n\nd\n\n"


# ---------------------------------------------------------------- chat

@test
def chat_normalization_edge_cases():
    U = lambda c: {"role": "user", "content": c}  # noqa: E731
    A = lambda c: {"role": "assistant", "content": c}  # noqa: E731
    S = lambda c: {"role": "system", "content": c}  # noqa: E731
    ok = lambda m: normalize_conversation(m)[0]  # noqa: E731
    why = lambda m: normalize_conversation(m)[1]  # noqa: E731
    assert ok([U(" hi "), A("hello\n")]) == [U("hi"), A("hello")]
    assert ok([S("Be brief."), U("hi"), A("yo")]) == [U("Be brief.\n\nhi"), A("yo")]
    assert ok([S(""), U("hi"), A("yo")]) == [U("hi"), A("yo")]
    assert ok([S("a"), S("b"), U("hi"), A("yo")])[0]["content"] == "a\n\nb\n\nhi"
    assert why([]) == "empty" and why(None) == "empty"
    assert why([S("only system")]) == "empty"
    assert why([A("hi"), U("yo")]) == "not_user_first"
    assert why([U("a"), U("b"), A("c")]) == "not_alternating"
    assert why([U("a"), A("b"), A("c")]) == "not_alternating"
    assert why([U("a"), A("b"), S("x"), U("c"), A("d")]) == "system_not_leading"
    assert why([U("a"), A("  ")]) == "empty_content"
    assert why([U("a")]) == "no_assistant_reply"
    assert ok([U("a"), A("b"), U("dangling")]) == [U("a"), A("b")]
    assert why([{"role": "tool", "content": "x"}, A("b")]) == "bad_message"
    assert why([{"role": "user", "content": None}, A("b")]) == "bad_message"
    assert ok([{"role": "human", "content": "a"}, {"role": "gpt", "content": "b"}]) == [U("a"), A("b")]
    # dolly
    assert dolly_to_messages({"instruction": "Q?", "context": "", "response": "R"}) == \
        [U("Q?"), A("R")]
    assert dolly_to_messages({"instruction": "Q?", "context": " ctx ", "response": "R"})[0] == U("Q?\n\nctx")


@test
def oasst_best_path():
    def m(mid, parent, role, text, rank=None, lang="en", deleted=False, review=True, t="2023-01-01"):
        return dict(message_id=mid, parent_id=parent, role=role, text=text, rank=rank, lang=lang,
                    deleted=deleted, review_result=review, created_date=t, message_tree_id="t")
    rows = [
        m("r", None, "prompter", "root q"),
        m("a1", "r", "assistant", "worse", rank=1),
        m("a0", "r", "assistant", "best", rank=0),
        m("a2", "r", "assistant", "deleted best", rank=None, deleted=True),
        m("p1", "a0", "prompter", "follow up"),
        m("b0", "p1", "assistant", "second best", rank=0, review=False),  # failed review -> removed
        m("b1", "p1", "assistant", "second ok", rank=1),
        m("p2", "b1", "prompter", "dangling question"),
        m("x", "a1", "prompter", "under worse branch"),
        # non-English tree is skipped
        m("r2", None, "prompter", "hola", lang="es"),
        m("c0", "r2", "assistant", "hola!", rank=0, lang="es"),
        # tree with only a prompt -> nothing
        m("r3", None, "prompter", "unanswered"),
        # unranked siblings: earliest created wins
        m("r4", None, "prompter", "q4"),
        m("d1", "r4", "assistant", "late", t="2023-02-01"),
        m("d0", "r4", "assistant", "early", t="2023-01-15"),
    ]
    random.Random(0).shuffle(rows)
    convs = oasst_best_paths(rows)
    assert convs == [
        [{"role": "prompter", "content": "root q"}, {"role": "assistant", "content": "best"},
         {"role": "prompter", "content": "follow up"}, {"role": "assistant", "content": "second ok"}],
        [{"role": "prompter", "content": "q4"}, {"role": "assistant", "content": "early"}],
    ], convs
    norm = [normalize_conversation(c)[0] for c in convs]
    assert norm[0][0] == {"role": "user", "content": "root q"} and len(norm[0]) == 4


def make_convs(n, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        turns = rng.randint(1, 3)
        msgs = []
        if rng.random() < 0.2:
            msgs.append({"role": "system", "content": f"sys{i}"})
        for t in range(turns):
            msgs.append({"role": "user", "content": f"q{i}.{t} " + "w" * rng.randint(1, 60)})
            msgs.append({"role": "assistant", "content": f"a{i}.{t} " + "v" * rng.randint(1, 120)})
        out.append(msgs)
    return out


@test
def chat_build_split_filter_shuffle():
    tmp = tempfile.mkdtemp()
    try:
        sources = {"smoltalk": make_convs(800, 1) + [[], [{"role": "assistant", "content": "x"}]],
                   "dolly": [dolly_to_messages({"instruction": f"i{k}", "context": "c" if k % 2 else "",
                                                "response": f"r{k}"}) for k in range(100)],
                   "oasst1": [[{"role": "prompter", "content": "p"}, {"role": "assistant", "content": "z" * 500}]]}
        kw = dict(max_tokens=300, val_frac=0.01, seed=7, max_examples={"smoltalk": 600},
                  shard_tokens=20_000, min_shard_tokens=5_000, floor_tokens=100,
                  class_spec=FAKE, log=quiet)
        st = prepare_chat.build_chat(sources, os.path.join(tmp, "a"), "x", workers=2, **kw)
        st2 = prepare_chat.build_chat(sources, os.path.join(tmp, "b"), "x", workers=0, **kw)
        for f in os.listdir(os.path.join(tmp, "a")):
            assert open(os.path.join(tmp, "a", f), "rb").read() == open(os.path.join(tmp, "b", f), "rb").read(), f
        d = st["datasets"]
        assert d["smoltalk"]["dropped"] == {"empty": 1, "not_user_first": 1}
        assert d["smoltalk"]["after_cap"] == 600 and d["dolly"]["after_cap"] == 100
        assert d["oasst1"]["too_long"] == 1 and d["oasst1"]["kept"] == 0
        kept = sum(x["kept"] for x in d.values())
        assert st["val_conversations"] == int(np.ceil(0.01 * kept))
        out = os.path.join(tmp, "a")
        val = read_tokens(os.path.join(out, "val_00000.bin"))
        trains = sorted(glob.glob(os.path.join(out, "train_*.bin")))
        assert [os.path.basename(t) for t in trains][0] == "train_00000.bin" and len(trains) >= 2
        allt = np.concatenate([val] + [read_tokens(t) for t in trains])
        assert allt.size == st["val_tokens"] + st["train_tokens"]
        # split the stream into conversations at <|bos|> and check structure
        starts = np.nonzero(allt == BOS)[0]
        assert starts[0] == 0 and len(starts) == kept
        assert (val == BOS).sum() == st["val_conversations"]
        convs = np.split(allt, starts[1:])
        dolly_seen = []
        for c in convs:
            assert c.size <= 300 and c[1] == USER and c[-1] == END
            roles = [int(x) for x in c if x in (USER, ASSIST)]
            assert roles == [USER, ASSIST] * (len(roles) // 2) and roles
            assert (c == END).sum() == len(roles)
            text = bytes(int(x) for x in c[2:] if x < 256).decode()
            if text.startswith("i"):
                dolly_seen.append(int(text.split("r")[-1]))
        # shuffled: dolly conversations are not in their original order
        assert len(dolly_seen) == 100 and dolly_seen != sorted(dolly_seen)
        meta = json.load(open(os.path.join(out, "meta.json")))
        assert meta["train_tokens"] == st["train_tokens"]
    finally:
        shutil.rmtree(tmp)


@test
def chat_main_with_fake_hub():
    tmp = tempfile.mkdtemp()
    try:
        root = os.path.join(tmp, "hub")
        st_dir = os.path.join(root, "HuggingFaceTB/smoltalk", "data", "all")
        os.makedirs(st_dir)
        convs = make_convs(300, 3)
        write_parquet(os.path.join(st_dir, "train-00000-of-00002.parquet"),
                      {"messages": convs[:150], "source": ["s"] * 150})
        write_parquet(os.path.join(st_dir, "train-00001-of-00002.parquet"),
                      {"messages": convs[150:], "source": ["s"] * 150})
        write_parquet(os.path.join(st_dir, "test-00000-of-00001.parquet"),
                      {"messages": [[{"role": "user", "content": "TESTSPLIT"},
                                     {"role": "assistant", "content": "no"}]], "source": ["s"]})
        dd = os.path.join(root, "databricks/databricks-dolly-15k")
        os.makedirs(dd)
        with open(os.path.join(dd, "databricks-dolly-15k.jsonl"), "w") as f:
            for k in range(50):
                f.write(json.dumps({"instruction": f"inst{k}", "context": "" if k % 3 else f"ctx{k}",
                                    "response": f"resp{k}", "category": "open_qa"}) + "\n")
        od = os.path.join(root, "OpenAssistant/oasst1", "data")
        os.makedirs(od)
        cols = {k: [] for k in prepare_chat.OASST_COLUMNS + ["user_id"]}
        for t in range(20):
            for mid, parent, role, rank in [(f"r{t}", None, "prompter", None),
                                            (f"a{t}", f"r{t}", "assistant", 0),
                                            (f"b{t}", f"r{t}", "assistant", 1)]:
                for k, v in dict(message_id=mid, parent_id=parent, role=role, text=f"{mid} text",
                                 lang="en" if t % 4 else "de", rank=rank, deleted=False,
                                 review_result=True, created_date="2023-01-01",
                                 message_tree_id=f"r{t}", user_id="u").items():
                    cols[k].append(v)
        write_parquet(os.path.join(od, "train-00000-of-00001-abc.parquet"), cols)
        hub = FakeHub(root)
        patch_hub(prepare_chat, hub)
        out = os.path.join(tmp, "out")
        real_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            run_main(prepare_chat, ["--tokenizer", "x", "--out", out, "--download-dir", os.path.join(tmp, "dl"),
                                    "--tokenizer-class", FAKE, "--workers", "2", "--shard-tokens", "5000",
                                    "--min-shard-tokens", "1000", "--max-examples", "dolly=40"])
        finally:
            sys.stdout.close()
            sys.stdout = real_stdout
        meta = json.load(open(os.path.join(out, "meta.json")))
        d = meta["datasets"]
        assert d["smoltalk"]["raw"] == 300 and d["dolly"]["after_cap"] == 40
        assert d["oasst1"]["raw"] == 15  # 20 trees, 5 German skipped
        allt = np.concatenate([read_tokens(f) for f in sorted(glob.glob(os.path.join(out, "*.bin")))])
        text = bytes(int(x) for x in allt if x < 256).decode()
        assert "TESTSPLIT" not in text and "a1 text" in text and "b1 text" not in text
        assert "ctx" in text
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------- real tokenizer

@test
def real_tokenizer_end_to_end():
    try:
        from tokenizer.bpe import BPETokenizer
    except Exception as e:  # noqa: BLE001
        print(f"    (skipped: tokenizer.bpe not importable: {e})")
        return
    tmp = tempfile.mkdtemp()
    try:
        corpus = os.path.join(tmp, "corpus.txt")
        docs = rand_docs(300, 42)
        with open(corpus, "w", encoding="utf-8") as f:
            make_tokenizer_sample.write_sample(docs, f.buffer, 10**9)
        tok = BPETokenizer.train([corpus], vocab_size=400, min_chunk_freq=1, workers=1, verbose=False)
        tpath = os.path.join(tmp, "tok.json")
        tok.save(tpath)
        cls = "tokenizer.bpe:BPETokenizer"
        out = os.path.join(tmp, "pre")
        docs2 = rand_docs(200, 43) + ["contains <|bos|> literally"]
        st = run_file(docs2, 0, out, carve_val=True, cls=cls, tok_path=tpath, val_tokens=500,
                      shard_tokens=2000, min_shard_tokens=500, floor_tokens=100)
        tok = BPETokenizer.load(tpath)
        bos = tok.special["<|bos|>"]
        allt = np.concatenate([read_tokens(os.path.join(out, "val_00000.bin"))] +
                              [read_tokens(f) for f in sorted(glob.glob(os.path.join(out, "train_*.bin")))])
        assert allt.size == st["tokens"] and allt[0] == bos and int(allt.max()) < tok.vocab_size
        starts = np.nonzero(allt == bos)[0]
        assert len(starts) == len(docs2)  # the literal "<|bos|>" text is not a special token
        decoded = [tok.decode(c[1:].tolist()) for c in np.split(allt, starts[1:])]
        assert decoded == [d.strip() for d in docs2]
        # chat path with the real encode_chat
        sources = {"dolly": [dolly_to_messages({"instruction": f"i{k}", "context": "", "response": "gamma delta"})
                             for k in range(300)]}
        stc = prepare_chat.build_chat(sources, os.path.join(tmp, "chat"), tpath, max_tokens=1024, val_frac=0.01,
                                      seed=1, shard_tokens=1000, min_shard_tokens=200, floor_tokens=50,
                                      workers=2, class_spec=cls, log=quiet)
        allc = np.concatenate([read_tokens(f) for f in sorted(glob.glob(os.path.join(tmp, "chat", "*.bin")))])
        assert allc.size == stc["val_tokens"] + stc["train_tokens"] and allc[0] == bos
        assert (allc == bos).sum() == 300
        seen = set()
        for c in np.split(allc, np.nonzero(allc == bos)[0][1:]):
            s = tok.decode(c.tolist())  # "<|bos|><|user|>iK<|end|><|assistant|>gamma delta<|end|>"
            k = s.split("<|user|>")[1].split("<|end|>")[0]
            exp = tok.encode_chat([{"role": "user", "content": k},
                                   {"role": "assistant", "content": "gamma delta"}])
            assert c.tolist() == exp, s
            seen.add(k)
        assert seen == {f"i{k}" for k in range(300)}
    finally:
        shutil.rmtree(tmp)


def main():
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
