"""
Benchmark our 124M model against OpenAI's GPT-2 124M on the same tasks.

    .venv/bin/python benchmark.py --ckpt final.pt --tokenizer tokenizer.json

Tasks (zero-shot, multiple choice scored by the model's loss on each option):
  hellaswag  sentence completion, 10,042 val questions, 4 options
  arc_easy   grade-school science, 2,376 test questions, 3-5 options
  lambada    predict the last word of a passage, 5,153 passages (exact match, greedy)
  wikitext   bits per byte on WikiText-2 test (lower is better; fair across tokenizers)

Multiple-choice options are compared by loss per byte of the option text, so the
two different tokenizers don't bias the comparison.
"""

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from model.gpt import GPT, GPTConfig
from tokenizer.bpe import BPETokenizer


# ---------------------------------------------------------------- models

class Ours:
    name = "ours (124M, 2.4B tok FineWeb-Edu + chat)"

    def __init__(self, ckpt, tok_path, device):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = GPT(GPTConfig(**ck["config"]))
        self.model.load_state_dict(ck["model"])
        self.model.to(device).eval()
        self.tok = BPETokenizer.load(tok_path)
        self.start = [self.tok.bos_id]
        self.ctx_len = self.model.cfg.block_size
        self.device = device

    def encode(self, s):
        return self.tok.encode(s)

    @torch.no_grad()
    def logits(self, idx):
        return self.model(idx, targets=idx)[0].float()   # passing targets returns all positions


class GPT2:
    name = "OpenAI GPT-2 124M"

    def __init__(self, device):
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.start = [self.tok.eos_token_id]   # <|endoftext|> separates GPT-2's training documents
        self.ctx_len = 1024
        self.device = device

    def encode(self, s):
        return self.tok.encode(s)

    @torch.no_grad()
    def logits(self, idx):
        return self.model(idx).logits.float()


# ---------------------------------------------------------------- scoring

def option_losses(m, context, options):
    """Summed loss of each option's tokens given the context. Returns list of floats."""
    ctx = m.start + m.encode(context)
    seqs, spans = [], []
    for o in options:
        oi = m.encode(o)
        s = (ctx + oi)[-m.ctx_len:]
        seqs.append(s)
        spans.append(len(oi))
    L = max(len(s) for s in seqs)
    idx = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        idx[i, :len(s)] = torch.tensor(s)
    idx = idx.to(m.device)
    logp = F.log_softmax(m.logits(idx), dim=-1)
    out = []
    for i, s in enumerate(seqs):
        n = spans[i]
        tgt = idx[i, len(s) - n:len(s)]
        lp = logp[i, len(s) - n - 1:len(s) - 1].gather(1, tgt[:, None]).squeeze(1)
        out.append(-lp.sum().item())
    return out


def mc_accuracy(m, items, limit):
    """items: (context, options, answer_index). Accuracy by loss-per-byte of each option."""
    correct = total = 0
    for ctx, opts, ans in items[:limit]:
        losses = option_losses(m, ctx, opts)
        per_byte = [l / max(1, len(o.encode("utf-8"))) for l, o in zip(losses, opts)]
        correct += int(min(range(len(opts)), key=per_byte.__getitem__) == ans)
        total += 1
    return correct / total, total


def lambada_accuracy(m, items, limit):
    correct = total = 0
    for text in items[:limit]:
        ctx, word = text.rsplit(" ", 1)
        c = m.start + m.encode(ctx)
        w = m.encode(" " + word)
        s = (c + w)[-m.ctx_len:]
        idx = torch.tensor([s], device=m.device)
        pred = m.logits(idx)[0, len(s) - len(w) - 1:len(s) - 1].argmax(-1).tolist()
        correct += int(pred == w)
        total += 1
    return correct / total, total


def bits_per_byte(m, text, stride=512):
    ids = m.start + m.encode(text)
    nll, scored = 0.0, 0
    prev_end = 1
    for begin in range(0, len(ids) - 1, stride):
        end = min(begin + m.ctx_len, len(ids))
        idx = torch.tensor([ids[begin:end]], device=m.device)
        logp = F.log_softmax(m.logits(idx)[0], dim=-1)
        first = max(prev_end, begin + 1)          # only score tokens not scored by an earlier window
        tgt = idx[0, first - begin:end - begin]
        lp = logp[first - begin - 1:end - begin - 1].gather(1, tgt[:, None])
        nll -= lp.sum().item()
        scored += end - first
        prev_end = end
        if end == len(ids):
            break
    return nll / math.log(2) / len(text.encode("utf-8"))


# ---------------------------------------------------------------- data

def load_tasks():
    from datasets import load_dataset

    hs = [(d["ctx"], [" " + e for e in d["endings"]], int(d["label"]))
          for d in load_dataset("Rowan/hellaswag", split="validation")]

    arc = []
    for d in load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"):
        labels = d["choices"]["label"]
        if d["answerKey"] not in labels:
            continue
        arc.append((f"Question: {d['question']}\nAnswer:",
                    [" " + t for t in d["choices"]["text"]],
                    labels.index(d["answerKey"])))

    lam = [d["text"] for d in load_dataset("EleutherAI/lambada_openai", "default", split="test")]

    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wiki = "".join(wt["text"])
    return {"hellaswag": hs, "arc_easy": arc, "lambada": lam, "wikitext": wiki}


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="final.pt")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--limit", type=int, default=100000, help="max questions per task (for quick runs)")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", default="benchmark_results.json")
    args = p.parse_args()

    tasks = load_tasks()
    results = {}
    for make in (lambda: Ours(args.ckpt, args.tokenizer, args.device), lambda: GPT2(args.device)):
        m = make()
        r = {}
        t0 = time.time()
        r["hellaswag"], n = mc_accuracy(m, tasks["hellaswag"], args.limit); print(m.name, "hellaswag", round(r["hellaswag"], 4), n, flush=True)
        r["arc_easy"], n = mc_accuracy(m, tasks["arc_easy"], args.limit);   print(m.name, "arc_easy", round(r["arc_easy"], 4), n, flush=True)
        r["lambada"], n = lambada_accuracy(m, tasks["lambada"], args.limit); print(m.name, "lambada", round(r["lambada"], 4), n, flush=True)
        r["wikitext_bpb"] = bits_per_byte(m, tasks["wikitext"]);             print(m.name, "wikitext bpb", round(r["wikitext_bpb"], 4), flush=True)
        r["seconds"] = round(time.time() - t0)
        results[m.name] = r
        del m
        if args.device == "mps":
            torch.mps.empty_cache()

    names = list(results)
    print(f"\n{'task':<14}" + "".join(f"{n[:34]:>36}" for n in names))
    for k, label in [("hellaswag", "HellaSwag acc"), ("arc_easy", "ARC-Easy acc"),
                     ("lambada", "LAMBADA acc"), ("wikitext_bpb", "WikiText bpb ↓")]:
        print(f"{label:<14}" + "".join(f"{results[n][k]:>36.4f}" for n in names))
    print("random chance: HellaSwag 0.25, ARC-Easy ~0.25, LAMBADA ~0")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
