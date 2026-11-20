"""
Turn a training checkpoint into the files published on Hugging Face.

    python scripts/export_hf.py --ckpt final.pt --tokenizer tokenizer.json --out hf_release/model

Writes model.safetensors (weights only, fp32, tied embeddings stored once), config.json and a
copy of tokenizer.json. The optimizer state in the checkpoint (about 1 GB) is left out.
"""

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt import GPT, GPTConfig  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="final.pt")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--out", default="hf_release/model")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = GPT(GPTConfig(**cfg))
    model.load_state_dict(ck["model"])

    state = {k: v.detach().contiguous().clone() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    os.makedirs(args.out, exist_ok=True)
    save_file(state, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})

    config = dict(cfg)
    config.update({
        "architecture": "GPT-2-small shape, written from scratch (model/gpt.py)",
        "tied_embeddings": True,
        "num_parameters": model.num_params(),
        "special_tokens": {"<|bos|>": 50000, "<|user|>": 50001, "<|assistant|>": 50002, "<|end|>": 50003},
        "chat_format": "<|bos|><|user|>{question}<|end|><|assistant|>{answer}<|end|>",
        "train_step": ck.get("step"),
        "train_tokens_seen": ck.get("tokens_seen"),
        "best_val_loss": ck.get("best_val"),
    })
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    shutil.copy(args.tokenizer, os.path.join(args.out, "tokenizer.json"))

    size = os.path.getsize(os.path.join(args.out, "model.safetensors")) / 1e6
    print(f"wrote {args.out}: model.safetensors ({size:.0f} MB), config.json, tokenizer.json")


if __name__ == "__main__":
    main()
