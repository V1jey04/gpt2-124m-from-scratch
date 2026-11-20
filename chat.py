"""
Talk to the model.

Chat with the fine-tuned model:
    python chat.py --ckpt chat/final.pt --tokenizer tokenizer.json

Plain text completion (useful for checking the pretrained model before fine-tuning):
    python chat.py --ckpt run/ckpt.pt --tokenizer tokenizer.json --complete "The capital of France is"
"""

import argparse

import torch

from model.gpt import GPT, GPTConfig
from tokenizer.bpe import BPETokenizer


def load_model(path, device):
    if path.endswith(".safetensors"):
        # the Hugging Face release: model.safetensors with config.json beside it
        import json, os
        from safetensors.torch import load_file
        with open(os.path.join(os.path.dirname(path) or ".", "config.json")) as f:
            cfg = json.load(f)
        model = GPT(GPTConfig(**{k: cfg[k] for k in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")}))
        state = load_file(path)
        state["lm_head.weight"] = state["wte.weight"]   # tied embeddings are stored once
        model.load_state_dict(state)
        return model.to(device).eval()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def stream(model, tok, ids, args, stop_ids):
    """Print tokens as they are generated; returns the generated ids."""
    out, shown = [], ""
    for t in model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                            top_k=args.top_k, top_p=args.top_p,
                            repetition_penalty=args.repetition_penalty, stop_ids=stop_ids):
        out.append(t)
        text = tok.decode(out, skip_special=True)
        if text.endswith("�"):
            continue   # in the middle of a multi-byte character; wait for the rest
        print(text[len(shown):], end="", flush=True)
        shown = text
    print()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--complete", default=None, help="complete this text instead of chatting")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = BPETokenizer.load(args.tokenizer)
    model = load_model(args.ckpt, device)
    block = model.cfg.block_size
    specials = tuple(tok.special.values())

    if args.complete is not None:
        ids = [tok.bos_id] + tok.encode(args.complete)
        print(args.complete, end="", flush=True)
        stream(model, tok, ids, args, stop_ids=(tok.bos_id,))
        return

    print("Type a message. /reset clears the conversation, /quit exits.")
    messages = []
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            messages = []
            continue
        messages.append({"role": "user", "content": user})

        # drop the oldest exchanges until the prompt leaves room for a reply
        while True:
            ids = tok.encode_chat(messages, add_generation_prompt=True)
            if len(ids) + args.max_new_tokens <= block or len(messages) <= 1:
                break
            messages = messages[2:]
        if len(ids) + args.max_new_tokens > block:
            ids = [tok.bos_id] + ids[-(block - args.max_new_tokens - 1):]

        print("bot> ", end="", flush=True)
        out = stream(model, tok, ids, args, stop_ids=specials)
        messages.append({"role": "assistant", "content": tok.decode(out, skip_special=True)})


if __name__ == "__main__":
    main()
