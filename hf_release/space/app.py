"""
Chat demo for gpt2-124m-from-scratch: a 124M model with its tokenizer and training code
written from scratch. Runs on the free CPU tier.

Weights come from the model repo (MODEL_REPO), or from a local folder when LOCAL_WEIGHTS is set.
"""

import json
import os
import threading

import gradio as gr
import torch
from safetensors.torch import load_file

from model.gpt import GPT, GPTConfig
from tokenizer.bpe import BPETokenizer

MODEL_REPO = os.environ.get("MODEL_REPO", "V1jey04/gpt2-124m-from-scratch")
CODE_URL = os.environ.get("CODE_URL", "https://github.com/V1jey04/gpt2-124m-from-scratch")
MAX_CONTEXT_REPLY = 1024

torch.set_num_threads(max(1, os.cpu_count() or 1))


def weights_dir():
    local = os.environ.get("LOCAL_WEIGHTS")
    if local:
        return local
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_REPO, allow_patterns=["model.safetensors", "config.json", "tokenizer.json"])


def load():
    d = weights_dir()
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    model = GPT(GPTConfig(**{k: cfg[k] for k in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")}))
    state = load_file(os.path.join(d, "model.safetensors"))
    state["lm_head.weight"] = state["wte.weight"]
    model.load_state_dict(state)
    model.eval()
    return model, BPETokenizer.load(os.path.join(d, "tokenizer.json"))


MODEL, TOK = load()
SPECIALS = tuple(TOK.special.values())
LOCK = threading.Lock()   # one generation at a time keeps the shared CPU responsive


def text_of(content):
    """Gradio may hand message content as a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content)


def respond(message, history, temperature, max_new_tokens):
    msgs = [{"role": m["role"], "content": text_of(m["content"])} for m in history if m.get("role") in ("user", "assistant")]
    msgs.append({"role": "user", "content": text_of(message)})
    block = MODEL.cfg.block_size
    max_new_tokens = int(max_new_tokens)
    # drop the oldest exchanges until the prompt leaves room for a reply
    while True:
        ids = TOK.encode_chat(msgs, add_generation_prompt=True)
        if len(ids) + max_new_tokens <= block or len(msgs) <= 1:
            break
        msgs = msgs[2:]
    if len(ids) + max_new_tokens > block:
        ids = [TOK.bos_id] + ids[-(block - max_new_tokens - 1):]

    out, shown = [], ""
    with LOCK:
        for t in MODEL.generate(ids, max_new_tokens=max_new_tokens, temperature=float(temperature),
                                top_k=50, top_p=0.95, repetition_penalty=1.1, stop_ids=SPECIALS):
            out.append(t)
            text = TOK.decode(out, skip_special=True)
            if text.endswith("�"):
                continue   # half of a multi-byte character; wait for the rest
            shown = text
            yield shown
    if not shown:
        yield "…"


DESCRIPTION = f"""
A **124M-parameter** model. Its tokenizer, transformer, training loop and this chat code were all
written from scratch, then trained on Kaggle's free T4 GPUs: 2.42B tokens of FineWeb-Edu, then
276M tokens of conversations. It runs here on a free CPU.

It is small, so it writes fluent English but often gets facts wrong.
[Code]({CODE_URL}) · [Model card](https://huggingface.co/{MODEL_REPO})
"""

demo = gr.ChatInterface(
    respond,
    title="GPT-2-size chat model, built from scratch",
    description=DESCRIPTION,
    examples=[["What is the capital of France?", 0.7, 200],
              ["Explain photosynthesis in two sentences.", 0.7, 200],
              ["Give me three tips for studying.", 0.7, 200],
              ["Write a short poem about the sea.", 0.8, 200]],
    additional_inputs=[
        gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(16, 512, value=200, step=8, label="Max new tokens"),
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
