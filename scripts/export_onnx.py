"""
Export the chat model to ONNX for the in-browser demo, with a KV cache.

    python scripts/export_onnx.py --weights hf_release/model --out hf_release/model/onnx

One graph does both the prompt and each new token:
  inputs   input_ids (1, T) int64, past_k / past_v (layers, 1, heads, P, head_dim) float32
  outputs  logits (1, vocab) for the last position, present_k / present_v with P + T positions
Start with P = 0. The graph is checked against PyTorch, then quantised to int8 (model_int8.onnx).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt import GPT, GPTConfig  # noqa: E402


class Step(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g
        self.H = g.cfg.n_head
        self.C = g.cfg.n_embd

    def forward(self, input_ids, past_k, past_v):
        g, H, C = self.g, self.H, self.C
        D = C // H
        T = input_ids.shape[1]
        P = past_k.shape[3]
        pos = torch.arange(T, device=input_ids.device) + P
        x = g.wte(input_ids) + g.wpe(pos)
        # query i (absolute position P+i) may see keys 0..P+i
        allowed = torch.arange(P + T)[None, :] <= (torch.arange(T)[:, None] + P)
        neg = torch.zeros(T, P + T).masked_fill(~allowed, float("-inf"))
        ks, vs = [], []
        for i, b in enumerate(g.blocks):
            h = b.ln_1(x)
            q, k, v = b.attn.c_attn(h).split(C, dim=2)
            q = q.view(1, T, H, D).transpose(1, 2)
            k = torch.cat([past_k[i], k.view(1, T, H, D).transpose(1, 2)], dim=2)
            v = torch.cat([past_v[i], v.view(1, T, H, D).transpose(1, 2)], dim=2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / D ** 0.5) + neg
            y = F.softmax(att, dim=-1) @ v
            y = y.transpose(1, 2).reshape(1, T, C)
            x = x + b.attn.c_proj(y)
            x = x + b.mlp(b.ln_2(x))
            ks.append(k)
            vs.append(v)
        x = g.ln_f(x[:, -1, :])
        logits = x @ g.wte.weight.t()
        return logits, torch.stack(ks), torch.stack(vs)


def load(weights):
    cfg = json.load(open(os.path.join(weights, "config.json")))
    g = GPT(GPTConfig(**{k: cfg[k] for k in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")}))
    st = load_file(os.path.join(weights, "model.safetensors"))
    st["lm_head.weight"] = st["wte.weight"]
    g.load_state_dict(st)
    return g.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="hf_release/model")
    ap.add_argument("--out", default="hf_release/model/onnx")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    g = load(args.weights)
    step = Step(g).eval()
    L, H, D = g.cfg.n_layer, g.cfg.n_head, g.cfg.n_embd // g.cfg.n_head
    ids = torch.tensor([[50000, 50001, 1550, 310]])
    pk = torch.zeros(L, 1, H, 3, D)
    pv = torch.zeros(L, 1, H, 3, D)
    fp32 = os.path.join(args.out, "model_fp32.onnx")
    T = torch.export.Dim("T", min=1, max=1024)
    Pd = torch.export.Dim("P", min=0, max=1023)
    with torch.no_grad():
        torch.onnx.export(step, (ids, pk, pv), fp32, dynamo=True, external_data=False,
                          input_names=["input_ids", "past_k", "past_v"],
                          output_names=["logits", "present_k", "present_v"],
                          dynamic_shapes={"input_ids": {1: T}, "past_k": {3: Pd}, "past_v": {3: Pd}})
    print("exported", fp32, f"{os.path.getsize(fp32)/1e6:.0f} MB")

    # check against the PyTorch model: prefill with an empty cache, then two cached steps
    import onnxruntime as ort
    from tokenizer.bpe import BPETokenizer
    tok = BPETokenizer.load(os.path.join(args.weights, "tokenizer.json"))
    prompt = tok.encode_chat([{"role": "user", "content": "What is the capital of France?"}], add_generation_prompt=True)

    def greedy_onnx(path, n=12):
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        k = np.zeros((L, 1, H, 0, D), np.float32)
        v = np.zeros((L, 1, H, 0, D), np.float32)
        feed = np.array([prompt], np.int64)
        out = []
        for _ in range(n):
            logits, k, v = s.run(None, {"input_ids": feed, "past_k": k, "past_v": v})
            nxt = int(logits[0, :50004].argmax())
            out.append(nxt)
            feed = np.array([[nxt]], np.int64)
        return out, logits

    with torch.no_grad():
        ref = list(g.generate(prompt, max_new_tokens=12, temperature=0, repetition_penalty=1.0))
        full = g(torch.tensor([prompt + ref[:-1]]), targets=torch.tensor([prompt + ref[:-1]]))[0][0, -1].numpy()
    got, last_logits = greedy_onnx(fp32)
    print("pytorch greedy:", tok.decode(ref))
    print("onnx    greedy:", tok.decode(got))
    print("max |logit diff| at step 12:", float(np.abs(last_logits[0] - full).max()))

    from onnxruntime.quantization import quantize_dynamic, QuantType
    q8 = os.path.join(args.out, "model_int8.onnx")
    # per-channel int8 keeps greedy output far closer to fp32 than per-tensor int8 on this model
    quantize_dynamic(fp32, q8, weight_type=QuantType.QUInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gather"])
    print("int8", f"{os.path.getsize(q8)/1e6:.0f} MB")
    got8, _ = greedy_onnx(q8)
    print("int8    greedy:", tok.decode(got8))


if __name__ == "__main__":
    main()
