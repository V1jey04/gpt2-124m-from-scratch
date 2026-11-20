"""
Decoder-only transformer in the GPT-2 small shape, written from scratch in plain PyTorch.

12 layers, 12 heads, 768 dims, 1024 context, tied input/output embeddings.
With a 50,304-token vocab that is 124.5M parameters.

Nothing here comes from a model library: the attention, MLP, blocks, loss and
sampling are all defined in this file.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304   # 50,000 BPE tokens + 4 special tokens, padded up to a multiple of 64
    block_size: int = 1024    # maximum context length
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        # one linear layer produces query, key and value for all heads at once
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj.IS_RESIDUAL_PROJ = True
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None):
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)   # (B, heads, T, head_dim)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        causal = True
        if cache is not None:
            # KV cache for generation: keys/values of earlier tokens are reused,
            # so each new token costs one row of attention instead of a full pass.
            if "k" in cache:
                assert T == 1, "with a warm cache, feed one token at a time"
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
                causal = False   # a single new token may attend to everything before it
            cache["k"], cache["v"] = k, v

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.act = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.c_proj.IS_RESIDUAL_PROJ = True
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln_1(x), cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)   # learned position embeddings
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight                 # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "IS_RESIDUAL_PROJ", False):
                # keep the residual stream's variance stable as depth grows
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, loss_mask=None, caches=None, pos_offset=0):
        """
        idx:       (B, T) token ids
        targets:   (B, T) next-token ids; if given, returns the cross-entropy loss
        loss_mask: (B, T) bool; if given, only masked-in positions count toward the loss
                   (used in chat fine-tuning to train on assistant tokens only)
        caches:    list of per-layer dicts for KV caching during generation
        """
        B, T = idx.shape
        assert pos_offset + T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for i, block in enumerate(self.blocks):
            x = block(x, None if caches is None else caches[i])
        x = self.ln_f(x)

        if targets is None:
            return self.lm_head(x[:, -1:, :]), None   # generation only needs the last position

        logits = self.lm_head(x)
        if loss_mask is None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
            m = loss_mask.view(-1).to(ce.dtype)
            loss = (ce * m).sum() / m.sum().clamp(min=1)
        return logits, loss

    def configure_optimizer(self, lr, weight_decay, betas=(0.9, 0.95), fused=False):
        # weight decay on matrices (weights, embeddings), none on biases and LayerNorm
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, fused=fused)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=256, temperature=0.7, top_k=50, top_p=0.95,
                 repetition_penalty=1.1, stop_ids=()):
        """
        Yields new token ids one at a time (so the caller can stream text).
        ids: list of prompt token ids. Sampling stops at any id in stop_ids,
        after max_new_tokens, or when the context window is full.
        """
        self.eval()
        device = self.wte.weight.device
        ids = list(ids)[-(self.cfg.block_size - 1):]
        seen = set(ids)
        caches = [{} for _ in self.blocks]
        logits, _ = self(torch.tensor([ids], dtype=torch.long, device=device), caches=caches)
        cur_len = len(ids)

        for _ in range(max_new_tokens):
            logits = logits[0, -1, :].float()

            if repetition_penalty != 1.0 and seen:
                s = torch.tensor(sorted(seen), device=device)
                vals = logits[s]
                logits[s] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

            if temperature <= 0:
                nxt = int(torch.argmax(logits))
            else:
                logits = logits / temperature
                if top_k and top_k < logits.size(-1):
                    kth = torch.topk(logits, top_k).values[-1]
                    logits[logits < kth] = -float("inf")
                if top_p and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum = torch.cumsum(probs, dim=-1)
                    sorted_logits[cum - probs > top_p] = -float("inf")   # always keeps the top token
                    logits = torch.full_like(logits, -float("inf")).scatter(0, sorted_idx, sorted_logits)
                nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))

            if nxt in stop_ids:
                return
            yield nxt
            seen.add(nxt)
            if cur_len >= self.cfg.block_size:
                return
            logits, _ = self(torch.tensor([[nxt]], dtype=torch.long, device=device),
                             caches=caches, pos_offset=cur_len)
            cur_len += 1


if __name__ == "__main__":
    model = GPT(GPTConfig())
    print(f"parameters: {model.num_params():,}")
