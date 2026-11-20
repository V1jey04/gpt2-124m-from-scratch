"""
Training script for the 124M model. Built for Kaggle's free T4 x2:

  * both GPUs via DistributedDataParallel   (launch with torchrun)
  * fp16 mixed precision with a GradScaler  (T4s have no fast bf16)
  * gradient accumulation to a large effective batch
  * checkpoints on a timer and a clean exit before the 12-hour session limit,
    so the next session resumes exactly where this one stopped

Pretraining, two T4s:
    torchrun --nproc_per_node=2 train.py --data-dir /kaggle/input/my-shards --out-dir /kaggle/working/run

Resume in the next session (previous output attached as an input dataset):
    torchrun --nproc_per_node=2 train.py --data-dir ... --out-dir /kaggle/working/run \
        --resume /kaggle/input/previous-run/run/ckpt.pt

Chat fine-tune from the pretrained weights (loss on assistant tokens only):
    torchrun --nproc_per_node=2 train.py --data-dir /kaggle/input/chat-shards --out-dir /kaggle/working/chat \
        --init-from /kaggle/input/pretrain-run/run/ckpt.pt --chat-mask --tokenizer tokenizer.json \
        --lr 5e-5 --warmup-steps 100 --max-steps 3000

Single GPU, CPU or Apple Silicon also works: python train.py ...

Data: a directory of train_*.bin and val_*.bin files holding raw uint16 token ids.
"""

import argparse
import csv
import glob
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.gpt import GPT, GPTConfig


def get_args():
    p = argparse.ArgumentParser()
    # data / io
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default="run")
    p.add_argument("--resume", default=None, help="checkpoint to continue from (model, optimizer, step, data position)")
    p.add_argument("--init-from", default=None, help="checkpoint to take model weights from only (for fine-tuning)")
    # model
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-embd", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.0)
    # batch
    p.add_argument("--batch-size", type=int, default=16, help="sequences per GPU per micro-step")
    p.add_argument("--total-batch-tokens", type=int, default=262144,
                   help="tokens per optimizer step, summed over GPUs and accumulation steps")
    # optimizer / schedule
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=700)
    p.add_argument("--max-steps", type=int, default=38000, help="38k steps x 262k tokens is about 10B tokens")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # eval / checkpoint
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=40, help="val batches per GPU")
    p.add_argument("--ckpt-minutes", type=float, default=30.0)
    p.add_argument("--max-hours", type=float, default=11.25,
                   help="save and exit after this long, before Kaggle's 12h limit kills the session")
    p.add_argument("--log-every", type=int, default=10)
    # chat fine-tuning
    p.add_argument("--chat-mask", action="store_true", help="only train on assistant tokens")
    p.add_argument("--tokenizer", default=None, help="tokenizer.json (needed with --chat-mask)")
    # misc
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


class ShardLoader:
    """
    Walks the token shards in order. With several GPUs every process reads a
    different slice of the same window, so no tokens are repeated or skipped.
    State is just (shard index, position), which makes resuming exact.
    """

    def __init__(self, files, B, T, rank, world, shard_idx=0, pos=0, epoch=0):
        assert files, "no shard files found"
        self.files, self.B, self.T, self.rank, self.world = files, B, T, rank, world
        self.epoch = epoch
        self.shard_idx = shard_idx % len(files)
        self._open(self.shard_idx)
        self.pos = pos

    def _open(self, i):
        self.tokens = np.memmap(self.files[i], dtype=np.uint16, mode="r")
        need = self.B * self.T * self.world + 1
        if len(self.tokens) < need:
            raise ValueError(f"{self.files[i]} has {len(self.tokens):,} tokens; "
                             f"need at least {need:,} for this batch size and GPU count")
        self.pos = 0

    def state(self):
        return {"shard_idx": self.shard_idx, "pos": self.pos, "epoch": self.epoch}

    def next_batch(self):
        B, T = self.B, self.T
        start = self.pos + self.rank * B * T
        buf = torch.from_numpy(self.tokens[start:start + B * T + 1].astype(np.int64))
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.pos += B * T * self.world
        if self.pos + B * T * self.world + 1 > len(self.tokens):
            self.shard_idx += 1
            if self.shard_idx == len(self.files):
                self.shard_idx = 0
                self.epoch += 1
            self._open(self.shard_idx)
        return x, y


def assistant_mask(x, assistant_id, special_lo):
    """
    True where the target token belongs to an assistant reply, i.e. the most
    recent special token at or before that input position is <|assistant|>.
    That covers the reply text and its closing <|end|>, and nothing else.
    """
    B, T = x.shape
    is_special = x >= special_lo
    idx = torch.where(is_special, torch.arange(T, device=x.device).expand(B, T), torch.full_like(x, -1))
    last = torch.cummax(idx, dim=1).values
    last_tok = x.gather(1, last.clamp(min=0))
    return (last >= 0) & (last_tok == assistant_id)


def main():
    args = get_args()

    # ---- distributed setup ----------------------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        use_cuda = torch.cuda.is_available()
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
        device = f"cuda:{local_rank}" if use_cuda else "cpu"
        if use_cuda:
            torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    master = rank == 0
    device_type = "cuda" if device.startswith("cuda") else device
    use_amp = device_type == "cuda"

    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    B, T = args.batch_size, args.block_size
    accum = max(1, round(args.total_batch_tokens / (B * T * world)))
    tokens_per_step = B * T * world * accum
    if master:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"device={device} gpus={world} batch={B}x{T} accum={accum} -> {tokens_per_step:,} tokens/step")

    # ---- checkpoint to start from ---------------------------------------
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    elif os.path.exists(os.path.join(args.out_dir, "ckpt.pt")):
        ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
        if master:
            print("found a checkpoint in --out-dir, resuming from it")

    # ---- model ----------------------------------------------------------
    if ckpt is not None:
        cfg = GPTConfig(**ckpt["config"])
    elif args.init_from:
        init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        cfg = GPTConfig(**init["config"])
        cfg.dropout = args.dropout
    else:
        cfg = GPTConfig(vocab_size=args.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    assert T <= cfg.block_size
    model = GPT(cfg)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    elif args.init_from:
        model.load_state_dict(init["model"])
        del init
    model.to(device)
    if master:
        print(f"model: {model.num_params():,} parameters")
    raw_model = model
    if args.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device_type == "cuda" else None)

    optimizer = raw_model.configure_optimizer(args.lr, args.weight_decay, fused=device_type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    best_val = float("inf")
    tokens_seen = 0
    loader_state = {}
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
        best_val = ckpt.get("best_val", best_val)
        tokens_seen = ckpt.get("tokens_seen", 0)
        loader_state = ckpt["loader"]
        if master:
            print(f"resumed at step {step:,} ({tokens_seen / 1e9:.2f}B tokens seen), data position {loader_state}")
    del ckpt

    # ---- data -----------------------------------------------------------
    train_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "train_*.bin"), recursive=True))
    val_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "val_*.bin"), recursive=True))
    if master:
        print(f"{len(train_files)} train shards, {len(val_files)} val shards")
    train_loader = ShardLoader(train_files, B, T, rank, world, **loader_state)

    special_lo = assistant_id = None
    if args.chat_mask:
        assert args.tokenizer, "--chat-mask needs --tokenizer"
        from tokenizer.bpe import BPETokenizer
        tok = BPETokenizer.load(args.tokenizer)
        assert tok.vocab_size <= cfg.vocab_size
        special_lo, assistant_id = min(tok.special.values()), tok.assistant_id

    def to_device(x, y):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = assistant_mask(x, assistant_id, special_lo) if args.chat_mask else None
        return x, y, mask

    def get_lr(it):
        if it < args.warmup_steps:
            return args.lr * (it + 1) / args.warmup_steps
        if it >= args.max_steps:
            return args.lr * args.min_lr_frac
        ratio = (it - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return args.lr * (args.min_lr_frac + coeff * (1 - args.min_lr_frac))

    @torch.no_grad()
    def evaluate():
        if not val_files:
            return float("nan")
        model.eval()
        loader = ShardLoader(val_files, B, T, rank, world)
        total = torch.zeros((), device=device)
        for _ in range(args.eval_batches):
            x, y, mask = to_device(*loader.next_batch())
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y, loss_mask=mask)
            total += loss.float()
        total /= args.eval_batches
        if ddp:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            total /= world
        model.train()
        return total.item()

    def save_checkpoint(name="ckpt.pt"):
        path = os.path.join(args.out_dir, name)
        tmp = path + ".tmp"
        torch.save({
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(cfg),
            "step": step,
            "best_val": best_val,
            "tokens_seen": tokens_seen,
            "loader": train_loader.state(),
            "args": vars(args),
        }, tmp)
        os.replace(tmp, path)   # never leaves a half-written checkpoint behind

    log_path = os.path.join(args.out_dir, "log.csv")
    if master and not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "tokens_seen", "train_loss", "val_loss", "lr", "tok_per_sec"])

    def log_row(train_loss, val_loss, lr, tps):
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([step, tokens_seen, train_loss, val_loss, lr, tps])

    # ---- training loop --------------------------------------------------
    t_start = time.time()
    t_last_ckpt = time.time()
    model.train()
    last_loss = float("nan")

    while step < args.max_steps:
        t0 = time.time()

        if step % args.eval_every == 0:
            val_loss = evaluate()
            if master:
                print(f"step {step:6d} | val loss {val_loss:.4f}")
                log_row("", val_loss, "", "")
                best_val = min(best_val, val_loss)

        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)
        for _ in range(accum):
            x, y, mask = to_device(*train_loader.next_batch())
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y, loss_mask=mask)
            loss = loss / accum
            loss_accum += loss.detach().float()
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()

        step += 1
        tokens_seen += tokens_per_step

        # All processes must agree on when to stop, so share the time-limit flag.
        out_of_time = torch.tensor(float(time.time() - t_start > args.max_hours * 3600), device=device)
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
            loss_accum /= world
            dist.all_reduce(out_of_time, op=dist.ReduceOp.MAX)
        stop = bool(out_of_time.item()) or step >= args.max_steps
        last_loss = loss_accum.item()

        if master:
            if not math.isfinite(last_loss):
                print("loss is not finite; stopping without overwriting the last good checkpoint")
            if step % args.log_every == 0 or stop:
                if device_type == "cuda":
                    torch.cuda.synchronize()
                tps = tokens_per_step / (time.time() - t0)
                print(f"step {step:6d} | loss {last_loss:.4f} | lr {lr:.2e} | grad norm {float(grad_norm):.2f} | "
                      f"{tps:,.0f} tok/s | {tokens_seen / 1e9:.3f}B tokens | {(time.time() - t_start) / 3600:.2f}h")
                log_row(last_loss, "", lr, round(tps))
            if math.isfinite(last_loss) and (stop or time.time() - t_last_ckpt > args.ckpt_minutes * 60):
                save_checkpoint()
                t_last_ckpt = time.time()
                print(f"checkpoint saved at step {step:,}")
        if not math.isfinite(last_loss) or stop:
            break

    if step >= args.max_steps and math.isfinite(last_loss):
        val_loss = evaluate()
        if master:
            print(f"final val loss {val_loss:.4f}")
            log_row("", val_loss, "", "")
            save_checkpoint("final.pt")
    elif master:
        print(f"stopped at step {step:,} of {args.max_steps:,}; resume with --resume {os.path.join(args.out_dir, 'ckpt.pt')}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
