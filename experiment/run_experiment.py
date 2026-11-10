# -*- coding: utf-8 -*-
"""
A small, self-contained SFT vs GRPO experiment on two-digit addition.

The task is verifiable: the reward is exact match of the final integer.
The model is a tiny character-level causal Transformer trained from scratch.

Usage:
    python run_experiment.py --out results --quick
    python run_experiment.py --out results
"""
import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    seed: int = 42
    n_train: int = 5000
    n_val: int = 500
    n_test: int = 500
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 24
    dropout: float = 0.0
    sft_epochs: int = 10
    sft_batch: int = 64
    sft_lr: float = 3e-3
    sft_weight_decay: float = 0.01
    grpo_steps: int = 200
    grpo_batch_prompts: int = 8
    grpo_group: int = 8
    grpo_lr: float = 3e-5
    grpo_beta: float = 0.005
    grpo_clip: float = 0.2
    max_new_tokens: int = 6
    temperature: float = 1.5
    dense_reward: bool = False


# ---------------------------------------------------------------------------
# Tokenizer / data
# ---------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self):
        chars = sorted(set("Q:0123456789+= "))
        self.itos = ["<pad>", "<bos>", "<eos>", "<unk>"] + chars
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i in (self.pad_id, self.bos_id):
                continue
            if i == self.eos_id:
                break
            out.append(self.itos[i])
        return "".join(out)

    def prompt_ids(self, prompt: str) -> List[int]:
        return [self.bos_id] + self.encode(prompt)

    def example_ids(self, prompt: str, answer: str) -> Tuple[List[int], List[int]]:
        p = self.prompt_ids(prompt)
        a = self.encode(answer) + [self.eos_id]
        ids = p + a
        labels = [-100] * len(p) + a
        return ids, labels


def make_pairs(n_train: int, n_val: int, n_test: int, seed: int):
    pairs = [(a, b) for a in range(100) for b in range(100)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    total = n_train + n_val + n_test
    if total > len(pairs):
        raise ValueError("not enough unique pairs")
    out = []
    for a, b in pairs[:total]:
        out.append({"prompt": f"Q:{a}+{b}=", "answer": str(a + b)})
    return out[:n_train], out[n_train:n_train + n_val], out[n_train + n_val:total]


def write_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, key_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        mask = causal.view(1, 1, T, T)
        if key_mask is not None:
            mask = mask & key_mask[:, None, None, :].bool()
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, key_mask=None):
        x = x + self.attn(self.ln1(x), key_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, key_mask=None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, key_mask)
        x = self.ln_f(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Batching / generation
# ---------------------------------------------------------------------------
def make_sft_batch(rows, tokenizer, cfg, shuffle=True):
    order = list(range(len(rows)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(rows), cfg.sft_batch):
        batch = [rows[i] for i in order[start:start + cfg.sft_batch]]
        seqs, labels = [], []
        for row in batch:
            ids, labs = tokenizer.example_ids(row["prompt"], row["answer"])
            seqs.append(ids)
            labels.append(labs)
        T = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), T), tokenizer.pad_id, dtype=torch.long)
        lab = torch.full((len(seqs), T), -100, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, labels)):
            input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            lab[i, :len(l)] = torch.tensor(l, dtype=torch.long)
            mask[i, :len(s)] = 1
        yield input_ids, lab, mask


@torch.no_grad()
def generate(model, prompt_ids: List[int], tokenizer, cfg, sample=False, temperature=1.0):
    model.eval()
    ids = list(prompt_ids)
    generated = []
    for _ in range(cfg.max_new_tokens):
        idx = torch.tensor([ids[-cfg.block_size:]], dtype=torch.long, device=next(model.parameters()).device)
        logits = model(idx)[0, -1]
        if sample:
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
        else:
            nxt = int(torch.argmax(logits).item())
        ids.append(nxt)
        if nxt == tokenizer.eos_id:
            break
        generated.append(nxt)
    return generated, tokenizer.decode(generated)


@torch.no_grad()
def sample_with_logp(model, prompt_ids: List[int], tokenizer, cfg):
    model.eval()
    device = next(model.parameters()).device
    ids = list(prompt_ids)
    generated = []
    logp_sum = 0.0
    for _ in range(cfg.max_new_tokens):
        idx = torch.tensor([ids[-cfg.block_size:]], dtype=torch.long, device=device)
        logits = model(idx)[0, -1]
        log_probs = F.log_softmax(logits / max(cfg.temperature, 1e-6), dim=-1)
        probs = log_probs.exp()
        nxt = int(torch.multinomial(probs, 1).item())
        logp_sum += float(log_probs[nxt].item())
        ids.append(nxt)
        if nxt == tokenizer.eos_id:
            break
        generated.append(nxt)
    return ids, generated, logp_sum, tokenizer.decode(generated)


def extract_last_int(text: str):
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def verifiable_reward(text: str, target: str, dense: bool = False) -> float:
    """Exact-match reward, optionally with digit-level partial credit."""
    pred = extract_last_int(text)
    if pred is None:
        return 0.0
    if pred == int(target):
        return 1.0
    if not dense:
        return 0.0
    sp, st = str(pred), str(int(target))
    if len(sp) < len(st):
        sp = sp.zfill(len(st))
    sp = sp[-len(st):]
    matches = sum(a == b for a, b in zip(sp, st))
    return 0.25 * matches / max(1, len(st))


@torch.no_grad()
def evaluate(model, rows, tokenizer, cfg):
    model.eval()
    correct, lengths = 0, []
    preds = []
    for row in rows:
        prompt_ids = tokenizer.prompt_ids(row["prompt"])
        generated, text = generate(model, prompt_ids, tokenizer, cfg, sample=False)
        pred = extract_last_int(text)
        ok = pred == int(row["answer"])
        correct += int(ok)
        lengths.append(len(generated))
        preds.append({"prompt": row["prompt"], "target": row["answer"], "pred": text, "ok": ok})
    n = max(1, len(rows))
    return correct / n, sum(lengths) / n, preds


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------
def train_sft(cfg, tokenizer, train_rows, val_rows, out_dir, device):
    model = TinyGPT(cfg, tokenizer.vocab_size, tokenizer.pad_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.sft_lr, weight_decay=cfg.sft_weight_decay)
    steps_per_epoch = math.ceil(len(train_rows) / cfg.sft_batch)
    total_steps = steps_per_epoch * cfg.sft_epochs
    warmup = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    history = []
    best_acc = -1.0
    best_state = None
    t0 = time.perf_counter()
    global_step = 0
    for epoch in range(1, cfg.sft_epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for input_ids, labels, mask in make_sft_batch(train_rows, tokenizer, cfg):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            logits = model(input_ids, mask)
            # Causal LM shift: logits at position t predict token t+1.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += float(loss.item())
            n_batches += 1
            global_step += 1
            history.append({"step": global_step, "epoch": epoch, "train_loss": float(loss.item()),
                            "lr": float(sched.get_last_lr()[0])})
        val_acc, val_len, _ = evaluate(model, val_rows, tokenizer, cfg)
        train_loss = running / max(1, n_batches)
        history.append({"epoch": epoch, "train_loss_epoch": train_loss, "val_acc": val_acc,
                        "val_len": val_len, "lr": float(sched.get_last_lr()[0])})
        print(f"[SFT] epoch {epoch:02d}/{cfg.sft_epochs} loss={train_loss:.4f} val_acc={val_acc:.4f} len={val_len:.2f}", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    sft_metrics = {
        "epochs": cfg.sft_epochs,
        "steps": global_step,
        "train_samples": len(train_rows) * cfg.sft_epochs,
        "train_time_sec": train_time,
        "best_val_acc": best_acc,
        "params": sum(p.numel() for p in model.parameters()),
    }
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "tokenizer": tokenizer.itos,
                "metrics": sft_metrics}, os.path.join(out_dir, "sft.pt"))
    return model, history, sft_metrics


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------
def sequence_logp(model, input_ids, completion_mask, pad_id, temperature=1.0):
    key_mask = (input_ids != pad_id).long()
    logits = model(input_ids, key_mask)
    log_probs = F.log_softmax(logits[:, :-1, :] / max(temperature, 1e-6), dim=-1)
    targets = input_ids[:, 1:]
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = completion_mask[:, 1:].float()
    return (token_logp * mask).sum(dim=1), token_logp, mask


def train_grpo(cfg, tokenizer, sft_model, train_rows, val_rows, out_dir, device):
    policy = TinyGPT(cfg, tokenizer.vocab_size, tokenizer.pad_id).to(device)
    policy.load_state_dict(sft_model.state_dict())
    ref = TinyGPT(cfg, tokenizer.vocab_size, tokenizer.pad_id).to(device)
    ref.load_state_dict(sft_model.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.grpo_lr, weight_decay=0.0)
    history = []
    best_acc = -1.0
    best_state = None
    t0 = time.perf_counter()
    generated_tokens = 0
    train_tokens = 0
    prompts_seen = 0

    for step in range(1, cfg.grpo_steps + 1):
        batch = random.sample(train_rows, min(cfg.grpo_batch_prompts, len(train_rows)))
        samples = []
        for row in batch:
            pids = tokenizer.prompt_ids(row["prompt"])
            group = []
            for _ in range(cfg.grpo_group):
                seq, gen, logp, text = sample_with_logp(policy, pids, tokenizer, cfg)
                reward = verifiable_reward(text, row["answer"], cfg.dense_reward)
                group.append({"seq": seq, "gen": gen, "prompt_len": len(pids), "target": row["answer"],
                              "old_logp": logp, "reward": reward, "text": text})
                generated_tokens += len(gen)
            rewards = torch.tensor([g["reward"] for g in group], dtype=torch.float32)
            adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
            for g, a in zip(group, adv.tolist()):
                g["adv"] = a
                samples.append(g)
            prompts_seen += 1

        max_len = max(len(s["seq"]) for s in samples)
        input_ids = torch.full((len(samples), max_len), tokenizer.pad_id, dtype=torch.long)
        comp_mask = torch.zeros((len(samples), max_len), dtype=torch.long)
        for i, s in enumerate(samples):
            input_ids[i, :len(s["seq"])] = torch.tensor(s["seq"], dtype=torch.long)
            comp_mask[i, s["prompt_len"]:len(s["seq"])] = 1  # generated tokens + optional EOS
        input_ids = input_ids.to(device)
        comp_mask = comp_mask.to(device)
        adv = torch.tensor([s["adv"] for s in samples], dtype=torch.float32, device=device)
        old_logp = torch.tensor([s["old_logp"] for s in samples], dtype=torch.float32, device=device)

        policy.train()
        new_logp, new_token_logp, mask = sequence_logp(policy, input_ids, comp_mask, tokenizer.pad_id, cfg.temperature)
        with torch.no_grad():
            ref_logp, ref_token_logp, _ = sequence_logp(ref, input_ids, comp_mask, tokenizer.pad_id, cfg.temperature)
        log_ratio = (new_logp - old_logp).clamp(-10.0, 10.0)
        ratio = log_ratio.exp()
        clipped = ratio.clamp(1.0 - cfg.grpo_clip, 1.0 + cfg.grpo_clip)
        obj = torch.min(ratio * adv, clipped * adv)
        # Token-level unbiased KL estimate (K3), masked to generated tokens.
        d = ref_token_logp - new_token_logp
        kl_tokens = torch.exp(d) - d - 1.0
        kl = (kl_tokens * mask).sum() / mask.sum().clamp_min(1.0)
        loss = -obj.mean() + cfg.grpo_beta * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        train_tokens += int(input_ids.numel())

        if step % 5 == 0 or step == 1 or step == cfg.grpo_steps:
            val_acc, val_len, _ = evaluate(policy, val_rows, tokenizer, cfg)
            mean_reward = float(torch.tensor([s["reward"] for s in samples]).mean().item())
            exact_acc = sum(1 for s in samples if extract_last_int(s["text"]) == int(s["target"])) / len(samples)
            mean_len = sum(len(s["gen"]) for s in samples) / len(samples)
            history.append({
                "step": step, "loss": float(loss.item()), "pg_loss": float(-obj.mean().item()),
                "grad_norm": float(grad_norm), "mean_reward": mean_reward,
                "sample_acc": exact_acc, "mean_len": mean_len, "val_acc": val_acc,
                "val_len": val_len, "kl": float(kl.item()),
            })
            print(f"[GRPO] step {step:03d}/{cfg.grpo_steps} loss={loss.item():.4f} "
                  f"reward={mean_reward:.3f} val_acc={val_acc:.4f} len={val_len:.2f}", flush=True)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    train_time = time.perf_counter() - t0
    if best_state is not None:
        policy.load_state_dict(best_state)
    grpo_metrics = {
        "steps": cfg.grpo_steps,
        "prompts_seen": prompts_seen,
        "generations": prompts_seen * cfg.grpo_group,
        "generated_tokens": generated_tokens,
        "train_tokens": train_tokens,
        "train_time_sec": train_time,
        "best_val_acc": best_acc,
        "params": sum(p.numel() for p in policy.parameters()),
    }
    torch.save({"model": policy.state_dict(), "config": asdict(cfg), "tokenizer": tokenizer.itos,
                "metrics": grpo_metrics}, os.path.join(out_dir, "grpo.pt"))
    return policy, history, grpo_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sft-only", action="store_true")
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--n-val", type=int)
    ap.add_argument("--n-test", type=int)
    ap.add_argument("--sft-epochs", type=int)
    ap.add_argument("--n-layer", type=int)
    ap.add_argument("--n-embd", type=int)
    ap.add_argument("--grpo-steps", type=int)
    ap.add_argument("--grpo-lr", type=float)
    ap.add_argument("--grpo-beta", type=float)
    ap.add_argument("--grpo-group", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--dense-reward", action="store_true")
    ap.add_argument("--init-sft", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = Config()
    if args.quick:
        cfg.n_train, cfg.n_val, cfg.n_test = 600, 120, 200
        cfg.sft_epochs = 5
        cfg.grpo_steps = 20
        cfg.grpo_batch_prompts = 4
        cfg.grpo_group = 4
    for name in ("n_train", "n_val", "n_test", "sft_epochs", "n_layer", "n_embd", "grpo_steps",
                 "grpo_lr", "grpo_beta", "grpo_group", "temperature"):
        val = getattr(args, name)
        if val is not None:
            setattr(cfg, name, val)
    if args.sft_only:
        cfg.grpo_steps = 0
    if args.dense_reward:
        cfg.dense_reward = True
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()

    train_rows, val_rows, test_rows = make_pairs(cfg.n_train, cfg.n_val, cfg.n_test, cfg.seed)
    data_dir = os.path.join(args.out, "data")
    os.makedirs(data_dir, exist_ok=True)
    write_jsonl(os.path.join(data_dir, "train.jsonl"), train_rows)
    write_jsonl(os.path.join(data_dir, "val.jsonl"), val_rows)
    write_jsonl(os.path.join(data_dir, "test.jsonl"), test_rows)

    print(f"device={device} params≈{sum(p.numel() for p in TinyGPT(cfg, tokenizer.vocab_size, tokenizer.pad_id).parameters()):,}", flush=True)
    if args.init_sft:
        ck = torch.load(args.init_sft, map_location="cpu")
        sft_model = TinyGPT(cfg, tokenizer.vocab_size, tokenizer.pad_id).to(device)
        sft_model.load_state_dict(ck["model"])
        sft_hist = []
        sft_metrics = ck["metrics"]
    else:
        sft_model, sft_hist, sft_metrics = train_sft(cfg, tokenizer, train_rows, val_rows, args.out, device)
    if cfg.grpo_steps > 0:
        grpo_model, grpo_hist, grpo_metrics = train_grpo(cfg, tokenizer, sft_model, train_rows, val_rows, args.out, device)
    else:
        grpo_model = sft_model
        grpo_hist = []
        grpo_metrics = {"steps": 0, "prompts_seen": 0, "generations": 0, "generated_tokens": 0,
                        "train_tokens": 0, "train_time_sec": 0.0, "best_val_acc": sft_metrics["best_val_acc"],
                        "params": sft_metrics["params"]}

    sft_acc, sft_len, sft_preds = evaluate(sft_model, test_rows, tokenizer, cfg)
    grpo_acc, grpo_len, grpo_preds = evaluate(grpo_model, test_rows, tokenizer, cfg)

    result = {
        "config": asdict(cfg),
        "device": str(device),
        "params": sft_metrics["params"],
        "sft": {**sft_metrics, "test_acc": sft_acc, "test_len": sft_len},
        "grpo": {**grpo_metrics, "test_acc": grpo_acc, "test_len": grpo_len},
        "delta": {"test_acc": grpo_acc - sft_acc, "test_len": grpo_len - sft_len},
        "sft_history": sft_hist,
        "grpo_history": grpo_hist,
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "sft_predictions.jsonl"), "w", encoding="utf-8") as f:
        for p in sft_preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "grpo_predictions.jsonl"), "w", encoding="utf-8") as f:
        for p in grpo_preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("sft", "grpo", "delta")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
