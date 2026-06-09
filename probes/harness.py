#!/usr/bin/env python3
"""Probe harness for computational historical characterology.

Two model backends with one interface:
  - NanochatModel: the governed_v4 d22 615M pre-1913 checkpoint + historical tokenizer
  - HFModel:       a HuggingFace causal base LM (e.g. gpt2) as a modern anchor

Scoring follows the probe-design rules:
  - score a candidate continuation by the summed conditional log-prob of its
    tokens given the prefix, length-normalized by BYTES (not tokens), so the
    number is comparable across tokenizers ordinally.
  - candidate token span = joint_encode(prefix+candidate)[len(encode(prefix)):],
    so the boundary token is attributed consistently across a probe's candidates.
  - NEVER compare raw cross-tokenizer likelihoods; compare within-model
    preference orderings, then compare orderings across models.
"""
import os, sys
import torch

ROOT = "/home/user/claudeworkspace/research/historical-nanochat"
os.environ.setdefault("NANOCHAT_BASE_DIR", ROOT)
sys.path.insert(0, os.path.join(ROOT, "nanochat"))


class NanochatModel:
    name = "nanochat_pre1913_615m"
    label = "Pre-1913 nanochat 615M (governed_v4 d22)"

    def __init__(self, step=70455, device=None):
        from nanochat.gpt import GPT, GPTConfig
        from nanochat.checkpoint_manager import load_checkpoint
        from nanochat.tokenizer import get_tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_dir = os.path.join(ROOT, "base_checkpoints/governed_v4_d22_r30_parallel_family")
        model_data, _, meta = load_checkpoint(ckpt_dir, step, self.device, load_optimizer=False)
        self.cfg = GPTConfig(**meta["model_config"])
        m = GPT(self.cfg).to(self.device)
        sd = model_data
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        m.load_state_dict(sd, strict=True)
        m.eval()
        self.model = m
        self.tok = get_tokenizer()
        self.bos = self.tok.get_bos_token_id()
        self.max_len = self.cfg.sequence_len

    def _encode(self, text):
        return [self.bos] + self.tok.encode(text)

    @torch.no_grad()
    def score(self, prefix, candidate):
        pre = self._encode(prefix)
        full = self._encode(prefix + candidate)
        n_pre = len(pre)
        if len(full) <= n_pre or len(full) > self.max_len:
            return None
        x = torch.tensor([full], dtype=torch.long, device=self.device)
        logits = self.model(x)  # (1, T, V)
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        total = 0.0
        for pos in range(n_pre - 1, len(full) - 1):
            total += logp[pos, full[pos + 1]].item()
        nbytes = max(1, len(candidate.encode("utf-8")))
        return {"sum": total, "per_byte": total / nbytes, "n_cand_tok": len(full) - n_pre}

    @torch.no_grad()
    def generate(self, prefix, n=60, temperature=0.8, top_k=40):
        ids = self._encode(prefix)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = []
        for _ in range(n):
            if x.size(1) >= self.max_len:
                break
            logits = self.model(x)[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
            out.append(nxt.item())
        return self.tok.decode(out)


class HFModel:
    def __init__(self, hf_id="gpt2", device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = f"hf_{hf_id.replace('/', '_')}"
        self.label = f"Modern base LM ({hf_id})"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(hf_id).to(self.device).eval()
        self.bos = self.tok.bos_token_id
        self.max_len = getattr(self.model.config, "n_positions", 1024)

    def _encode(self, text):
        ids = self.tok.encode(text)
        if self.bos is not None and (not ids or ids[0] != self.bos):
            ids = [self.bos] + ids
        return ids

    @torch.no_grad()
    def score(self, prefix, candidate):
        pre = self._encode(prefix)
        full = self._encode(prefix + candidate)
        n_pre = len(pre)
        if len(full) <= n_pre or len(full) > self.max_len:
            return None
        x = torch.tensor([full], dtype=torch.long, device=self.device)
        logits = self.model(x).logits
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        total = 0.0
        for pos in range(n_pre - 1, len(full) - 1):
            total += logp[pos, full[pos + 1]].item()
        nbytes = max(1, len(candidate.encode("utf-8")))
        return {"sum": total, "per_byte": total / nbytes, "n_cand_tok": len(full) - n_pre}

    @torch.no_grad()
    def generate(self, prefix, n=60, temperature=0.8, top_k=40):
        ids = self._encode(prefix)
        out = self.model.generate(
            torch.tensor([ids], device=self.device),
            max_new_tokens=n, do_sample=True, temperature=temperature, top_k=top_k,
            pad_token_id=self.tok.eos_token_id,
        )
        return self.tok.decode(out[0][len(ids):])


def rank_candidates(model, prefix, candidates):
    """candidates: dict label->text. Returns list of (label, per_byte, sum) sorted desc."""
    rows = []
    for label, text in candidates.items():
        s = model.score(prefix, text)
        if s is None:
            continue
        rows.append((label, s["per_byte"], s["sum"], s["n_cand_tok"]))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows
