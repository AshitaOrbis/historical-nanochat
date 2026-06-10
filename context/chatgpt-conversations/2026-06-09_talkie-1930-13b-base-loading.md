# Talkie-1930-13b-Base Loading

- Source: https://chatgpt.com/c/6a28a33d-97c4-83e8-9a6b-93a6243a193a
- Conversation ID: 6a28a33d-97c4-83e8-9a6b-93a6243a193a
- Created: 2026-06-09T23:35:26.002Z
- Archived: false
- Messages: 4
- Captured: 2026-06-09 (Fable-5 session, CDP backend-api content-search)

---

## USER

I need to run a specific historical language model locally for research and I'm hitting a loading problem. Please investigate carefully and give a concrete, correct recipe.

THE MODEL: "talkie-1930-13b-base" — a 13B-parameter "vintage" base language model trained on pre-1931 English text, released ~April 2026 by talkie-lm (Alec Radford, Nick Levine, David Duvenaud). HuggingFace org: talkie-lm. The official base repo `talkie-lm/talkie-1930-13b-base` ships a custom format: files `final.ckpt` + `vocab.txt` (NOT standard HF transformers). There's an official inference library at github.com/talkie-lm/talkie. There's also a matched modern twin `talkie-lm/talkie-web-13b-base` (same architecture/FLOPs, trained on FineWeb).

MY GOAL: I do NOT need text generation. I need per-token LOG-PROBABILITIES (logits) so I can score the likelihood of specific candidate continuations given a prefix (forced-choice posture probes). I have a single RTX 3090 (24 GB VRAM), 62 GB system RAM, Python 3.12, PyTorch 2.12+cu130, transformers 5.5.4, gptqmodel 7.1.0, triton 3.7 installed.

WHAT I OBSERVED (and suspect I misdiagnosed): I tried the community GPTQ-int4 conversion `dtestnyrr/talkie-1930-13b-base-gptq-int4`. Its config.json says vocab_size=65536 and architectures=["TalkieForCausalLM"] with custom modeling_talkie.py (trust_remote_code). But the TOKENIZER bundled in that same repo reports vocab_size=262144, and encoding ordinary English produced token ids up to ~222,758 — which are out of range for a 65536-row embedding, causing CUDA device-side asserts on any text hitting rarer tokens. The official `vocab.txt` in talkie-lm/talkie-1930-13b-base has 262,144 lines. Several other community HF conversions (niklassheth/talkie-1930-13b-base-hf, xlr8harder/talkie-1930-13b-base-tf, jannalu/talkie-1930-13b-base-hf) all report config vocab_size=65536.

KEY QUESTIONS — please resolve definitively:
1. What is talkie-1930-13b-base's TRUE vocabulary size — 65,536 or 262,144? Reconcile the 262,144-line vocab.txt with the 65,536 model configs. (Is vocab.txt a byte/merge artifact? Is the real tokenizer 65,536? Is the dtestnyrr repo simply bundling a WRONG tokenizer, e.g. a Gemma 256k tokenizer, against a correct 65,536 model?)
2. Which specific community conversion is KNOWN-GOOD for transformers + logits, especially anything by xlr8harder (the user is certain xlr8harder published working talkie checkpoints)? Give exact HF repo IDs.
3. What is the correct tokenizer to pair with the model, and how do I load it so token ids stay in-range? If the model is 65,536-vocab, where do I get the matching 65,536 tokenizer (repo + filename)?
4. Concrete loading recipe to get logits on a 24 GB GPU: e.g., load a correct GPTQ-int4 or AWQ build (≈7-9 GB) with AutoModelForCausalLM(trust_remote_code=True), OR load bf16 (~26 GB) with bitsandbytes 4-bit, OR the official talkie library's underlying nn.Module + a manual forward. Include the exact tokenizer-loading line that produces in-range ids. Note: the official talkie lib README says it needs ≥28 GB VRAM for bf16 — how do people run it on a 24 GB card?
5. Does the official talkie GitHub library expose logits/logprobs anywhere (e.g., a forward() on the underlying module, an eval/score function), or must I call the raw nn.Module myself?

Please be concrete and cite exact HF repo IDs, filenames, config fields, and code. If you can check the actual talkie-lm HF repos / the github.com/talkie-lm/talkie source / xlr8harder's repos, do so. The single most important deliverable: the exact (model_repo, tokenizer_source, loading_code) triple that yields correct, in-range logprobs on a 24 GB GPU.

---

## ASSISTANT

I’ll check the official repo, talkie source, and community conversions directly, then resolve the tokenizer/config mismatch and produce a concrete 24 GB logits recipe.

---

## ASSISTANT

I’ve found the key reconciliation: the official code uses a 65,536-token model vocabulary by filtering the 262,144-line BPE file. Your out-of-range IDs point to a wrongly packaged tokenizer.

---

## ASSISTANT

## Bottom line

Use this triple for a 24 GB RTX 3090:

```text
weights/model repo:      dtestnyrr/talkie-1930-13b-base-gptq-int4
tokenizer source repo:   xlr8harder/talkie-1930-13b-base-tf
tokenizer files used:    tokenization_talkie.py + tokenizer_config.json + vocab.txt
loader:                  GPTQModel.load(...) for weights, AutoTokenizer.from_pretrained(..., trust_remote_code=True) from xlr8harder
```

Do **not** use the tokenizer bundled in `dtestnyrr/talkie-1930-13b-base-gptq-int4`. The model vocabulary is **65,536**, and any tokenizer producing IDs like `222758` is wrong for the 65,536-row embedding.

---

## 1. True vocabulary size: 65,536, not 262,144

The official base model’s effective model vocabulary is **65,536**. The 262,144-line `vocab.txt` is a tiktoken/BPE rank file with more ranks than the model uses; the official tokenizer code explicitly filters it down.

The official GitHub code defines `BASE_VOCAB_SIZE = 65536`, maps `<|endoftext|>` to ID `65535`, and builds the tokenizer by loading the BPE ranks and keeping only ranks `< BASE_VOCAB_SIZE - 1`. That means the base tokenizer has **65,535 mergeable tokens plus one special token**, not 262,144 embedding rows. citeturn168192view1

The official model code is consistent with that: the `GPTConfig` default `vocab_size` is `65536`, the embedding is `nn.Embedding(config.vocab_size, ...)`, and the LM head projects to `config.vocab_size`. The checkpoint loader also infers `vocab_size` from `state_dict["embed.weight"].shape[0]`, so the embedding matrix is authoritative. citeturn168192view0

So the reconciliation is:

```text
vocab.txt lines / BPE ranks available: 262,144
Talkie base model usable token IDs:    0..65,535
EOS token ID:                          65,535
Instruction-tuned extra specials:      65,536..65,539
Base model vocab_size:                 65,536
IT model vocab_size:                   65,540
```

Your diagnosis of the failure mode is right, but the conclusion should be: **the dtestnyrr tokenizer packaging is wrong/incompatible**, not that the model has a 262,144-token embedding. The dtestnyrr model card itself describes the architecture as `vocab = 65536` and “65,535 merges + 1 special token,” while its published `config.json` has `vocab_size: 65536` but `eos_token_id: 262143` / `pad_token_id: 262143`, which is internally inconsistent for this model. citeturn871090view0turn481433view0

---

## 2. Known-good community conversion for Transformers/logits

The community conversion I would treat as the known-good Transformers reference is:

```text
xlr8harder/talkie-1930-13b-base-tf
```

Its model card says it is a Transformers-compatible BF16 safetensors conversion of `talkie-lm/talkie-1930-13b-base`, includes custom Talkie modeling/tokenizer code, and reports validation against the original Talkie runtime: top-10 logits matched exactly on the tested pass, with max absolute logit difference `0.03125` after BF16 casting. citeturn705257view0

The repo contains the expected files: sharded BF16 safetensors, `config.json`, `modeling_talkie.py`, `tokenization_talkie.py`, `tokenizer_config.json`, and `vocab.txt`. citeturn705257view1 Its `config.json` uses `vocab_size: 65536`, `eos_token_id: 65535`, and `pad_token_id: 65535`. citeturn322369view0 Its tokenizer code also uses `BASE_VOCAB_SIZE = 65536` and filters the BPE ranks down to the Talkie base vocabulary. citeturn322369view1

The xlr8harder Talkie repos I found are:

```text
xlr8harder/talkie-1930-13b-base-tf       # original 1930 base, BF16 Transformers
xlr8harder/talkie-web-13b-base-tf        # modern FineWeb twin, BF16 Transformers
xlr8harder/talkie-1930-13b-it-tf         # instruction-tuned variant
xlr8harder/talkie-1930-13b-yarn-32k-tf   # long-context derivative, not identical base likelihoods
```

Those appear in xlr8harder’s Talkie collection. citeturn924605view0 For strict likelihood research against the released base model, use `xlr8harder/talkie-1930-13b-base-tf`, not the YaRN derivative.

For a 24 GB GPU, the xlr8harder BF16 model alone is too large to place fully on the GPU: its safetensors shards total about 26.6 GB, before runtime overhead. citeturn705257view1 That makes it the correctness reference, but not the easiest 3090 path unless you use CPU offload or 4-bit quantization.

---

## 3. Correct tokenizer

For a Transformers-style tokenizer, use:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "xlr8harder/talkie-1930-13b-base-tf",
    trust_remote_code=True,
    use_fast=False,
)

assert tok.vocab_size == 65536
assert tok.eos_token_id == 65535
```

That loads `TalkieTokenizer` from `tokenization_talkie.py`, not a generic 262k fast tokenizer. The relevant files are in `xlr8harder/talkie-1930-13b-base-tf`: `tokenization_talkie.py`, `tokenizer_config.json`, and `vocab.txt`. citeturn705257view1turn506666view0

For the official non-Transformers tokenizer, the equivalent is:

```python
from huggingface_hub import hf_hub_download
from talkie.tokenizer import build_tokenizer

vocab_path = hf_hub_download(
    repo_id="talkie-lm/talkie-1930-13b-base",
    filename="vocab.txt",
)

enc = build_tokenizer(vocab_path, style="base")
ids = enc.encode("Some ordinary English text")
assert max(ids) < 65536
```

The official base HF repo contains only `final.ckpt`, `vocab.txt`, `.gitattributes`, and `README.md`; it is not a standard Transformers repo. citeturn484024view1

---

## 4. Concrete 24 GB recipe for logits/logprobs

### Recommended 24 GB path: dtestnyrr GPTQ weights + xlr8harder tokenizer

This uses the ~8 GB GPTQ model for VRAM fit, but replaces the bad tokenizer with the verified 65,536-token Talkie tokenizer.

```python
# pip install -U gptqmodel transformers huggingface_hub tiktoken safetensors accelerate

import sys
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from gptqmodel import GPTQModel

WEIGHTS_REPO = "dtestnyrr/talkie-1930-13b-base-gptq-int4"
TOKENIZER_REPO = "xlr8harder/talkie-1930-13b-base-tf"

# 1) Load the correct 65,536-token Talkie tokenizer.
tok = AutoTokenizer.from_pretrained(
    TOKENIZER_REPO,
    trust_remote_code=True,
    use_fast=False,
)

assert tok.vocab_size == 65536, tok.vocab_size
assert tok.eos_token_id == 65535, tok.eos_token_id

# 2) Load GPTQ weights. Ignore/avoid the tokenizer attached to the dtestnyrr repo.
model_dir = snapshot_download(WEIGHTS_REPO)

# Some GPTQModel versions need the repo's registration file imported explicitly.
sys.path.insert(0, model_dir)
try:
    import talkie_qmodel  # noqa: F401
except Exception as e:
    print(f"Warning: explicit talkie_qmodel import failed: {e}")

q = GPTQModel.load(model_dir, trust_remote_code=True)

# GPTQModel may return a wrapper; the underlying HF module is usually q.model.
model = getattr(q, "model", q)
model.eval()

cfg = getattr(model, "config", getattr(q, "config", None))
if cfg is not None:
    cfg.eos_token_id = tok.eos_token_id
    cfg.pad_token_id = tok.eos_token_id

vocab_size = int(getattr(cfg, "vocab_size", 65536))
assert vocab_size == 65536, vocab_size

device = next(model.parameters()).device
print("model device:", device)
print("tokenizer vocab_size:", tok.vocab_size)


def encode_checked(text: str) -> torch.Tensor:
    ids = tok(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids

    if ids.numel() and int(ids.max()) >= vocab_size:
        raise RuntimeError(
            f"Tokenizer produced out-of-range ID {int(ids.max())}; "
            f"model vocab_size is {vocab_size}. "
            f"You are not using the Talkie 65,536-token tokenizer."
        )

    return ids.to(device)


@torch.inference_mode()
def continuation_logprobs(prefix: str, continuation: str):
    """
    Returns per-token log-probs for `continuation` conditioned on `prefix`.

    Important: this assumes the tokenizer boundary is stable, i.e.
    tokenize(prefix + continuation) starts with tokenize(prefix).
    For forced-choice probes, make continuations include the exact leading
    space/punctuation you intend to score.
    """
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    full_ids = tok.encode(prefix + continuation, add_special_tokens=False)

    if not prefix_ids:
        raise ValueError("Empty prefix: Talkie base has no BOS token to score the first token from.")

    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "Tokenizer boundary changed between prefix and prefix+continuation. "
            "Make the probe boundary token-stable, usually by including the intended "
            "leading whitespace in the continuation."
        )

    ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    if int(ids.max()) >= vocab_size:
        raise RuntimeError(
            f"Out-of-range token ID {int(ids.max())} for vocab_size={vocab_size}"
        )

    out = model(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else out[0]  # [1, T, V]

    P = len(prefix_ids)

    # Token at absolute position k is predicted by logits at k-1.
    # Continuation tokens are full_ids[P:].
    pred_logits = logits[:, P - 1 : -1, :]      # [1, continuation_len, V]
    targets = ids[:, P:]                        # [1, continuation_len]

    token_logprobs = torch.log_softmax(pred_logits.float(), dim=-1)
    token_logprobs = token_logprobs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)[0]

    pieces = [tok.decode([int(t)]) for t in targets[0]]
    return [
        {
            "token_id": int(tid),
            "text": piece,
            "logprob": float(lp),
        }
        for tid, piece, lp in zip(targets[0].tolist(), pieces, token_logprobs.cpu())
    ]


def continuation_total_logprob(prefix: str, continuation: str) -> float:
    return sum(x["logprob"] for x in continuation_logprobs(prefix, continuation))


# Smoke test: this must not produce IDs >= 65536.
sample = "The gentleman opened the door and said"
sample_ids = encode_checked(sample)
print("max sample id:", int(sample_ids.max()))

# Example forced-choice scoring.
prefix = "The gentleman opened the door and said"
candidates = [
    " nothing.",
    " hello.",
    " farewell.",
]

for c in candidates:
    print(repr(c), continuation_total_logprob(prefix, c))
```

The dtestnyrr model card reports a 4-bit GPTQ quantization around 7.4–7.6 GB VRAM for short contexts, using GPTQModel, which is why it is the practical 24 GB route. citeturn871090view0 GPTQModel’s own README shows the current `GPTQModel.load(...)` inference API. citeturn497708view3

The caveat: these are **quantized** logits. For many forced-choice experiments they may be fine, but for a final paper-grade measurement I would validate a subset against the BF16 xlr8harder conversion with CPU offload.

---

### Exact BF16 reference path, with CPU offload

This is slower but closer to the reference conversion:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = "xlr8harder/talkie-1930-13b-base-tf"

tok = AutoTokenizer.from_pretrained(
    REPO,
    trust_remote_code=True,
    use_fast=False,
)

model = AutoModelForCausalLM.from_pretrained(
    REPO,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "22GiB", "cpu": "55GiB"},
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

assert tok.vocab_size == 65536
assert model.config.vocab_size == 65536
```

The xlr8harder model supports normal `AutoModelForCausalLM` forward calls and returns logits. Its `TalkieForCausalLM.forward` can return full logits or restrict logits via `logits_to_keep`, which is useful for memory-sensitive scoring. citeturn700649view2

For xlr8harder/BF16 scoring, you can use this more memory-efficient forward:

```python
@torch.inference_mode()
def continuation_logprobs_xlr(prefix: str, continuation: str):
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    full_ids = tok.encode(prefix + continuation, add_special_tokens=False)

    if not prefix_ids:
        raise ValueError("Empty prefix: no BOS token.")
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Tokenizer boundary changed; make the continuation token-boundary-stable.")

    device = next(model.parameters()).device
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    P = len(prefix_ids)
    positions = torch.arange(P - 1, len(full_ids) - 1, device=device)

    out = model(
        input_ids=ids,
        use_cache=False,
        logits_to_keep=positions,
    )

    logits = out.logits.float()       # [1, continuation_len, 65536]
    targets = ids[:, P:]              # [1, continuation_len]

    lp = torch.log_softmax(logits, dim=-1).gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)[0]

    return [
        {
            "token_id": int(tid),
            "text": tok.decode([int(tid)]),
            "logprob": float(val),
        }
        for tid, val in zip(targets[0].tolist(), lp.cpu())
    ]
```

---

### BitsAndBytes 4-bit from the xlr8harder BF16 repo

This gives a single-repo Transformers load, but requires `bitsandbytes` and performs on-the-fly 4-bit quantization rather than using the calibrated GPTQ checkpoint.

```python
# pip install -U bitsandbytes accelerate

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

REPO = "xlr8harder/talkie-1930-13b-base-tf"

tok = AutoTokenizer.from_pretrained(
    REPO,
    trust_remote_code=True,
    use_fast=False,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    REPO,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map={"": 0},
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

assert tok.vocab_size == 65536
assert model.config.vocab_size == 65536
```

Transformers documents `BitsAndBytesConfig(load_in_4bit=True, ...)`, including BF16 compute dtype for 4-bit loading. citeturn735941search0

---

## 5. Official Talkie library and logits/logprobs

The official `talkie-lm/talkie` library is mainly a generation wrapper. The README shows `Talkie(...).generate(...)` and `.stream(...)`, and says BF16 inference needs at least 28 GB VRAM. citeturn484024view2

The underlying official model **does** expose logits through the raw module, but only for the **last position**: `TalkieModel.forward(input_ids)` returns `[B, V]` logits predicting the next token after the provided prefix. citeturn168192view0 I did not find a public `score()` or `logprobs()` helper in the official wrapper; for scoring you call the raw module yourself.

Official-library scoring sketch:

```python
import torch
from talkie import Talkie

t = Talkie("talkie-1930-13b-base", device="cuda")  # needs >=28 GB for normal BF16 path

@torch.inference_mode()
def official_next_logits(text: str):
    ids = t.tokenizer.encode(text)
    ids = torch.tensor([ids], dtype=torch.long, device=t.device)
    logits = t.model(ids)  # [1, 65536], predicts the next token
    return logits
```

For a multi-token continuation with the official library, loop over increasing prefixes:

```python
@torch.inference_mode()
def official_continuation_logprob(prefix: str, continuation: str) -> float:
    prefix_ids = t.tokenizer.encode(prefix)
    full_ids = t.tokenizer.encode(prefix + continuation)

    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Tokenizer boundary changed.")

    ids = torch.tensor([full_ids], dtype=torch.long, device=t.device)

    total = 0.0
    P = len(prefix_ids)

    for pos in range(P, len(full_ids)):
        # logits for token full_ids[pos], conditioned on full_ids[:pos]
        logits = t.model(ids[:, :pos])          # [1, 65536]
        lp = torch.log_softmax(logits.float(), dim=-1)
        total += float(lp[0, ids[0, pos]])

    return total
```

That official path is not the practical 3090 path unless you modify it for offload/quantization. The reason the README asks for ≥28 GB is straightforward: the official HF repo ships a 53.1 GB checkpoint file, and the reference runtime loads BF16 weights onto the device; the xlr8harder BF16 conversion is still about 26.6 GB before activation/runtime overhead. citeturn484024view1turn705257view1 People run it on 24 GB by using a quantized build, BitsAndBytes 4-bit, or CPU offload—not by loading the official BF16 runtime fully on the GPU.

---

## Practical rule

For your current setup, use:

```text
model weights:  dtestnyrr/talkie-1930-13b-base-gptq-int4
tokenizer:      AutoTokenizer.from_pretrained(
                    "xlr8harder/talkie-1930-13b-base-tf",
                    trust_remote_code=True,
                    use_fast=False,
                )
assertions:     tok.vocab_size == 65536
                tok.eos_token_id == 65535
                input_ids.max() < 65536
```

That is the concrete fix for the CUDA assert: the 262k tokenizer is the poison pill.

---
