#!/usr/bin/env python3
"""Run the v1 615M characterology pilot: Pre-1913 nanochat vs a modern base anchor
(gpt2) across Family F (closure, with falsification battery) + Family B (minimal
pairs) + a little Family A texture. Writes results.json + report.md.

Usage: NANOCHAT_BASE_DIR=$PWD .venv/bin/python probes/run_pilot.py
"""
import os, sys, json
from statistics import mean

ROOT = "/home/user/claudeworkspace/research/historical-nanochat"
sys.path.insert(0, os.path.join(ROOT, "probes"))
from harness import NanochatModel, HFModel, GPTQModel, rank_candidates
import probe_sets as P

DATE = "2026-06-09"
OUT_DIR = os.path.join(ROOT, "runs", f"probe_pilot_{DATE}_3anchor")
os.makedirs(OUT_DIR, exist_ok=True)


def cluster_means(ranked_rows, cand_meta):
    """ranked_rows: [(label, per_byte, sum, ntok)]. cand_meta: label->cluster."""
    by_cluster = {}
    for label, per_byte, _s, _n in ranked_rows:
        by_cluster.setdefault(cand_meta[label], []).append(per_byte)
    return {c: mean(v) for c, v in by_cluster.items()}


def run_family_f(models):
    out = {}
    for vkey, vdef in P.FAMILY_F.items():
        prefix = vdef["prefix"]
        cand_meta = {label: cl for label, cl, _t in vdef["candidates"]}
        cands = {label: text for label, _cl, text in vdef["candidates"]}
        out[vkey] = {"prefix": prefix, "models": {}}
        for m in models:
            ranked = rank_candidates(m, prefix, cands)
            cm = cluster_means(ranked, cand_meta)
            pre = cm.get("pre")
            post = cm.get("post")
            modern = cm.get("modern")
            out[vkey]["models"][m.name] = {
                "label": m.label,
                "order": [r[0] for r in ranked],
                "per_byte": {r[0]: round(r[1], 4) for r in ranked},
                "cluster_means": {k: round(v, 4) for k, v in cm.items()},
                "pre_minus_post": round(pre - post, 4) if pre is not None and post is not None else None,
                "pre_minus_modern": round(pre - modern, 4) if pre is not None and modern is not None else None,
            }
    return out


def run_family_b(models):
    out = {}
    for pkey, pdef in P.FAMILY_B.items():
        prefix = pdef["prefix"]
        cands = {label: text for label, _cl, text in pdef["candidates"]}
        pos, neg = pdef["contrast"]
        out[pkey] = {"prefix": prefix, "contrast": f"{pos} - {neg}", "models": {}}
        for m in models:
            ranked = rank_candidates(m, prefix, cands)
            pb = {r[0]: r[1] for r in ranked}
            contrast = pb[pos] - pb[neg] if pos in pb and neg in pb else None
            out[pkey]["models"][m.name] = {
                "order": [r[0] for r in ranked],
                "per_byte": {r[0]: round(r[1], 4) for r in ranked},
                "contrast": round(contrast, 4) if contrast is not None else None,
            }
    return out


def run_family_a(models):
    out = {}
    gen_models = [m for m in models if getattr(m, "can_generate", True)]
    for stem in P.FAMILY_A_STEMS:
        out[stem] = {m.name: m.generate(stem, n=50, temperature=0.8).replace("\n", " ").strip()
                     for m in gen_models}
    return out


def fmt(x):
    return f"{x:+.4f}" if isinstance(x, float) else str(x)


def write_report(results, models):
    mnames = [m.name for m in models]
    L = []
    L.append(f"# Computational Historical Characterology — v1 615M Pilot ({DATE})\n")
    L.append("Pre-1913 nanochat 615M vs a modern base anchor (gpt2), length-normalized "
             "log-prob per byte. Per probe-design.md: compare WITHIN-model preference "
             "orderings, then orderings across models. One small modern anchor + one "
             "pre-1913 model — a **two-point pilot**, not the three-anchor result.\n")

    # Family F headline
    L.append("## Family F — closure / tragic emplotment (the core family)\n")
    L.append("Predicted: a pre-1913 habitus prefers the **pre** cluster (providence/duty) over "
             "**post** (absurd/anti-progress) and over **modern** (therapeutic). "
             "`pre_minus_post>0` and `pre_minus_modern>0` are the pre-WWI signature; a modern "
             "model should show `pre_minus_modern<0`.\n")
    L.append("| variant | model | pre−post | pre−modern | top-3 order |")
    L.append("|---|---|---|---|---|")
    robust = {m.name: {"pre_post_pos": 0, "pre_modern_pos": 0, "n": 0} for m in models}
    for vkey, vdef in results["family_f"].items():
        for mn in mnames:
            d = vdef["models"][mn]
            pp, pm = d["pre_minus_post"], d["pre_minus_modern"]
            robust[mn]["n"] += 1
            if pp is not None and pp > 0: robust[mn]["pre_post_pos"] += 1
            if pm is not None and pm > 0: robust[mn]["pre_modern_pos"] += 1
            L.append(f"| {vkey} | {mn} | {fmt(pp)} | {fmt(pm)} | {' > '.join(d['order'][:3])} |")
    L.append("")
    L.append("**Robustness across the 5 Family-F variants (incl. falsifiers):**")
    for mn in mnames:
        r = robust[mn]
        L.append(f"- `{mn}`: pre>post in {r['pre_post_pos']}/{r['n']} variants; "
                 f"pre>modern in {r['pre_modern_pos']}/{r['n']}.")
    L.append("")

    # must-see full ordering
    ms = results["family_f"]["must_see"]["models"]
    L.append("### The must-see probe, full ordering\n")
    L.append(f"> *{results['family_f']['must_see']['prefix']}*\n")
    for mn in mnames:
        d = ms[mn]
        L.append(f"- **{d['label']}**: " + " > ".join(d["order"]))
    L.append("")

    # Family B
    L.append("## Family B — minimal-pair posture contrasts\n")
    L.append("Each value = per-byte log-prob of the pre-posture candidate minus the "
             "post/modern contrast candidate. Pre-WWI predicted > 0.\n")
    L.append("| probe | contrast | " + " | ".join(mnames) + " |")
    L.append("|---|---|" + "|".join(["---"] * len(mnames)) + "|")
    for pkey, pdef in results["family_b"].items():
        row = [pkey, pdef["contrast"]]
        for mn in mnames:
            row.append(fmt(pdef["models"][mn]["contrast"]))
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # Family A texture
    L.append("## Family A — free-generation texture (illustrative, not scored)\n")
    for stem, gens in results["family_a"].items():
        L.append(f"**{stem!r}**")
        for mn in mnames:
            if mn in gens:  # generation-disabled anchors (talkie/GPTQ) are absent
                L.append(f"- `{mn}`: {gens[mn][:240]!r}")
        L.append("")

    L.append("## Caveats (held from probe-design.md)\n")
    L.append("- **Two-point, not three-anchor.** The headline characterology result needs "
             "Talkie-1930 (post-WWI) as the third anchor; gpt2 is a stand-in modern base, "
             "different scale/architecture/tokenizer. Comparisons here are ordinal only.\n"
             "- **Scale + tokenizer confounds** are real (615M vs 124M, different BPE). "
             "Per-byte normalization + ordinal comparison mitigate but don't eliminate them.\n"
             "- **gpt2 is web-text 2019**, not a matched-pipeline modern nanochat; treat as a "
             "rough modern-text anchor, and note it is not instruction-tuned (good — base posture).\n"
             "- A single family is not a finding; the design requires **cross-family convergence**. "
             "Family A/C/D/E/G/H are not yet run.\n")
    return "\n".join(L)


def main():
    print("loading Pre-1913 nanochat 615M ...")
    nano = NanochatModel()
    print("loading Talkie-1930 13B (post-WWI anchor): dtestnyrr GPTQ model + xlr8harder tokenizer ...")
    talkie = GPTQModel(
        "dtestnyrr/talkie-1930-13b-base-gptq-int4",
        "Talkie-1930 13B (pre-1931 corpus, post-WWI anchor)",
        tokenizer_id="xlr8harder/talkie-1930-13b-base-tf",  # correct 65536 TalkieTokenizer
    )
    print("loading modern anchor gpt2 ...")
    gpt2 = HFModel("gpt2")
    # order: pre-1913 (615M) -> 1930 (13B) -> modern (124M). Scale is NON-monotonic
    # with era, which helps separate corpus/era effects from raw scale effects.
    models = [nano, talkie, gpt2]

    print("running Family F (closure + falsifiers) ...")
    family_f = run_family_f(models)
    print("running Family B (minimal pairs) ...")
    family_b = run_family_b(models)
    print("running Family A (texture) ...")
    family_a = run_family_a(models)

    results = {"date": DATE, "models": {m.name: m.label for m in models},
               "family_f": family_f, "family_b": family_b, "family_a": family_a}

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    report = write_report(results, models)
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write(report)
    print(f"\nwrote {OUT_DIR}/results.json and report.md")
    print("\n" + report[report.index("## Family F"):report.index("### The must-see")])


if __name__ == "__main__":
    main()
