# Cloud GPU Training Research: nanochat 1-3B from Scratch
## deliberation-2026-05-12

*Prepared May 2026. All prices ±15-20% — GPU spot markets reprice weekly.*

---

## 1. Provider Landscape

Seven providers viable for a PyTorch/CUDA/bfloat16 workload requiring H100 SXM access and checkpoint persistence across multi-hour runs.

| Provider | H100 SXM $/hr (1×) | H100 SXM $/hr (8× node) | Stop-without-terminate | Notes |
|---|---|---|---|---|
| Lambda Labs | $4.29 | $31.92 ($3.99/GPU) | No — terminate only | Karpathy endorses; guaranteed NVLink; zero egress |
| RunPod Secure Cloud | $2.99 | $23.92 ($2.99/GPU) | Yes (pod stop) | NVLink status for 8× unconfirmed; ~25% cheaper |
| RunPod Community Cloud | ~$2.69 | ~$21.52 | Yes | Spot-like; preemption risk; aggressive checkpointing required |
| Hyperbolic | $1.50-$3.20 | Not offered | Unknown | Wide price range; verify at hyperbolic.ai/marketplace before booking |
| Vast.ai | ~$1.73 (PCIe) | Not typical | Yes | H100 on Vast.ai is predominantly PCIe; ~20-40% DDP overhead at 8× |
| Together AI | Not offered | $23.92 ($2.99/GPU, min 8×) | Unknown | Cluster-reservation model; not optimized for short self-serve runs |
| Modal | ~$3.95 | Scale-out only | N/A (functions) | 24hr function timeout is a hard blocker for runs >24hrs |
| Spheron | $1.33 | Not offered | Unknown | Cheapest; insufficient ops documentation |

**NVLink vs PCIe matters for multi-GPU DDP.** H100 SXM with NVLink = 900 GB/s. H100 PCIe = ~64 GB/s. nanochat uses torchrun for distributed training; NVLink cuts all-reduce overhead from ~15% of step time to <2% at 8×. For 1×GPU, irrelevant.

---

## 2. What \$200 Buys

**Verified reference run (nanochat Discussion #481, Jan 2026):**
- d24 speedrun: 1.38B params, ~8.8B tokens, 3.04 hrs on 8×H100 at ~$24/hr → **~$73**

The historical-nanochat corpus (~16-20 GB tokenized) is comparable in size to FineWeb-edu at this scale. d24 was Karpathy's reference target.

At $200:

| Option | Config | Wall-clock | Est. cost | Budget remaining |
|---|---|---|---|---|
| Lambda 8×H100 SXM | d24 speedrun | ~3.0 hrs | ~$96 | ~$104 (one full retry) |
| RunPod Secure Cloud 8×H100 | d24 speedrun | ~3.0 hrs | ~$72 | ~$128 |
| RunPod or Lambda 1×H100 | d24 speedrun | ~24 hrs | ~$72-$103 | ~$97-$128 |
| Hyperbolic 1×H100 | d24 speedrun | ~24 hrs | ~$36-$77 | ~$123-$164 |

**\$200 covers one full d24 run on any viable provider with a meaningful retry buffer.**

d24 at 8.8B tokens is slightly undertrained vs Chinchilla-optimal (~14.5B for 1.38B params at 10.5×). Karpathy's own choice — still achieves CORE ≈ 0.258, beating GPT-2.

---

## 3. What \$2,000 Buys

FLOPs scale as ~6 × params × tokens. Extrapolating from d24:

| Target | Approx params | Tokens | Est. 8×H100 hrs | Lambda 8× cost | RunPod 8× cost |
|---|---|---|---|---|---|
| d24 (reference) | 1.38B | 8.8B | 3.0 | ~$96 | ~$72 |
| d24 × 2 epochs | 1.38B | 17.6B | ~6.0 | ~$191 | ~$144 |
| ~2B params (new depth) | ~2B | ~21B @ 10.5× | ~9-13 | ~$287-$415 | ~$215-$311 |
| ~3B params (new depth) | ~3B | ~31.5B @ 10.5× | ~18-25 | ~$575-$799 | ~$431-$599 |
| ~3B Chinchilla-optimal | ~3B | ~60B @ 20× | ~35-50 | ~$1,118-$1,597 | ~$838-$1,198 |

*Rows beyond d24 are FLOPs extrapolations; ±30% accuracy.*

**Corpus constraint:** A 3B compute-optimal model needs ~60B tokens. The 16-20B governed corpus supports either (a) a 3B model severely undertrained — useful if the research question is about the corpus, or (b) a ~800M-1B model at full Chinchilla-optimal. d24 is the sweet spot where corpus size and compute-optimal training converge.

At $2,000: run d24 (~$96), then ~$1,900 for ablation variants (different depths, extended training, or a larger undertrained 2-3B run).

---

## 4. Provider Recommendation

**Primary: Lambda Labs**

1. **Karpathy explicitly endorses Lambda** in the nanochat README: *"I use and like Lambda"* and *"Thank you Lambda for the compute."*
2. **Guaranteed NVLink SXM topology.** Lambda's 8×H100 nodes are H100 SXM with NVLink (verified). Exact hardware nanochat was developed on.
3. **Zero egress fees.** Critical for pulling 16-20 GB corpus in and pushing checkpoints out.
4. **On-demand, no auction.** No preemption risk during multi-hour runs.

**Critical operational hazard:** Lambda has no stop/pause — only terminate. Terminate wipes the local SSD.

**Mandatory mitigation before launch:**
- Attach Lambda's persistent filesystem add-on (~$0.20/GiB/mo) BEFORE instance launch. Cannot be attached after.
- Or: configure nanochat's `checkpoint_manager.py` to sync to S3/rclone after every N steps.

**Backup: RunPod Secure Cloud**

When Lambda has no 8×H100 availability (common — SXM nodes sell out quickly):
- H100 SXM Secure Cloud: $2.99/GPU (~25% cheaper than Lambda)
- Pod stop preserves `/workspace` container disk
- NVLink for RunPod 8×GPU pods is not definitively confirmed in public documentation; if low GPU utilization on DDP, fall back to 1×GPU

**Avoid for this workload:**
- **Modal:** 24-hour function timeout
- **Vast.ai:** H100 is predominantly PCIe; DDP overhead
- **Spheron:** Insufficient operational documentation

---

## 5. Practical Setup Steps

### Lambda Labs

```bash
# STEP 0: Before launching instance
# Dashboard → Storage → Filesystems → Create filesystem
# Size: ≥50 GB  (~$10/mo)
# Attach to instance at launch, mount at: /home/ubuntu/checkpoints

# STEP 1: Launch instance
# Dashboard → Instances → Launch → gpu_8x_h100_sxm5
# Attach persistent filesystem created in Step 0

# STEP 2: Transfer corpus (from local or S3)
rsync -avz --progress --compress ./data/tokenized/ \
  ubuntu@<LAMBDA_IP>:/home/ubuntu/data/tokenized/
# ~16-20 GB; at 1 Gbps ≈ 2-3 min

# STEP 3: Setup nanochat
git clone https://github.com/karpathy/nanochat.git
cd nanochat
pip install -r requirements.txt

# STEP 4: Run speedrun (d24, matches corpus size)
tmux new-session -s training
bash runs/speedrun.sh

# STEP 5: Verify checkpoints are going to persistent volume
# Redirect to: /home/ubuntu/checkpoints/
```

### RunPod Secure Cloud (fallback)

```bash
# STEP 0: Create network volume before pod launch
# Console → Storage → Network Volumes → Create (≥50 GB)
# Attach to pod at creation; mount at /workspace

# STEP 1: Launch H100 SXM pod
# Secure Cloud tab → filter by H100 SXM → 8× or 1× GPU
# Attach network volume

# STEP 2-5: Same as Lambda
# Pod stop (not terminate) preserves /workspace between sessions
```

### Checkpoint Resume

```bash
torchrun --nproc_per_node=8 train.py --resume /path/to/checkpoint_dir
```

---

## 6. Failure Modes

| Failure | Likelihood | Mitigation |
|---|---|---|
| **Lambda 8×H100 unavailable** | Medium-high. SXM nodes frequently sold out. | Monitor dashboard; have RunPod Secure Cloud account ready as fallback |
| **Lambda terminate wipes checkpoint** | Certain if persistent filesystem not pre-attached | Attach persistent filesystem BEFORE instance launch |
| **RunPod multi-GPU is PCIe, not NVLink** | Unknown | Monitor GPU utilization. If <80% on DDP, switch to 1×GPU |
| **Corpus upload bottleneck** | Low | Pre-stage on S3 and pull from within instance |
| **OOM during bfloat16 training** | Low for d24 on H100 SXM 80GB | Use `runs/speedrun.sh` exactly as-is |
| **Training divergence / NaN loss** | Low with default config | No optimizer tuning until baseline run completes |
| **Cost overrun** | Medium without monitoring | Set spend alerts. d24 should finish in ~3 hrs; set 6-hr wall-clock alarm |
| **Spot preemption mid-run** | High over 24hrs on spot instances | On-demand for any run >6 hrs |

---

## 7. Concrete First-Run Recommendation

**Lambda Labs · 8×H100 SXM on-demand · d24 depth · one speedrun**

| Parameter | Value |
|---|---|
| Provider | Lambda Labs |
| Instance | 8×H100 SXM on-demand (`gpu_8x_h100_sxm5`) |
| Depth | d24 (1.38B params) |
| Tokens | ~8.8B (matches corpus; use nanochat's default speedrun config) |
| Wall-clock | ~3.0 hrs |
| Estimated cost | ~$96 ($31.92/hr × 3 hrs) |
| Budget consumed | ~$96 of $200 minimum; ~$104 buffer for one full retry |
| Expected output | CORE ≈ 0.258; matches nanochat's published benchmark |

**Pre-launch checklist:**
- [ ] Lambda persistent filesystem created (≥50 GB) and attached before launch
- [ ] Corpus uploaded to instance before starting training
- [ ] tmux session active (SSH disconnect must not kill training)
- [ ] Checkpoint path verified to write to persistent filesystem, not local SSD
- [ ] Spend alert set at $150 in Lambda dashboard
- [ ] `runs/speedrun.sh` reviewed — confirm data path matches corpus location

**If Lambda 8×H100 unavailable:** RunPod Secure Cloud, 8×H100 SXM pod, same d24 depth, network volume attached, ~$72.

**If only 1×GPU available:** Lambda or RunPod 1×H100, same d24 depth, ~24 hrs wall-clock, ~$72-$103.

---

## Sources

| # | Source | URL | Key Contribution |
|---|--------|-----|------------------|
| 1 | nanochat Discussion #481 | https://github.com/karpathy/nanochat/discussions/481 | Official d24 cost anchor: 1.38B params, 8.8B tokens, 3.04 hrs, ~$73 |
| 2 | nanochat README | https://github.com/karpathy/nanochat/blob/master/README.md | Karpathy endorses Lambda; 1×GPU = 8× slower but identical |
| 3 | nanochat Discussion #420 | https://github.com/karpathy/nanochat/discussions/420 | Miniseries d10-d20 training times on 8×H100 |
| 4 | lambda.ai/instances | https://lambda.ai/instances | Current pricing: 1×H100 $4.29/hr, 8×H100 $31.92/hr; zero egress |
| 5 | Alpha One Index (RunPod) | https://alphaoneindex.com/ai-infra/providers/runpod/ | RunPod H100 SXM Secure Cloud $2.99/GPU; stop/restart behavior |
| 6 | DeployBase (Vast.ai) | https://deploybase.ai/articles/vast-ai-gpu-cloud-pricing-complete-guide-vs-hr-for-every-gpu | Vast.ai H100 PCIe ~$1.73/hr |
| 7 | Hyperbolic pricing | https://www.hyperbolic.ai/blog/gpu-cloud-pricing | H100 from $1.50/hr; weekly fluctuation |
| 8 | Modal pricing | https://modal.com/pricing | H100 ~$3.95/hr; 24hr function timeout |
| 9 | Together AI | https://www.together.ai/gpu-clusters | 8× H100 $2.99/GPU = $23.92/hr; min 8× cluster |
| 10 | Spheron | https://www.spheron.network/gpu-rental/h100/ | H100 from $1.33/hr |
| 11 | RunPod docs | https://docs.runpod.io/pods/manage-pods | Pod stop vs terminate behavior |
| 12 | Lambda docs | https://docs.lambda.ai/public-cloud/on-demand/getting-started/ | Persistent filesystem must be pre-attached; no stop/pause |

## Confidence Assessment

- **High:** Lambda pricing ($31.92/hr 8×H100), d24 official run specs, Karpathy Lambda endorsement, Lambda terminate-wipes-disk risk, RunPod pod-stop behavior
- **Medium:** RunPod NVLink for 8×GPU pods, Hyperbolic pricing stability, extrapolated FLOPs scaling for d28+ depths
- **Needs verification:** Lambda 8×H100 SXM availability at run time; exact nanochat `--depth` values for 2-3B range
