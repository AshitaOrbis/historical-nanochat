# Historical Nanochat

Training vintage LLMs on pre-1913/1914 historical texts using the nanochat framework.

## Project Structure

- `data/` - Corpus download and processing scripts
- `nanochat/` - Training framework (submodule/fork)
- `tokenizer/` - Custom 32K vocab tokenizer trained on historical corpus

## Training Configuration

- **Model**: 125M params (d12 - 12 layers, 768 dim)
- **Tokenizer**: Custom 32K vocab BPE trained on historical corpus
- **Data**: ~16B tokens from pre-1913 sources (American Stories, Gutenberg, Chronicling America, etc.)
- **Framework**: nanochat with Muon optimizer

## Data Locations

All data is on native Linux ext4 (copied from Windows NTFS, Apr 2026):

- **Tokenizer**: `tokenizer/` (in repo — tokenizer.json + vocab.txt)
- **Shards (training)**: `~/historical-nanochat/data/shards/` (322 parquet files, 36G)
- **Shards (small)**: `~/historical-nanochat/data/shards-small/` (3,469 parquet files, 65G)
- **Deduped corpus**: `~/historical-nanochat/data/deduped/` (398G — 10 source dirs)
- **Download/process scripts**: `~/historical-nanochat/data/{download,process,stats}/`
- **Checkpoints**: None surviving (lost during WSL→native Linux migration)

### Windows drive copies (can be deleted to reclaim ~499G on NTFS)

- `/mnt/c/Users/<user>/D-drive-data/historical-nanochat-deduped/`
- `/mnt/c/Users/<user>/D-drive-data/historical-nanochat-shards/`
- `/mnt/c/Users/<user>/D-drive-data/historical-nanochat-shards-small/`

### Deleted Data (Feb 2026 cleanup)

- `data/raw/` (419G) — re-downloadable from HuggingFace/public APIs via `data/download/` scripts
- `.venv/` (7.3G) — regenerable with `python -m venv .venv && pip install -e .`

## Known Issues

### GPU Memory Pressure on RTX 3090 (Critical)

**Problem**: Training with `device_batch_size=4` and `max_seq_len=2048` uses ~98% of 24GB VRAM, leaving no headroom. This causes sporadic GPU memory allocation failures (`dxgkio_make_resident: Ioctl failed: -12`) that stall training for 30-60+ minutes per step.

**Symptoms**:
- Step times vary wildly (6s to 3800s)
- `dxgkio_make_resident: Ioctl failed: -12` errors in dmesg (ENOMEM)
- GPU shows 100% utilization but throughput drops to <100 tok/sec
- Progressive degradation as memory fragments

**Root Cause**:
- RTX 3090 has 24GB VRAM but needs headroom for:
  - CUDA memory allocator overhead
  - Temporary buffers during forward/backward passes
  - Memory fragmentation over time
- At 98% usage, allocation failures cause massive stalls

**Solution**: Use `device_batch_size=2` instead of 4. This:
- Reduces VRAM usage from ~24GB (98%) to ~8GB (32%)
- Doubles gradient accumulation steps (64 instead of 32)
- Maintains same total batch size (262144 tokens)
- Gives consistent ~7.1s steps, ~36K tok/sec

### WSL Filesystem Performance (Secondary)

**Recommendation**: Store training shards on native Linux filesystem (ext4 inside WSL) rather than `/mnt/d/` (Windows NTFS via 9P protocol) for better I/O performance. While not the primary cause of the severe slowdowns, native Linux FS has lower latency.

### torch.compile on WSL

`torch.compile` fails with nvcc permission errors on WSL. Use `TORCH_COMPILE_DISABLE=1` environment variable.

## Training Commands

```bash
# Recreate venv first (deleted during Feb 2026 cleanup)
# (run from the repository root)
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Resume training from checkpoint (use device_batch_size=2 on 3090!)
cd nanochat
source ../.venv/bin/activate
PYTHONUNBUFFERED=1 WANDB_MODE=offline TORCH_COMPILE_DISABLE=1 python -m scripts.base_train \
  --depth=12 --num_iterations=15250 --device_batch_size=2 --max_seq_len=2048 \
  --total_batch_size=262144 --eval_every=1000 --save_every=1500 \
  --run=historical_1913 --model_tag=d12_v1 --resume_from_step=3000
```

## Checkpoints

| Step | Val BPB | Notes |
|------|---------|-------|
| 3000 | 3.73 | First stable checkpoint |
