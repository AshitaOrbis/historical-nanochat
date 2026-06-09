# token_cache_v3 Smoke Report

- cache_dir: `/home/user/historical-nanochat/data/token_cache_v4_balanced_candidate`
- batch: 2
- seq_len: 128
- timestamp: 2026-04-24T11:43:49.361780Z
- overall: **PASS**

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | cache_manifest well-formed (v4 train+val) | PASS | train: dtype=uint16, shards=18926; val: dtype=uint16, shards=2881 |
| 2 | at least one .bin shard (v4 train+val) | PASS | train=18926 bins, val=2881 bins |
| 3 | dataloader yields (B,T) tensors (train split) | PASS | x=(2, 128) y=(2, 128) state_keys=['per_rank', 'shard_idx', 'token_off'] layout=v4_train |
| 4 | tokens in vocab range [0, 32768) | PASS | min=10 max=32759 |
| 5 | at least one special token visible in first batch | PASS | found specials: [32759] |
| 6 | dataloader yields (B,T) tensors (val split) | PASS | x=(2, 128) y=(2, 128) layout=v4_val |
| 7 | tiny model forward on CPU (no NaN/inf) | PASS | logits shape=(2, 128, 32768), mean=-0.0000, std=0.0113 |
