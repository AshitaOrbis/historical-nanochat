from collections import deque

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

def _require_every_rank_owns_rows(all_parquet_paths, world_size):
    """Refuse a raw split in which some DDP rank would never receive a document.

    Row groups are dealt out by index: rank r reads groups r, r+W, r+2W, ... of
    every file in its split. Two parquet files are enough for the train/val split
    to exist, but not for every rank to own data in it. With one row group per
    file and W=2, rank 1 never enters a file's row-group loop, the epoch loop
    restarts forever, and it neither yields nor raises while its peers block in
    the next collective (bq-2039).

    Both splits are checked whichever one is being built, so a validation split
    too small for the world size fails at startup instead of at the first
    evaluation. Every rank is checked, not only this one, so all ranks refuse
    together rather than one refusing while the others wait on it.
    """
    splits = {"train": all_parquet_paths[:-1], "val": all_parquet_paths[-1:]}
    for split_name, paths in splits.items():
        owners = set()
        for path in paths:
            metadata = pq.read_metadata(path)
            for rg_idx in range(metadata.num_row_groups):
                if metadata.row_group(rg_idx).num_rows > 0:
                    owners.add(rg_idx % world_size)
        starved = sorted(set(range(world_size)) - owners)
        if starved:
            raise RuntimeError(
                f"raw loader {split_name} split gives no rows to DDP rank(s) {starved} "
                f"of world_size={world_size}: rank r reads row groups r, r+{world_size}, "
                "... of each file, and no file in this split has a non-empty row group "
                "at any of those indices. Repackage with smaller row groups or more "
                "shards, or train on fewer ranks."
            )


def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None, parquet_dir=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")
    all_parquet_paths = list_parquet_files(data_dir=parquet_dir)
    if not all_parquet_paths:
        raise RuntimeError(
            f"No dataset parquet files found in {parquet_dir or '<NANOCHAT_PARQUET_DIR or <base_dir>/base_data>'}. "
            "Either run `python -m nanochat.dataset` to download FineWeb, or set "
            "NANOCHAT_PARQUET_DIR to your shard directory."
        )
    if len(all_parquet_paths) < 2:
        raise RuntimeError(
            "Legacy raw train/validation splitting requires at least two parquet "
            "shards. Supply a separate validation corpus or deterministically split "
            "the source before training; one shard must never serve both splits."
        )
    parquet_paths = (
        all_parquet_paths[:-1] if split == "train" else all_parquet_paths[-1:]
    )
    if resume_state_dict is not None:
        if not isinstance(resume_state_dict, dict):
            raise RuntimeError("raw loader resume state must be an object")
        resume_pq_idx = resume_state_dict.get("pq_idx")
        if (
            not isinstance(resume_pq_idx, int)
            or isinstance(resume_pq_idx, bool)
            or not 0 <= resume_pq_idx < len(parquet_paths)
        ):
            raise RuntimeError(
                f"raw loader resume pq_idx={resume_pq_idx!r} is outside the "
                f"split parquet domain 0..{len(parquet_paths) - 1}"
            )

    # Resolve split cardinality before returning the generator. This prevents an
    # empty training list from becoming an infinite, non-yielding iterator.
    _ddp, ddp_rank, _ddp_local_rank, ddp_world_size = get_dist_info()
    _require_every_rank_owns_rows(all_parquet_paths, ddp_world_size)
    def document_batches():
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                # I know this state resumption is a little bit tricky and a little bit hacky... sigh.
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size # in units of ddp_world_size
                    base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None # set to None as we only want to do this a single time
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist() # each batch is a parquet group, e.g. 1024 rows
                    # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size # advance to the next row group (in DDP)
                pq_idx += 1 # advance to the next parquet file
            first_pass = False
    def iterator():
        batches = document_batches()
        needed_tokens = B * T + 1
        tokenizer = get_tokenizer()
        bos_token = tokenizer.get_bos_token_id()
        token_buffer = deque()
        while True:
            while len(token_buffer) < needed_tokens:
                doc_batch, (pq_idx, rg_idx) = next(batches)
                token_lists = tokenizer.encode(
                    doc_batch, prepend=bos_token, num_threads=tokenizer_threads
                )
                for tokens in token_lists:
                    token_buffer.extend(tokens)
            tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
            use_cuda_optimizations = device == "cuda"
            scratch = torch.tensor(
                tokens, dtype=torch.long, pin_memory=use_cuda_optimizations
            )
            inputs_cpu = scratch[:-1]
            targets_cpu = scratch[1:]
            inputs = inputs_cpu.view(B, T).to(
                device=device, non_blocking=use_cuda_optimizations
            )
            targets = targets_cpu.view(B, T).to(
                device=device, non_blocking=use_cuda_optimizations
            )
            state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx}
            yield inputs, targets, state_dict

    return iterator()

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
