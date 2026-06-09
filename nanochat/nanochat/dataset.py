"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.

Dataset directory resolution (first match wins):
    1) explicit `data_dir` argument passed to a helper
    2) `NANOCHAT_PARQUET_DIR` env var
    3) `<NANOCHAT_BASE_DIR>/base_data` (legacy FineWeb layout, auto-downloadable)

Set `NANOCHAT_PARQUET_DIR` (or pass `--parquet-dir` to scripts that wire it through)
to point training directly at a pre-packaged historical corpus without needing to
mirror the legacy `base_data/` wrapper directory.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from functools import partial
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames


def get_parquet_dir(explicit=None):
    """
    Resolve the directory that holds pretraining parquet shards.

    Precedence: explicit > NANOCHAT_PARQUET_DIR > <base_dir>/base_data.

    The legacy path is still created/returned when neither override is set,
    so the FineWeb auto-download path (see __main__) keeps working.
    """
    if explicit:
        data_dir = os.path.expanduser(explicit)
    elif os.environ.get("NANOCHAT_PARQUET_DIR"):
        data_dir = os.path.expanduser(os.environ["NANOCHAT_PARQUET_DIR"])
    else:
        data_dir = os.path.join(get_base_dir(), "base_data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


# Kept for backwards compatibility with code that imports DATA_DIR directly.
# Callers that want override-aware behavior should use get_parquet_dir() instead.
DATA_DIR = get_parquet_dir()

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = get_parquet_dir(explicit=data_dir)
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1, data_dir=None):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    - data_dir overrides NANOCHAT_PARQUET_DIR / legacy base_data lookup.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(data_dir=data_dir)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index, data_dir=None):
    """ Downloads a single file index, with some backoff. Always writes to the
    FineWeb legacy <base_dir>/base_data by default. Pass `data_dir` explicitly
    to redirect; NANOCHAT_PARQUET_DIR is intentionally ignored here because the
    FineWeb URL only matches the legacy layout."""

    # Construct the local filepath for this file and skip if it already exists
    if data_dir is None:
        data_dir = os.path.join(get_base_dir(), "base_data")
        os.makedirs(data_dir, exist_ok=True)
    filename = index_to_filename(index)
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Target directory (defaults to <base_dir>/base_data). "
                             "Only used by the FineWeb downloader; historical runs should use "
                             "NANOCHAT_PARQUET_DIR to point training at pre-packaged shards.")
    args = parser.parse_args()

    target_dir = args.data_dir or os.path.join(get_base_dir(), "base_data")
    os.makedirs(target_dir, exist_ok=True)

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {target_dir}")
    print()
    worker = partial(download_single_file, data_dir=target_dir)
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(worker, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {target_dir}")
