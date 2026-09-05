"""
Download historical newspapers from Chronicling America (Library of Congress).

Covers newspapers from 1756-1963.
Uses the loc.gov API (updated August 2025) for metadata and OCR downloads.

API Documentation: https://libraryofcongress.github.io/data-exploration/
Rate limits: 20 requests per 10 seconds (crawl), 20 requests per minute (burst)
"""
import os
import json
import argparse
import requests
from typing import Optional, Dict, Any, List
from tqdm import tqdm
from datetime import datetime
import time
import gzip

# New loc.gov API base URL (as of August 2025)
LOC_API_BASE = "https://www.loc.gov"

# Collection endpoint for Chronicling America
CHRONAM_COLLECTION = "/collections/chronicling-america/"


def search_pages(
    date_start: str,
    date_end: str,
    state: Optional[str] = None,
    page: int = 1,
    rows: int = 100,
) -> Dict[str, Any]:
    """
    Search for newspaper pages within a date range using the loc.gov API.

    Args:
        date_start: Start date in YYYY or YYYY-MM-DD format
        date_end: End date in YYYY or YYYY-MM-DD format
        state: Optional state filter (lowercase, e.g., 'california')
        page: Page number for pagination
        rows: Number of results per page (max typically 100)
    """
    # Build query parameters for new loc.gov API
    params = {
        "dl": "page",  # Display level: page
        "start_date": f"{date_start}-01-01" if len(str(date_start)) == 4 else date_start,
        "end_date": f"{date_end}-12-31" if len(str(date_end)) == 4 else date_end,
        "fo": "json",  # Format: JSON
        "c": rows,  # Count per page
        "sp": page,  # Starting page
    }

    if state:
        params["location_state"] = state.lower()

    url = f"{LOC_API_BASE}{CHRONAM_COLLECTION}"
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def get_page_ocr(lccn: str, date: str, edition: int, sequence: int) -> Optional[str]:
    """
    Get OCR text for a specific newspaper page.

    Args:
        lccn: Library of Congress Control Number
        date: Date in YYYY-MM-DD format
        edition: Edition number
        sequence: Page sequence number
    """
    url = f"{LOC_API_BASE}/lccn/{lccn}/{date}/ed-{edition}/seq-{sequence}/ocr.txt"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.text
        return None
    except Exception:
        return None


def download_chronicling_america(
    cutoff: str = "1913",
    output_dir: str = "data/raw/chronicling_america",
    max_pages: Optional[int] = None,
    states: Optional[List[str]] = None,
    min_year: int = 1800,
) -> Dict[str, Any]:
    """
    Download newspaper OCR from Chronicling America.

    Args:
        cutoff: Temporal cutoff year
        output_dir: Output directory
        max_pages: Maximum pages to download
        states: List of states to filter by
        min_year: Minimum year to include
    """
    from data.download.gutenberg_download import CUTOFF_CONFIGS

    if cutoff not in CUTOFF_CONFIGS:
        raise ValueError(f"Unknown cutoff: {cutoff}")

    cutoff_year = CUTOFF_CONFIGS[cutoff]["year"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Chronicling America Historical Newspapers")
    print(f"Date range: {min_year} to {cutoff_year}")
    print(f"Output directory: {output_dir}")

    stats = {
        "records_requested": 0,
        "records_fetched": 0,
        "records_written": 0,
        "total_processed": 0,
        "accepted": 0,
        "rejected": 0,
        "total_chars": 0,
        "years_distribution": {},
        "states_distribution": {},
    }

    output_file = os.path.join(output_dir, f"newspapers_{cutoff}.jsonl")

    # Rate limiting - loc.gov API: 20 requests per 10 seconds
    # Being conservative to avoid 503 errors
    request_delay = 1.0  # seconds between requests
    max_retries = 3

    with open(output_file, 'w', encoding='utf-8') as f:
        # Search for pages in date range
        print(f"\nSearching for pages from {min_year} to {cutoff_year}...")
        print("(Using loc.gov API - rate limited to ~20 req/10sec)")

        page_num = 1
        total_results = None

        while True:
            if max_pages and stats["accepted"] >= max_pages:
                break

            try:
                # Retry logic for search
                results = None
                for retry in range(max_retries):
                    try:
                        time.sleep(request_delay * (retry + 1))  # Exponential backoff
                        results = search_pages(
                            date_start=str(min_year),
                            date_end=str(cutoff_year),
                            page=page_num,
                            rows=25,  # Smaller batches for rate limiting
                        )
                        break
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 503 and retry < max_retries - 1:
                            print(f"  Rate limited (503), waiting {(retry + 2) * 5}s before retry...")
                            time.sleep((retry + 2) * 5)
                        else:
                            raise

                if results is None:
                    raise RuntimeError("Chronicling America search returned no response")

                # New loc.gov API uses 'pagination' and 'results' fields
                if total_results is None:
                    pagination = results.get("pagination", {})
                    total_results = pagination.get("total", 0)
                    if total_results == 0:
                        # Try alternate field names
                        total_results = results.get("count", 0)
                    print(f"Found {total_results:,} total pages")

                # Results can be in 'results' or 'content' depending on endpoint
                items = results.get("results", results.get("content", []))
                if not items:
                    print(f"No more items found on page {page_num}")
                    break

                for item in tqdm(items, desc=f"Page {page_num}", leave=False):
                    if max_pages and stats["accepted"] >= max_pages:
                        break

                    stats["records_requested"] += 1
                    stats["total_processed"] += 1

                    # Extract metadata - new API uses different field names
                    # Date might be in 'date', 'dates', or nested in 'item'
                    date_str = ""
                    if isinstance(item.get("date"), str):
                        date_str = item.get("date", "")
                    elif isinstance(item.get("date"), list) and item.get("date"):
                        date_str = item["date"][0]
                    elif item.get("dates"):
                        dates = item.get("dates", [])
                        date_str = dates[0] if dates else ""

                    try:
                        year = int(date_str[:4]) if date_str and len(date_str) >= 4 else None
                    except ValueError:
                        year = None

                    if year is None or year > cutoff_year:
                        stats["rejected"] += 1
                        continue

                    # Get OCR URL - construct from LCCN, date, and page number
                    # Format: chroniclingamerica.loc.gov/lccn/{lccn}/{date}/ed-{ed}/seq-{seq}/ocr.txt
                    ocr_url = None

                    # Extract LCCN and page number from API response
                    lccn_list = item.get("number_lccn", [])
                    lccn = lccn_list[0] if lccn_list else None

                    page_list = item.get("number_page", [])
                    page_num_str = page_list[0] if page_list else "0000000001"
                    try:
                        seq_num = int(page_num_str.lstrip("0") or "1")
                    except ValueError:
                        seq_num = 1

                    if lccn and date_str:
                        # Construct the Chronicling America OCR URL
                        # Most newspapers use ed-1 (edition 1)
                        ocr_url = f"https://chroniclingamerica.loc.gov/lccn/{lccn}/{date_str}/ed-1/seq-{seq_num}/ocr.txt"

                    # Fallback: try to extract from item URL if it contains /lccn/
                    if not ocr_url and item.get("url"):
                        item_url = item.get("url", "")
                        if "/lccn/" in item_url:
                            ocr_url = item_url.rstrip("/") + "/ocr.txt"

                    if not ocr_url:
                        stats["rejected"] += 1
                        continue

                    # Download OCR
                    try:
                        time.sleep(request_delay)
                        response = requests.get(ocr_url, timeout=30)
                        if response.status_code != 200:
                            stats["rejected"] += 1
                            continue

                        stats["records_fetched"] += 1
                        text = response.text.strip()

                        if len(text) < 100:
                            stats["rejected"] += 1
                            continue

                        stats["accepted"] += 1
                        stats["records_written"] += 1
                        stats["total_chars"] += len(text)

                        # Record statistics
                        decade = (year // 10) * 10
                        stats["years_distribution"][decade] = stats["years_distribution"].get(decade, 0) + 1

                        # Extract state from various possible fields
                        state = "Unknown"
                        if item.get("location_state"):
                            state = item["location_state"][0] if isinstance(item["location_state"], list) else item["location_state"]
                        elif item.get("state"):
                            state = item["state"][0] if isinstance(item["state"], list) else item["state"]
                        elif item.get("location"):
                            loc = item["location"]
                            state = loc[0] if isinstance(loc, list) else loc

                        stats["states_distribution"][state] = stats["states_distribution"].get(state, 0) + 1

                        # Get title
                        title = ""
                        if item.get("title"):
                            title = item["title"] if isinstance(item["title"], str) else item["title"][0]
                        elif item.get("partof_title"):
                            title = item["partof_title"] if isinstance(item["partof_title"], str) else item["partof_title"][0]

                        record = {
                            "text": text,
                            "source": "chronicling_america",
                            "date": date_str,
                            "year": year,
                            "title": title,
                            "state": state,
                            "url": item.get("url", item.get("id", "")),
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')

                    except Exception as e:
                        print(f"Error downloading {ocr_url}: {e}")
                        stats["rejected"] += 1

                page_num += 1

                # Check if we've reached the end
                pagination = results.get("pagination", {})
                next_url = pagination.get("next")
                if not next_url and page_num > 1:
                    # No more pages
                    if len(items) < 25:  # Less than requested = last page
                        break

            except Exception as e:
                print(f"Error on page {page_num}: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError("Chronicling America acquisition failed") from e

    # Save stats
    stats_file = os.path.join(output_dir, f"newspapers_{cutoff}_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults:")
    print(f"  Processed: {stats['total_processed']}")
    print(f"  Accepted: {stats['accepted']}")
    print(f"  Total chars: {stats['total_chars']:,}")
    print(f"  Output: {output_file}")
    print(
        "  Acquisition counts: "
        f"requested={stats['records_requested']} "
        f"fetched={stats['records_fetched']} written={stats['records_written']}"
    )

    if stats["records_written"] == 0:
        raise RuntimeError(
            "Chronicling America acquired no records "
            f"(requested={stats['records_requested']}, "
            f"fetched={stats['records_fetched']}, written=0)"
        )
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Chronicling America newspapers")
    parser.add_argument("--cutoff", type=str, default="1913",
                        help="Temporal cutoff year")
    parser.add_argument("--output-dir", type=str, default="data/raw/chronicling_america",
                        help="Output directory")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Maximum pages to download")
    parser.add_argument("--min-year", type=int, default=1800,
                        help="Minimum year to include")

    args = parser.parse_args()

    download_chronicling_america(
        cutoff=args.cutoff,
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        min_year=args.min_year,
    )
