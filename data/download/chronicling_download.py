"""
Download historical newspapers from Chronicling America (Library of Congress).

Covers newspapers from 1756-1963.
Uses the LOC API for metadata and bulk OCR downloads.
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

# LOC API base URL
LOC_API_BASE = "https://chroniclingamerica.loc.gov"

# Bulk OCR download base
BULK_OCR_BASE = "https://chroniclingamerica.loc.gov/ocr"


def search_pages(
    date_start: str,
    date_end: str,
    state: Optional[str] = None,
    page: int = 1,
    rows: int = 100,
) -> Dict[str, Any]:
    """
    Search for newspaper pages within a date range.

    Args:
        date_start: Start date in YYYY format
        date_end: End date in YYYY format
        state: Optional state filter
        page: Page number for pagination
        rows: Number of results per page
    """
    params = {
        "dateFilterType": "yearRange",
        "date1": date_start,
        "date2": date_end,
        "format": "json",
        "page": page,
        "rows": rows,
    }

    if state:
        params["state"] = state

    url = f"{LOC_API_BASE}/search/pages/results/"
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
        "total_processed": 0,
        "accepted": 0,
        "rejected": 0,
        "total_chars": 0,
        "years_distribution": {},
        "states_distribution": {},
    }

    output_file = os.path.join(output_dir, f"newspapers_{cutoff}.jsonl")

    # Rate limiting
    request_delay = 1.0  # seconds between requests (be nice to LOC servers)

    with open(output_file, 'w', encoding='utf-8') as f:
        # Search for pages in date range
        print(f"\nSearching for pages from {min_year} to {cutoff_year}...")

        page_num = 1
        total_results = None

        while True:
            if max_pages and stats["accepted"] >= max_pages:
                break

            try:
                results = search_pages(
                    date_start=str(min_year),
                    date_end=str(cutoff_year),
                    page=page_num,
                    rows=100,
                )

                if total_results is None:
                    total_results = results.get("totalItems", 0)
                    print(f"Found {total_results:,} total pages")

                items = results.get("items", [])
                if not items:
                    break

                for item in tqdm(items, desc=f"Page {page_num}"):
                    if max_pages and stats["accepted"] >= max_pages:
                        break

                    stats["total_processed"] += 1

                    # Extract metadata
                    date_str = item.get("date", "")
                    try:
                        year = int(date_str[:4]) if date_str else None
                    except ValueError:
                        year = None

                    if year is None or year > cutoff_year:
                        stats["rejected"] += 1
                        continue

                    # Get OCR URL
                    ocr_url = item.get("ocr_eng")
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

                        text = response.text.strip()

                        if len(text) < 100:
                            stats["rejected"] += 1
                            continue

                        stats["accepted"] += 1
                        stats["total_chars"] += len(text)

                        # Record statistics
                        decade = (year // 10) * 10
                        stats["years_distribution"][decade] = stats["years_distribution"].get(decade, 0) + 1

                        state = item.get("state", ["Unknown"])[0] if item.get("state") else "Unknown"
                        stats["states_distribution"][state] = stats["states_distribution"].get(state, 0) + 1

                        record = {
                            "text": text,
                            "source": "chronicling_america",
                            "date": date_str,
                            "year": year,
                            "title": item.get("title", ""),
                            "state": state,
                            "city": item.get("city", ""),
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')

                    except Exception as e:
                        print(f"Error downloading {ocr_url}: {e}")
                        stats["rejected"] += 1

                page_num += 1
                time.sleep(request_delay)  # Rate limiting between pages

            except Exception as e:
                print(f"Error on page {page_num}: {e}")
                break

    # Save stats
    stats_file = os.path.join(output_dir, f"newspapers_{cutoff}_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults:")
    print(f"  Processed: {stats['total_processed']}")
    print(f"  Accepted: {stats['accepted']}")
    print(f"  Total chars: {stats['total_chars']:,}")
    print(f"  Output: {output_file}")

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
