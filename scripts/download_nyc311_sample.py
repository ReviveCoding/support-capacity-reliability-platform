from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

from support_capacity_reliability.data.public_adapters import nyc311_query_url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50_000)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--output", default="data/public/nyc311_sample.csv")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    url = nyc311_query_url(args.limit, args.start_date)
    print(f"Downloading public NYC 311 proxy data from: {url}")
    urlretrieve(url, output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
