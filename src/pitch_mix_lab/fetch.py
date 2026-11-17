"""Download a date-bounded Statcast pitch dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from pybaseball import statcast


def fetch(start: str, end: str, output: Path) -> None:
    """Fetch Statcast pitches and save them as CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    data = statcast(start_dt=start, end_dt=end)
    if data is None or data.empty:
        raise RuntimeError(f"No Statcast rows returned for {start} through {end}.")
    data.to_csv(output, index=False)
    print(f"Saved {len(data):,} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date, YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=Path("data/raw/statcast.csv"))
    args = parser.parse_args()
    fetch(args.start, args.end, args.output)


if __name__ == "__main__":
    main()
