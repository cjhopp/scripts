#!/usr/bin/env python3

import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

LOG_RANGE_RE = re.compile(
    r"Processing time range:\s+([0-9T:\-.]+)Z\s+to\s+([0-9T:\-.]+)Z"
)


def parse_iso_utc(value: str) -> datetime:
    """Parse ISO datetime string as UTC, allowing fractional seconds."""
    if value.endswith("Z"):
        value = value[:-1]
    if "." in value:
        base, frac = value.split(".", 1)
        frac = (frac + "000000")[:6]
        value = f"{base}.{frac}"
        dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f")
    else:
        dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    return dt.replace(tzinfo=timezone.utc)


def parse_date_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def days_for_interval(start: datetime, end: datetime) -> List[str]:
    """
    Replicate the workflow's daily stepping logic and emit YYYYMMDD labels.
    """
    days: List[str] = []
    current = start
    while current < end:
        days.append(current.strftime("%Y%m%d"))
        current = min(current + timedelta(days=1), end)
    return days


def range_from_log(log_file: Path) -> Optional[Tuple[datetime, datetime]]:
    """Extract processing range from an error log file."""
    if not log_file.exists():
        return None
    try:
        for line in log_file.read_text(errors="ignore").splitlines():
            match = LOG_RANGE_RE.search(line)
            if match:
                return parse_iso_utc(match.group(1)), parse_iso_utc(match.group(2))
    except OSError:
        return None
    return None


def range_from_split(instance: int, start: datetime, end: datetime, splits: int) -> Tuple[datetime, datetime]:
    total = (end - start).total_seconds()
    chunk = total / splits
    task_start = start + timedelta(seconds=instance * chunk)
    task_end = min(start + timedelta(seconds=(instance + 1) * chunk), end)
    return task_start, task_end


def parse_indices(indices: Optional[str], indices_file: Optional[Path], logs_dir: Path) -> List[int]:
    if indices:
        return sorted({int(x.strip()) for x in indices.split(",") if x.strip()})

    if indices_file:
        text = indices_file.read_text().strip()
        if not text:
            return []
        return sorted({int(x.strip()) for x in text.split(",") if x.strip()})

    found = []
    for log in logs_dir.glob("Smackover-MF_analyzed_err_*.txt"):
        m = re.search(r"Smackover-MF_analyzed_err_(\d+)\.txt$", log.name)
        if m:
            found.append(int(m.group(1)))
    return sorted(set(found))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether each expected day for each array index has output files. "
            "Expected days are read from log processing ranges when available; "
            "otherwise derived from --start/--end/--splits."
        )
    )
    parser.add_argument("--indices", help="Comma-separated array indices to check")
    parser.add_argument("--indices-file", type=Path, help="File containing comma-separated indices")
    parser.add_argument("--logs-dir", type=Path, default=Path.home(), help="Directory with Smackover-MF_analyzed_err_*.txt")

    parser.add_argument("--start", default="2009-02-12", help="Global workflow start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-31", help="Global workflow end date (YYYY-MM-DD)")
    parser.add_argument("--splits", type=int, default=368, help="Total number of split jobs")

    parser.add_argument(
        "--party-dir",
        type=Path,
        default=Path("/global/scratch/users/chopp/chet-meq/smackover/detections/parties/smackover_north_analyzed/MAD12_2hr"),
        help="Directory containing party_YYYYMMDD.tgz files",
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=None,
        help="Directory containing refined_picks_YYYYMMDD.xml files (default: party-dir)",
    )
    parser.add_argument(
        "--require",
        choices=["party", "catalog", "either", "both"],
        default="either",
        help="File requirement per day",
    )
    parser.add_argument(
        "--only-missing-indices",
        action="store_true",
        help="Print only the comma-separated indices that are missing one or more days",
    )
    parser.add_argument(
        "--only-complete-indices",
        action="store_true",
        help="Print only the comma-separated indices with no missing days",
    )
    return parser


def day_is_complete(day: str, party_dir: Path, catalog_dir: Path, require: str) -> bool:
    party_file = party_dir / f"party_{day}.tgz"
    catalog_file = catalog_dir / f"refined_picks_{day}.xml"
    party_exists = party_file.exists()
    catalog_exists = catalog_file.exists()

    if require == "party":
        return party_exists
    if require == "catalog":
        return catalog_exists
    if require == "both":
        return party_exists and catalog_exists
    return party_exists or catalog_exists


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    logs_dir = args.logs_dir
    party_dir = args.party_dir
    catalog_dir = args.catalog_dir or party_dir

    global_start = parse_date_utc(args.start)
    global_end = parse_date_utc(args.end)

    indices = parse_indices(args.indices, args.indices_file, logs_dir)
    if not indices:
        print("No indices found to check.")
        return 0

    missing_indices: List[int] = []
    complete_indices: List[int] = []

    if args.only_missing_indices and args.only_complete_indices:
        parser.error("--only-missing-indices and --only-complete-indices cannot be used together")

    for idx in indices:
        log_file = logs_dir / f"Smackover-MF_analyzed_err_{idx}.txt"
        log_range = range_from_log(log_file)
        if log_range is not None:
            task_start, task_end = log_range
            source = "log"
        else:
            task_start, task_end = range_from_split(idx, global_start, global_end, args.splits)
            source = "split"

        expected_days = days_for_interval(task_start, task_end)
        missing_days = [
            day for day in expected_days
            if not day_is_complete(day, party_dir=party_dir, catalog_dir=catalog_dir, require=args.require)
        ]

        if missing_days:
            missing_indices.append(idx)
        else:
            complete_indices.append(idx)

        if not args.only_missing_indices:
            status = "COMPLETE" if not missing_days else "MISSING"
            print(
                f"index={idx} status={status} expected_days={len(expected_days)} "
                f"missing_days={len(missing_days)} range_source={source}"
            )
            if missing_days:
                print("  missing: " + ",".join(missing_days))

    if args.only_missing_indices:
        print(",".join(str(i) for i in missing_indices))
    elif args.only_complete_indices:
        print(",".join(str(i) for i in complete_indices))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
