#!/usr/bin/env python3

import argparse
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOG_RANGE_RE = re.compile(
    r"Processing time range:\s+([0-9T:\-.]+)Z\s+to\s+([0-9T:\-.]+)Z"
)
DAY_ERROR_RE = re.compile(r"Error processing day\s+([0-9]{4}-[0-9]{2}-[0-9]{2}):\s+(.*)")
PROCESSING_DAY_RE = re.compile(r"Processing\s+([0-9]{4}-[0-9]{2}-[0-9]{2})\.\.\.")

CLUSTER_HOME = Path("/global/home/users/chopp")
CLUSTER_SCRATCH = Path("/global/scratch/users/chopp")
CLUSTER_DETECTION_LOGS = Path(
    "/global/scratch/users/chopp/chet-meq/smackover/detections/logs/"
    "smackover_north_analyzed/MAD12_2hr"
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


def parse_day_errors(log_file: Path) -> Dict[str, List[str]]:
    """Parse per-day processing errors from the job log."""
    errors: Dict[str, List[str]] = {}
    if not log_file.exists():
        return errors
    try:
        for line in log_file.read_text(errors="ignore").splitlines():
            match = DAY_ERROR_RE.search(line)
            if not match:
                continue
            day = match.group(1).replace("-", "")
            msg = match.group(2).strip()
            errors.setdefault(day, []).append(msg)
    except OSError:
        return {}
    return errors


def parse_log_diagnostics(log_file: Path) -> Dict[str, object]:
    """Extract coarse health diagnostics from a job error log."""
    diagnostics: Dict[str, object] = {
        "log_exists": log_file.exists(),
        "range_found": False,
        "processing_complete": False,
        "cancelled_time_limit": False,
        "slurm_broken_pipe": False,
        "attempted_days": set(),
        "error_lines": [],
        "last_line": "",
    }
    if not diagnostics["log_exists"]:
        return diagnostics

    try:
        lines = log_file.read_text(errors="ignore").splitlines()
    except OSError:
        return diagnostics

    for line in lines:
        if LOG_RANGE_RE.search(line):
            diagnostics["range_found"] = True
        day_match = PROCESSING_DAY_RE.search(line)
        if day_match:
            diagnostics["attempted_days"].add(day_match.group(1).replace("-", ""))
        if "[INFO] Processing complete." in line:
            diagnostics["processing_complete"] = True
        if re.search(r"CANCELLED AT .* DUE TO TIME LIMIT", line):
            diagnostics["cancelled_time_limit"] = True
        if "BrokenPipeError" in line or "srun: error" in line:
            diagnostics["slurm_broken_pipe"] = True
        if "[ERROR]" in line:
            diagnostics["error_lines"].append(line.strip())

    # Track last non-empty line to catch abrupt exits with little metadata.
    for line in reversed(lines):
        if line.strip():
            diagnostics["last_line"] = line.strip()
            break
    return diagnostics


def classify_error(message: str) -> str:
    """Map an error message to a coarse cause category."""
    text = message.lower()
    bad_data_markers = [
        "less than 80 percent of the desired length",
        "will not pad",
        "ignore_bad_data",
        "gappy",
        "mostly zeros",
        "no appropriate data found",
    ]
    fdsn_markers = [
        "could not download data",
        "fdsn",
        "http",
        "service unavailable",
        "timed out",
        "timeout",
    ]
    if any(m in text for m in bad_data_markers):
        return "bad_data_preprocessing"
    if any(m in text for m in fdsn_markers):
        return "download_or_client"
    if "cancelled" in text and "time limit" in text:
        return "time_limit"
    if "brokenpipeerror" in text or "srun: error" in text:
        return "broken_pipe_or_slurm"
    return "other"


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


def resolve_log_file(idx: int, primary_logs_dir: Path, extra_log_dirs: List[Path]) -> Path:
    """Resolve the best-available log file path for an index."""
    filename = f"Smackover-MF_analyzed_err_{idx}.txt"
    candidates: List[Path] = [primary_logs_dir / filename]
    for extra in extra_log_dirs:
        candidate = extra / filename
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=CLUSTER_HOME,
        help="Primary directory with Smackover-MF_analyzed_err_*.txt",
    )
    parser.add_argument(
        "--extra-log-dirs",
        nargs="*",
        type=Path,
        default=[CLUSTER_HOME, CLUSTER_SCRATCH, CLUSTER_DETECTION_LOGS],
        help="Additional directories to search for log files",
    )

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
    parser.add_argument(
        "--max-missing-days-to-print",
        type=int,
        default=12,
        help="Maximum number of missing days to print per index in detailed mode",
    )
    parser.add_argument(
        "--print-log-diagnostics",
        action="store_true",
        help="Print log health diagnostics (range found, completion, attempted days, etc.)",
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
    extra_log_dirs = args.extra_log_dirs or [CLUSTER_HOME, CLUSTER_SCRATCH, CLUSTER_DETECTION_LOGS]

    global_start = parse_date_utc(args.start)
    global_end = parse_date_utc(args.end)

    indices = parse_indices(args.indices, args.indices_file, logs_dir)
    if not indices:
        print("No indices found to check.")
        return 0

    missing_indices: List[int] = []
    complete_indices: List[int] = []
    overall_error_categories: Counter = Counter()
    logs_missing_count = 0

    if args.only_missing_indices and args.only_complete_indices:
        parser.error("--only-missing-indices and --only-complete-indices cannot be used together")

    for idx in indices:
        log_file = resolve_log_file(idx=idx, primary_logs_dir=logs_dir, extra_log_dirs=extra_log_dirs)
        log_range = range_from_log(log_file)
        day_errors = parse_day_errors(log_file)
        log_diag = parse_log_diagnostics(log_file)
        if not log_diag["log_exists"]:
            logs_missing_count += 1
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

        attempted_days = log_diag["attempted_days"]
        missing_day_error_info: List[Tuple[str, str, str, str]] = []
        for day in missing_days:
            messages = day_errors.get(day, [])
            attempt_status = "attempted" if day in attempted_days else "not_attempted_in_log"
            if messages:
                msg = messages[-1]
                category = classify_error(msg)
                overall_error_categories[category] += 1
                missing_day_error_info.append((day, category, msg, attempt_status))
            else:
                overall_error_categories["no_day_error_in_log"] += 1
                missing_day_error_info.append((day, "no_day_error_in_log", "", attempt_status))

        if not args.only_missing_indices:
            status = "COMPLETE" if not missing_days else "MISSING"
            index_categories = Counter(category for _, category, _, _ in missing_day_error_info)
            attempt_summary = Counter(attempt_status for _, _, _, attempt_status in missing_day_error_info)
            category_summary = ",".join(
                f"{cat}:{count}" for cat, count in sorted(index_categories.items())
            ) or "none"
            attempt_summary_str = ",".join(
                f"{label}:{count}" for label, count in sorted(attempt_summary.items())
            ) or "none"
            print(
                f"index={idx} status={status} expected_days={len(expected_days)} "
                f"missing_days={len(missing_days)} range_source={source} "
                f"error_categories={category_summary} missing_day_attempts={attempt_summary_str}"
            )
            if args.print_log_diagnostics:
                print(
                    "  log_diagnostics="
                    f"path:{log_file} "
                    f"exists:{log_diag['log_exists']} "
                    f"range_found:{log_diag['range_found']} "
                    f"processing_complete:{log_diag['processing_complete']} "
                    f"cancelled_time_limit:{log_diag['cancelled_time_limit']} "
                    f"slurm_broken_pipe:{log_diag['slurm_broken_pipe']} "
                    f"attempted_days:{len(attempted_days)} "
                    f"error_lines:{len(log_diag['error_lines'])}"
                )
                if log_diag["error_lines"]:
                    print(f"  last_error={log_diag['error_lines'][-1]}")
                elif log_diag["last_line"]:
                    print(f"  last_line={log_diag['last_line']}")
            if missing_day_error_info:
                to_print = missing_day_error_info[: args.max_missing_days_to_print]
                for day, category, msg, attempt_status in to_print:
                    if msg:
                        print(
                            f"  missing_day={day} cause={category} "
                            f"attempt_status={attempt_status} msg={msg}"
                        )
                    else:
                        print(
                            f"  missing_day={day} cause={category} "
                            f"attempt_status={attempt_status}"
                        )
                if len(missing_day_error_info) > args.max_missing_days_to_print:
                    skipped = len(missing_day_error_info) - args.max_missing_days_to_print
                    print(f"  ... {skipped} additional missing days omitted")

    if args.only_missing_indices:
        print(",".join(str(i) for i in missing_indices))
    elif args.only_complete_indices:
        print(",".join(str(i) for i in complete_indices))
    elif overall_error_categories:
        summary = ",".join(
            f"{cat}:{count}" for cat, count in sorted(overall_error_categories.items())
        )
        print(f"overall_missing_day_error_categories={summary}")

    if logs_missing_count > 0:
        print(
            f"warning=missing_logs count={logs_missing_count} "
            "hint=use --logs-dir and/or --extra-log-dirs to point at your SLURM err files"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
