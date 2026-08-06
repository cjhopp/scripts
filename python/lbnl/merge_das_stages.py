#!/usr/bin/env python3
"""
Merge per-event SCML files into one XML per stage.

Uses fast ElementTree splicing by default (--method=fast), which extracts
the children of <EventParameters> from each file and concatenates them
into a single SC3ML document without invoking scxmlmerge.  This avoids
the O(n²) publicID deduplication that makes scxmlmerge prohibitively slow
on large stages (1000+ files).

Use --method=scxmlmerge to fall back to the original behaviour.

Handles:
  - Combining _P_ttime and _S_ttime files for the same stage
  - Normalising duplicate naming variants (7pa_26 == 7pa26 etc.)
  - Combined FebMar files alongside _P/_S splits (groups them together;
    see NOTE below)

NOTE: BearskinFeb_1IA-13 has three source files (combined + _P + _S).
All three map to the same stage key and will appear multiple times in
the merged output.

Usage:
    python3 merge_das_stages.py [--scml-dir DIR] [--out-dir DIR] [--dry-run]
                                [--method {fast,scxmlmerge}]
"""
import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

DEFAULT_SCML_DIR = (
    "/media/chopp/HDD1/chet-meq/cape_modern/catalogs/fervo/"
    "DAS_picks/scml_events_v2"
)

# Files in the SCML directory to skip entirely
SKIP_FILES = {"DAS_stations.xml"}


def canonical_stage(stem: str) -> str:
    """
    Map an SCML file stem to a normalised stage key.

    Transformation steps (applied in order):
      1. Strip __<event_id> suffix
      2. Strip _P_ttime or _S_ttime suffix   (JulAug ttime files)
      3. Strip bare _ttime suffix            (P_test combined files)
      4. Strip bare _P or _S suffix          (FebMar split files)
      5. Collapse 7pa_N → 7paN              (duplicate naming variants)

    Examples
    --------
    7pa26_P_ttime__14453    → 7pa26
    7pa_26_S_ttime__14453   → 7pa26    (underscore variant normalised)
    BearskinFeb_1IA-13_P__50016 → BearskinFeb_1IA-13
    BearskinFeb_1IA-13__50016   → BearskinFeb_1IA-13
    P_test_ttime__11000     → P_test
    P_test_old_ttime__11000 → P_test_old
    post_stim_P_ttime__9999 → post_stim
    7pa22_23_23_S_ttime__1  → 7pa22_23_23
    """
    s = stem
    s = re.sub(r'__\d+$',       '', s)   # strip event ID
    s = re.sub(r'_[PS]_ttime$', '', s)   # strip _P_ttime / _S_ttime
    s = re.sub(r'_ttime$',      '', s)   # strip bare _ttime
    s = re.sub(r'_[PS]$',       '', s)   # strip bare _P / _S
    s = re.sub(r'(7pa)_(\d)',   r'\1\2', s)  # 7pa_NN → 7paNsuffix
    return s


def _merge_fast(files: list, out: Path) -> None:
    """
    Merge SC3ML files by splicing <EventParameters> children directly.

    Reads the namespace and schema version from the first file, then
    appends every child element from every file's <EventParameters>
    block into a single output document.  O(n) in total pick count —
    no deduplication, no cross-referencing.
    """
    NS   = "http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.12"
    EP   = f"{{{NS}}}EventParameters"
    ROOT = f"{{{NS}}}seiscomp"

    ET.register_namespace("", NS)

    root_out = None
    ep_out   = None

    for fpath in files:
        try:
            tree = ET.parse(fpath)
        except ET.ParseError as exc:
            print(f"    WARN: skipping {fpath.name}: {exc}", file=sys.stderr)
            continue

        root_in = tree.getroot()
        ep_in   = root_in.find(EP)
        if ep_in is None:
            continue

        if root_out is None:
            # Initialise output tree from first file
            root_out = ET.Element(root_in.tag, root_in.attrib)
            ep_out   = ET.SubElement(root_out, EP)

        for child in ep_in:
            ep_out.append(child)

    if root_out is None:
        print(f"    WARN: no valid input for {out.name}", file=sys.stderr)
        return

    ET.indent(root_out, space="  ")
    tree_out = ET.ElementTree(root_out)
    tree_out.write(str(out), encoding="unicode", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scml-dir", default=DEFAULT_SCML_DIR,
                        help="Directory containing per-event SCML files")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for merged files "
                             "(default: <scml-dir>/merged)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print groupings without merging")
    parser.add_argument("--method", choices=["fast", "scxmlmerge"], default="fast",
                        help="Merge method: 'fast' (default) splices XML directly; "
                             "'scxmlmerge' calls the SeisComP tool")
    args = parser.parse_args()

    scml_dir = Path(args.scml_dir)
    out_dir  = Path(args.out_dir) if args.out_dir else scml_dir / "merged"

    if not scml_dir.is_dir():
        sys.exit(f"ERROR: SCML directory not found: {scml_dir}")

    # ── Group files by canonical stage ─────────────────────────────────────
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(scml_dir.glob("*.xml")):
        if f.name in SKIP_FILES:
            continue
        stage = canonical_stage(f.stem)
        groups[stage].append(f)

    n_files = sum(len(v) for v in groups.values())
    print(f"Found {len(groups)} stages across {n_files} SCML files")
    print(f"Output directory: {out_dir}  method: {args.method}\n")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Merge ───────────────────────────────────────────────────────────────
    errors = []
    for stage in sorted(groups):
        files = groups[stage]
        out   = out_dir / f"{stage}.xml"
        print(f"  {stage:35s}  {len(files):5d} files  →  {out.name}", flush=True)

        if args.dry_run:
            continue

        if args.method == "fast":
            _merge_fast(files, out)
        else:
            cmd = ["scxmlmerge"] + [str(f) for f in files]
            with open(out, "w") as fh:
                result = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                msg = f"scxmlmerge failed for stage '{stage}': {result.stderr.strip()}"
                print(f"    WARNING: {msg}", file=sys.stderr)
                errors.append(msg)

    print(f"\nDone. {len(groups) - len(errors)} stages merged successfully.")
    if errors:
        print(f"{len(errors)} stage(s) failed:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
