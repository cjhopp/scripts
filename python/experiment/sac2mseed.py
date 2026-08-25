#!/usr/bin/env python3
"""
sac2mseed.py -- SAC float32 millivolts (at ADC input) -> miniSEED int32 counts,
written directly into an SDS archive tree.

Input is one SAC file per node-channel-day at 1000 Hz, nominally 86400001
samples, where the final sample is the 00:00:00.000 sample of the *next* day
and is trimmed.  Files that start partway through a day are handled: the trim
is computed from the actual start time, and a file whose samples extend more
than one sample past midnight is rejected unless --allow-crossing is given,
because truncating it would silently discard data.

Output layout is SDS:

    <root>/<YEAR>/<NET>/<STA>/<CHA>.D/<NET>.<STA>.<LOC>.<CHA>.D.<YEAR>.<JDAY>

so the tree can be rsync'd into a live archive with no repacking.  An SDS day
file has no room to disambiguate partial days, so all SAC files belonging to
one (station, location, channel, day) are grouped, sorted by start time, and
concatenated into a single day file.  The group is the unit of work: it is
built in a temp file alongside its destination, every trace is round-trip
verified before it is appended, and only then is it os.replace()d into
position.  Presence of the final path is therefore proof of a complete,
verified day file, which is what makes --sentinel and the rolling reap
meaningful.  A destination that already exists is skipped by default, so a
preempted task is restarted by rerunning it with identical arguments.

Nothing from the SAC identifier fields survives.  network / station / location
/ channel and the sampling rate are all imposed; the discarded SAC ids are
recorded in the manifest so the character-overflow question can be answered
later without re-reading the data.

Feed one station per invocation -- 'sacfiles' accepts '-' to read paths from
stdin.  Concurrent tasks must not share a station, since the destination path
is derived from it; with one station per task no two tasks can ever target the
same day file, and only <YEAR> and <YEAR>/<NET> are created concurrently
(mkdir exist_ok handles that).
"""

import argparse
import io
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from obspy import UTCDateTime, read

# ---- deployment configuration (was station_config.py) ----------------------
COUNTS_NUM, COUNTS_DEN = 33554432.0, 10000.0  # 2**25 / 10000 -> counts per mV
SAMPLING_RATE = 1000.0                        # asserted against sac.delta below

NETWORK = "IG"     # <-- FDSN code; "Cape" is 4 chars and will not fit SEED

# STATION_MAP is deliberately empty: the 4-digit IGU node id parsed from the
# filename IS the station code.  Passthrough here is intended, not unfinished.
# Populate it only if a deployment actually renames nodes.
STATION_MAP = {}

# CHANNEL_MAP is strict -- an unmapped channel is a hard error, never a
# passthrough.  SEED band code 'D' is only valid for 250 <= sps < 1000; at
# exactly 1000 sps the band code must be 'G'.  A passthrough would write 'DPZ',
# which is a well-formed 3-character code and sails through every length check
# downstream, so the only place this can be caught is here.
# Census of cape_manifest_dedup.tsv (386,117 files): GPZ 128723, GPN 128707,
# GPE 128687.  The band code is already G, which is what 1000 sps requires,
# so this map is the identity -- it exists to make an unrecognised code a
# hard error instead of a silent passthrough on the next deployment.
# Orientation is geographic (N/E), which asserts the nodes were oriented;
# StationXML must declare azimuth 0/90 and dip 0 to match.
CHANNEL_MAP = {"GPZ": "GPZ", "GPN": "GPN", "GPE": "GPE"}
# ---------------------------------------------------------------------------

SCHEMA = 5                      # 4 -> 5: SDS layout, day grouping, atomic place
CHUNK = 1 << 22                 # 4 Mi samples ~= 33 MB as float64
ADC_FULL_SCALE = 1 << 23        # 24-bit signed converter
DATAQUALITY = "D"
RECLEN = 4096

if int(SAMPLING_RATE) != SAMPLING_RATE:
    raise ImportError(f"SAMPLING_RATE {SAMPLING_RATE!r} must be an integer")
NS_PER_SAMPLE, _rem = divmod(1_000_000_000, int(SAMPLING_RATE))
if _rem:
    raise ImportError(f"{SAMPLING_RATE} Hz does not divide 1e9 ns evenly; "
                      "the integer sample-count arithmetic below is invalid")

FNAME = re.compile(r"^(?P<proj>[^.]+)\.(?P<node>\d{4})\.(?P<loc>\d{2})\."
                   r"(?P<year>\d{4})\.(?P<jday>\d{3})\."
                   r"(?P<hh>\d{2})\.(?P<mm>\d{2})\.(?P<ss>\d{2})\."
                   r"(?P<chan>[A-Z0-9]+)\.sac$")


def _round_ns(ns, quantum=1_000_000):
    """Round an integer nanosecond count to the nearest quantum (default 1 ms),
    ties away from zero.  Pure integer arithmetic: UTCDateTime.ns for 2024 is
    ~1.7e18, far past 2**53, so anything routed through float64 can land off a
    millisecond boundary by a couple hundred nanoseconds."""
    if ns < 0:
        return -(((-ns) + quantum // 2) // quantum) * quantum
    return ((ns + quantum // 2) // quantum) * quantum


def parse_name(path):
    m = FNAME.match(Path(path).name)
    if not m:
        raise ValueError("filename does not match expected pattern")
    return m


def map_station(node):
    sta = STATION_MAP.get(node, node)
    if not (1 <= len(sta) <= 5) or not sta.isalnum():
        raise ValueError(f"station code {sta!r} invalid for SEED "
                         f"(1-5 alphanumeric)")
    return sta


def map_channel(chan):
    try:
        return CHANNEL_MAP[chan]
    except KeyError:
        raise ValueError(
            f"channel code {chan!r} not in CHANNEL_MAP "
            f"(known: {', '.join(sorted(CHANNEL_MAP))})") from None


def sds_path(root, net, sta, loc, cha, year, jday, quality=DATAQUALITY):
    """<root>/<Y>/<NET>/<STA>/<CHA>.<Q>/<NET>.<STA>.<LOC>.<CHA>.<Q>.<Y>.<JJJ>"""
    name = f"{net}.{sta}.{loc}.{cha}.{quality}.{year:04d}.{jday:03d}"
    return (Path(root) / f"{year:04d}" / net / sta / f"{cha}.{quality}" / name)


def read_and_check(path, headonly=False):
    """Read one SAC file and vet every header field the conversion depends on."""
    m = parse_name(path)

    st = read(path, format="SAC", headonly=headonly)   # fsize check on by default
    if len(st) != 1:
        raise ValueError(f"expected 1 trace, got {len(st)}")
    tr = st[0]
    sac = tr.stats.sac

    if sac.get("iftype", 1) != 1:
        raise ValueError(f"iftype={sac.get('iftype')} is not ITIME")
    if sac.get("leven", 1) != 1:
        raise ValueError("unevenly sampled")
    if abs(sac.delta - 1.0 / SAMPLING_RATE) > 1e-9:
        raise ValueError(f"delta {sac.delta!r} inconsistent with {SAMPLING_RATE} Hz")

    ft = UTCDateTime(year=int(m["year"]), julday=int(m["jday"]),
                     hour=int(m["hh"]), minute=int(m["mm"]), second=int(m["ss"]))
    skew = float(tr.stats.starttime - ft)
    if abs(skew) > 0.5:
        raise ValueError(f"header start {tr.stats.starttime} "
                         f"disagrees with filename {ft}")
    return tr, m, skew


def restat(tr, m):
    """Replace every identifier and the sample rate.  Returns the discarded
    SAC identifier tuple."""
    sac_ids = (tr.stats.network, tr.stats.station,
               tr.stats.location, tr.stats.channel)

    station = map_station(m["node"])
    channel = map_channel(m["chan"])
    location = m["loc"]              # filename is authoritative

    # The MSEED writer truncates over-long codes rather than refusing them.
    for label, val, n in (("network", NETWORK, 2), ("station", station, 5),
                          ("location", location, 2), ("channel", channel, 3)):
        if len(val) > n:
            raise ValueError(f"{label} code {val!r} exceeds {n} characters")
    if len(channel) != 3:
        raise ValueError(f"channel code {channel!r} must be exactly 3 characters")

    # sampling_rate is the stored field; delta is derived as 1/sampling_rate and
    # endtime is refreshed on every assignment, so ordering here is irrelevant.
    tr.stats.sampling_rate = SAMPLING_RATE          # never 1/delta from SAC
    tr.stats.starttime = UTCDateTime(ns=_round_ns(tr.stats.starttime.ns))
    tr.stats.network = NETWORK
    tr.stats.station = station
    tr.stats.location = location
    tr.stats.channel = channel
    tr.stats.mseed = {"dataquality": DATAQUALITY}
    del tr.stats.sac
    return sac_ids


def plan_trim(tr):
    """Return (n_keep, n_past_midnight).  Call only after restat(), which
    ms-aligns starttime, so the sample count to midnight is exact in integers."""
    npts = tr.stats.npts
    t0 = tr.stats.starttime
    midnight = UTCDateTime(t0.year, t0.month, t0.day) + 86400
    n_to_midnight = (midnight.ns - t0.ns) // NS_PER_SAMPLE
    n_past = max(0, npts - n_to_midnight)
    n_keep = min(npts, n_to_midnight)
    return int(n_keep), int(n_past)


def check_crossing(n_past, allow_crossing=False):
    if n_past > 1 and not allow_crossing:
        raise ValueError(
            f"file extends {n_past} samples past midnight "
            f"({(n_past - 1) / SAMPLING_RATE:.3f} s of data would be discarded); "
            f"use --allow-crossing to truncate anyway")


def to_counts(tr, n_keep, strict=False):
    """float32 mV -> int32 counts, chunked, keeping the first n_keep samples."""
    npts = tr.stats.npts
    if n_keep <= 0:
        raise ValueError("nothing left after trimming")

    x = tr.data
    counts = np.empty(n_keep, dtype=np.int32)
    rmax, rsum = 0.0, 0.0
    y = n = r = None
    for i in range(0, n_keep, CHUNK):
        j = min(i + CHUNK, n_keep)
        y = (x[i:j].astype(np.float64) * COUNTS_NUM) / COUNTS_DEN
        if not np.isfinite(y).all():
            raise ValueError(f"non-finite sample near index {i}")
        n = np.rint(y)
        r = np.abs(y - n)
        rmax = max(rmax, float(r.max()))
        rsum += float((r ** 2).sum())
        if n.min() < -2**31 or n.max() > 2**31 - 1:
            raise ValueError("int32 overflow")
        counts[i:j] = n.astype(np.int32)
    tr.data = counts                  # also refreshes stats.npts and endtime
    del x, y, n, r                    # release the 345 MB float32 array

    # The SAC files hold exact integer counts expressed in float32 mV, but
    # float32 has 24 mantissa bits and one count is 1e4/2**25 mV, so counts
    # above ~2**24/625 (~26.8k) are not exactly representable and come back
    # off by up to ~count*2**-24.  A fixed tolerance therefore rejects loud
    # but valid traces.  Scale the per-sample gate with amplitude and let
    # the RMS do the real work: a wrong COUNTS_NUM/COUNTS_DEN spreads
    # residuals uniformly over every sample and lands near 0.289, while
    # float32 quantization stays several orders of magnitude below that.
    cmax = float(np.abs(counts).max())
    rrms = (rsum / n_keep) ** 0.5
    tol = max(0.01, 4.0 * cmax * 2.0 ** -24)
    if rmax >= tol or rrms >= 0.05:
        raise ValueError(f"not integer counts: max residual {rmax:.3e} "
                         f"(tol {tol:.3e}), rms {rrms:.3e}")
    if counts.min() <= -ADC_FULL_SCALE or counts.max() >= ADC_FULL_SCALE - 1:
        raise ValueError(f"samples at or beyond ADC full scale "
                         f"(+/-{ADC_FULL_SCALE}) -- clipped, or gain is not 16")
    if strict:
        u = np.unique(counts)
        if u.size >= 2:
            g = int(np.gcd.reduce(np.diff(u)))
            if g != 1:
                raise ValueError(f"coarser lattice present: gcd={g}")

    return dict(schema=SCHEMA, npts_in=int(npts), npts_out=int(n_keep),
                dropped=int(npts - n_keep), residual_max=rmax,
                residual_rms=rrms,
                count_min=int(counts.min()), count_max=int(counts.max()))


def convert(path, strict=False, allow_crossing=False):
    tr, m, skew = read_and_check(path)
    sac_ids = restat(tr, m)
    n_keep, n_past = plan_trim(tr)
    check_crossing(n_past, allow_crossing)
    rec = to_counts(tr, n_keep, strict)
    rec.update(source=str(path), node=m["node"], location=m["loc"],
               sac_ids_discarded=list(sac_ids), n_past_midnight=n_past,
               filename_skew_s=skew)
    return tr, rec


def encode_and_verify(tr, reclen=RECLEN):
    """Encode one trace to STEIM2 in memory and round-trip verify it.  Returns
    the raw record bytes, ready to append to a day file."""
    buf = io.BytesIO()
    tr.write(buf, format="MSEED", encoding="STEIM2", reclen=reclen, byteorder=">")
    raw = buf.getvalue()
    if not raw or len(raw) % reclen:
        raise ValueError(f"encoded length {len(raw)} is not a multiple of {reclen}")
    back = read(io.BytesIO(raw), format="MSEED")
    s = tr.stats
    good = (len(back) == 1
            and back[0].id == tr.id
            and back[0].stats.npts == s.npts
            and back[0].stats.starttime.ns == s.starttime.ns
            and abs(back[0].stats.sampling_rate - SAMPLING_RATE) <= 1e-9
            and back[0].data.dtype == np.int32
            and np.array_equal(back[0].data, tr.data))
    if not good:
        raise ValueError("round-trip verification failed")
    del back
    return raw


def build_day(dest, sources, strict=False, allow_crossing=False,
              reclen=RECLEN, quiet=False):
    """Convert every source for one SDS day file into a temp file alongside the
    destination, then os.replace() it into position.  Returns the manifest
    records.  On any failure the temp file is removed and dest is untouched."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(dir=str(dest.parent), prefix=dest.name + ".tmp.")
    tmp = Path(tmpname)
    recs = []
    prev_end_ns, prev_src = None, None
    try:
        with os.fdopen(fd, "wb") as fh:
            for idx, src in enumerate(sources):
                tr, rec = convert(src, strict, allow_crossing)
                s = tr.stats
                if prev_end_ns is not None:
                    delta_ns = s.starttime.ns - prev_end_ns
                    if delta_ns < NS_PER_SAMPLE:
                        raise ValueError(
                            f"{src} overlaps {prev_src} by "
                            f"{(NS_PER_SAMPLE - delta_ns) / 1e9:.6f} s within one "
                            f"day file; SDS cannot hold both without merging")
                    rec["gap_before_s"] = (delta_ns - NS_PER_SAMPLE) / 1e9
                raw = encode_and_verify(tr, reclen)
                fh.write(raw)
                prev_end_ns, prev_src = s.endtime.ns, src
                rec.update(dest=str(dest), index_in_day=idx,
                           sources_in_day=len(sources), seed_id=tr.id,
                           starttime=str(s.starttime), endtime=str(s.endtime),
                           sampling_rate=SAMPLING_RATE, reclen=reclen,
                           bytes_written=len(raw), roundtrip="OK",
                           preamp_removed=False,
                           factor=f"{COUNTS_NUM}/{COUNTS_DEN} counts per mV @ADC input")
                recs.append(rec)
                if not quiet:
                    print(f"{src}\n  -> {dest.name} [{idx + 1}/{len(sources)}]  "
                          f"n={s.npts}  {s.starttime} .. {s.endtime}  "
                          f"res={rec['residual_max']:.2e}")
                del tr, raw
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
        try:
            dfd = os.open(str(dest.parent), os.O_DIRECTORY)
            os.fsync(dfd)
            os.close(dfd)
        except OSError:
            pass
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return recs


def iter_paths(argv_paths):
    if len(argv_paths) == 1 and argv_paths[0] == "-":
        for line in sys.stdin:
            line = line.strip()
            if line:
                yield line
    else:
        yield from argv_paths


def write_sentinel(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp.")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmpname, path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sacfiles", nargs="+", help="SAC paths, or '-' to read from stdin")
    ap.add_argument("-o", "--outdir", required=True, help="SDS archive root")
    ap.add_argument("--manifest", default=None,
                    help="JSONL manifest path; default "
                         "<outdir>/../_state/manifests/conversion_manifest.<tag>.jsonl")
    ap.add_argument("--tag", default=None,
                    help="per-task manifest/sentinel tag; default SLURM_ARRAY_TASK_ID "
                         "or the pid.  One task per station means one file per "
                         "station, which avoids concurrent O_APPEND to a shared "
                         "manifest -- Lustre does not give append atomicity.")
    ap.add_argument("--sentinel", default=None,
                    help="write this file only if the whole task succeeded; "
                         "keep it OUTSIDE the SDS root")
    ap.add_argument("--allow-crossing", action="store_true",
                    help="truncate files that extend past midnight instead of failing")
    ap.add_argument("--strict", action="store_true",
                    help="also run the full-array GCD lattice check (slow)")
    ap.add_argument("--on-existing", choices=("skip", "overwrite", "fail"),
                    default="skip",
                    help="destination day file already present (default: skip, "
                         "which is what makes a rerun idempotent)")
    ap.add_argument("--reclen", type=int, default=RECLEN)
    ap.add_argument("--dry-run", action="store_true",
                    help="headers only: parse, vet, report, write nothing")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="suppress per-file lines; print the summary only")
    a = ap.parse_args()

    root = Path(a.outdir)
    tag = a.tag or os.environ.get("SLURM_ARRAY_TASK_ID") or str(os.getpid())

    # Group inputs by destination day file.  The key comes entirely from the
    # filename, so this costs no I/O and needs no headers.
    groups, fails, n_in = defaultdict(list), 0, 0
    for p in iter_paths(a.sacfiles):
        n_in += 1
        try:
            m = parse_name(p)
            dest = sds_path(root, NETWORK, map_station(m["node"]), m["loc"],
                            map_channel(m["chan"]), int(m["year"]), int(m["jday"]))
            groups[dest].append(((int(m["hh"]), int(m["mm"]), int(m["ss"]),
                                  str(p)), str(p)))
        except Exception as e:
            print(f"FAIL {p}: {e}", file=sys.stderr)
            fails += 1
    for dest in groups:
        groups[dest] = [p for _, p in sorted(groups[dest])]

    stations = sorted({d.parts[-3] for d in groups})
    multi = sum(1 for v in groups.values() if len(v) > 1)
    print(f"{n_in} inputs -> {len(groups)} day files across {len(stations)} "
          f"station(s); {multi} day(s) built from >1 file", file=sys.stderr)

    ok_days = skipped = 0
    crossings = shortdays = skewed = 0

    if a.dry_run:
        for dest in sorted(groups):
            srcs, total, prev_end_ns, bad = groups[dest], 0, None, False
            for src in srcs:
                try:
                    tr, m, skew = read_and_check(src, headonly=True)
                    restat(tr, m)
                    n_keep, n_past = plan_trim(tr)
                    if n_past > 1:
                        crossings += 1
                        print(f"CROSSING {src}: {n_past} samples past midnight",
                              file=sys.stderr)
                    check_crossing(n_past, a.allow_crossing)
                    if abs(skew) > 0.0005:
                        skewed += 1
                        print(f"SKEW {src}: header start off filename by "
                              f"{skew:+.6f} s", file=sys.stderr)
                    if prev_end_ns is not None:
                        d_ns = tr.stats.starttime.ns - prev_end_ns
                        if d_ns < NS_PER_SAMPLE:
                            raise ValueError(f"overlaps previous file in day by "
                                             f"{(NS_PER_SAMPLE - d_ns) / 1e9:.6f} s")
                    prev_end_ns = (tr.stats.starttime.ns
                                   + (n_keep - 1) * NS_PER_SAMPLE)
                    total += n_keep
                    if not a.quiet:
                        print(f"{src}\n  -> {dest.name}  {tr.id}  n={n_keep}  "
                              f"{tr.stats.starttime}")
                except Exception as e:
                    print(f"FAIL {src}: {e}", file=sys.stderr)
                    fails += 1
                    bad = True
            if total < 86400 * int(SAMPLING_RATE):
                shortdays += 1
            if not bad:
                ok_days += 1
        print(f"\ndry-run: {ok_days} day files would be written, {fails} failed; "
              f"{shortdays} short day(s), {crossings} midnight crossing(s), "
              f"{skewed} header/filename skew(s) > 0.5 ms", file=sys.stderr)
        sys.exit(1 if fails else 0)

    if a.manifest:
        mpath = Path(a.manifest)
    else:
        mpath = root.parent / "_state" / "manifests" / f"conversion_manifest.{tag}.jsonl"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    with open(mpath, "a") as mf:
        for dest in sorted(groups):
            if dest.exists():
                if a.on_existing == "skip":
                    skipped += 1
                    if not a.quiet:
                        print(f"SKIP {dest} (exists)")
                    continue
                if a.on_existing == "fail":
                    print(f"FAIL {dest}: exists (use --on-existing skip|overwrite)",
                          file=sys.stderr)
                    fails += 1
                    continue
            try:
                recs = build_day(dest, groups[dest], a.strict, a.allow_crossing,
                                 a.reclen, a.quiet)
            except Exception as e:
                print(f"FAIL {dest}: {e}", file=sys.stderr)
                fails += 1
                continue
            for rec in recs:
                mf.write(json.dumps(rec) + "\n")
            mf.flush()
            os.fsync(mf.fileno())
            ok_days += 1

    print(f"\n{ok_days} day files written, {skipped} skipped, {fails} failed "
          f"(manifest {mpath})", file=sys.stderr)

    if a.sentinel and not fails:
        write_sentinel(a.sentinel, dict(schema=SCHEMA, tag=tag,
                                        stations=stations, inputs=n_in,
                                        days_written=ok_days, days_skipped=skipped,
                                        manifest=str(mpath)))
        print(f"sentinel {a.sentinel}", file=sys.stderr)

    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
