#!/usr/bin/env python3
"""
Plot AI (PhaseNet) picks vs. standard autopicks for many events.

Window:
    origin_time  …  origin_time + 5 s
Fallback:
    first_pick - 2 s … first_pick + 10 s

Events that have no picks inside the selected window are skipped.
"""

from __future__ import annotations
import argparse
import glob
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from itertools import islice
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Optional

import matplotlib.pyplot as plt
from obspy import read, read_events, Catalog, UTCDateTime

# ---------------------------------------------------------------------#
# --- helper: walk up the directory tree until a file is encountered --#
# ---------------------------------------------------------------------#
def _find_in_parents(
    start_dir: str,
    base_name: str,
    extensions: Iterable[str],
    max_levels: int = 5,
) -> Optional[str]:
    """
    Search *start_dir* and up to *max_levels* parents for a file whose
    basename matches *base_name* **or any truncated version obtained by
    repeatedly dropping the last '_'-separated token**.

    Example
    -------
    base_name = "lbnl2024dkex_default"
      -> we look for        lbnl2024dkex_default.ext
         then (if not found) lbnl2024dkex.ext
         then                lbnl2024dkex              (etc.)
    """
    # list of candidate basenames we are willing to try
    parts = base_name.split("_")
    candidates = ["_".join(parts[:i]) for i in range(len(parts), 0, -1)]

    cur = Path(start_dir).resolve()
    for _ in range(max_levels + 1):
        for stem in candidates:
            for ext in extensions:
                f = cur / f"{stem}{ext}"
                if f.is_file():
                    return str(f)
        cur = cur.parent
    return None

# -----------------------------------------------------------------------------#
# -------------------------  helper: load pick files  -------------------------#
# -----------------------------------------------------------------------------#
def _picks_to_dict(cat: Catalog) -> Dict[Tuple[str, str, str, str], List]:
    out: Dict[Tuple[str, str, str, str], List] = defaultdict(list)
    for pick in cat.picks:
        wf = pick.waveform_id
        key = (
            wf.network_code or "",
            wf.station_code or "",
            wf.location_code or "",
            wf.channel_code or "",
        )
        out[key].append(pick)
    return out


def _manual_pick_parse(xml_file: str) -> Dict[Tuple[str, str, str, str], List]:
    class _MiniPick:
        def __init__(self, t, phase, net, sta, loc, cha):
            self.time = UTCDateTime(t)
            self.phase_hint = phase
            class _WfID:
                pass
            wf = _WfID()
            wf.network_code = net
            wf.station_code = sta
            wf.location_code = loc
            wf.channel_code = cha
            self.waveform_id = wf

    out: Dict[Tuple[str, str, str, str], List] = defaultdict(list)

    it = ET.iterparse(xml_file)
    for _, el in it:
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    root = it.root

    for p in root.iter("pick"):
        t_txt = p.findtext("./time/value")
        if t_txt is None:
            continue
        phase = p.findtext("phaseHint") or ""
        wf = p.find("waveformID")
        net = wf.attrib.get("networkCode", "") if wf is not None else ""
        sta = wf.attrib.get("stationCode", "") if wf is not None else ""
        loc = wf.attrib.get("locationCode", "") if wf is not None else ""
        cha = wf.attrib.get("channelCode", "") if wf is not None else ""

        out[(net, sta, loc, cha)].append(_MiniPick(t_txt, phase, net, sta, loc, cha))
    return out


def load_pick_file(xml_file: str) -> Dict[Tuple[str, str, str, str], List]:
    try:
        cat = read_events(xml_file)
        if len(cat.picks):
            return _picks_to_dict(cat)
    except Exception:
        pass
    return _manual_pick_parse(xml_file)


# -----------------------------------------------------------------------------#
# ---------------------  helper: read origin time -----------------------------#
# -----------------------------------------------------------------------------#
def read_origin_time(origin_xml: str) -> Optional[UTCDateTime]:
    if not os.path.isfile(origin_xml):
        return None
    try:
        cat = read_events(origin_xml)
        for ev in cat:
            if ev.origins:
                return ev.origins[0].time
    except Exception:
        pass
    try:
        it = ET.iterparse(origin_xml)
        for _, el in it:
            if "}" in el.tag:
                el.tag = el.tag.split("}", 1)[1]
        root = it.root
        otime = root.findtext(".//origin/time/value")
        if otime:
            return UTCDateTime(otime)
    except Exception:
        pass
    return None


# -----------------------------------------------------------------------------#
# ----------------------------- single-event plot -----------------------------#
# -----------------------------------------------------------------------------#
def plot_event(
    event_dir: str,
    phasenet_fname: str = "xml/dlpicks_phasenet_instance.xml",
    autopick_glob: str = "xml/*autopicks.xml",
    show: bool = True,
    savefile: Optional[str] = None,
    fig_dpi: int = 200,
) -> None:
    event_dir = os.path.abspath(event_dir)
    evid = os.path.basename(event_dir)
    parent = os.path.dirname(event_dir)

    # ---------- 1. load picks ------------------------------------------------
    picks_ph, picks_auto = {}, {}
    phfile = os.path.join(event_dir, phasenet_fname)
    if os.path.isfile(phfile):
        picks_ph = load_pick_file(phfile)

    auto_xmls = glob.glob(os.path.join(event_dir, autopick_glob))
    if auto_xmls:
        picks_auto = load_pick_file(auto_xmls[0])

    # flatten list of all pick times
    all_picks = [
        pick
        for lst in list(picks_ph.values()) + list(picks_auto.values())
        for pick in lst
    ]
    all_times = [p.time for p in all_picks]

    # ---------- 2. determine time window ------------------------------------
    # 1) try to locate the origin file in this dir or in a parent dir
    origin_xml = _find_in_parents(
        event_dir,
        evid,
        extensions=[".xml"],
        max_levels=5,                      # <-- raise if necessary
    )
    t_origin = read_origin_time(origin_xml) if origin_xml else None
    if t_origin is not None:
        t0, t1 = t_origin, t_origin + 5.0
    elif all_times:
        first_pick = min(all_times)
        t0, t1 = first_pick - 2.0, first_pick + 10.0
    else:
        raise ValueError("no picks and no origin -> skip")

    # ---------- 3. skip event if no pick in window --------------------------
    n_in_window = sum(1 for p in all_picks if t0 <= p.time <= t1)
    if n_in_window == 0:
        raise ValueError("no picks in window")

    # ---------- 4. waveform file --------------------------------------------
    mseed_file = _find_in_parents(
        event_dir,
        evid,
        extensions=[".ms", ".mseed", ".miniseed"],
        max_levels=5,
    )
    if mseed_file is None:
        raise FileNotFoundError(
            f"Could not locate waveform file {evid}.ms/.mseed/.miniseed "
            f"in {event_dir} or any of its parents"
        )

    st = read(mseed_file)
    st.sort()
    st.trim(starttime=t0, endtime=t1, pad=True, fill_value=0.0)

    # ---------- 5. plotting --------------------------------------------------
    ntr = len(st)
    fig, axes = plt.subplots(
        ntr, 1, figsize=(12, 1.6 * ntr), sharex=True, squeeze=False
    )
    axes = axes.ravel()

    for ax, tr in zip(axes, st):
        ax.plot_date(tr.times("matplotlib"), tr.data, "k-", lw=0.6)
        ax.set_ylabel(tr.id, rotation=0, ha="right", va="center", fontsize=8)

        key = (
            tr.stats.network,
            tr.stats.station,
            tr.stats.location,
            tr.stats.channel,
        )

        for pick in picks_ph.get(key, []):
            if not (t0 <= pick.time <= t1):
                continue
            t = pick.time.matplotlib_date
            ax.axvline(t, color="red", lw=1)
            ax.text(
                t,
                ax.get_ylim()[1],
                pick.phase_hint or "",
                color="red",
                fontsize=7,
                rotation=90,
                va="top",
                ha="center",
            )

        for pick in picks_auto.get(key, []):
            if not (t0 <= pick.time <= t1):
                continue
            t = pick.time.matplotlib_date
            ax.axvline(t, color="blue", lw=1, ls="--")
            ax.text(
                t,
                ax.get_ylim()[0],
                pick.phase_hint or "",
                color="blue",
                fontsize=7,
                rotation=90,
                va="bottom",
                ha="center",
            )

    axes[-1].set_xlabel("Time (UTC)")
    handles = [
        plt.Line2D([], [], color="red", lw=1, label="PhaseNet"),
        plt.Line2D([], [], color="blue", lw=1, ls="--", label="Autopick"),
    ]
    fig.legend(handles=handles, loc="upper right")
    suffix = f"origin {t_origin}" if t_origin else f"window {t0}–{t1}"
    fig.suptitle(f"{evid}  ({suffix})")
    fig.autofmt_xdate()

    if savefile:
        fig.savefig(savefile, dpi=fig_dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------#
# ----------------------------- event discovery ------------------------------ #
# -----------------------------------------------------------------------------#
def discover_event_dirs(root: str, recursive: bool = False) -> Iterable[str]:
    root = os.path.abspath(root)

    if recursive:
        walker = os.walk(root)
    else:
        walker = [
            (
                root,
                [
                    d
                    for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d))
                ],
                [],
            )
        ]

    for cur_dir, subdirs, _ in walker:
        for d in subdirs:
            ev_dir = os.path.join(cur_dir, d)
            # --- NEW: does this dir contain *any* known pick file? ----------
            if (
                os.path.isfile(os.path.join(ev_dir, "xml/dlpicks_phasenet_instance.xml"))
                or glob.glob(os.path.join(ev_dir, "xml/*cape_autopicks.xml"))
            ):
                yield ev_dir

# -----------------------------------------------------------------------------#
# ------------------------- bulk processing helper --------------------------- #
# -----------------------------------------------------------------------------#
def plot_all_events(
    root: str,
    recursive: bool = False,
    out_dir: Optional[str] = None,
    overwrite: bool = True,
    show_when_saving: bool = False,
    debug: bool = False,
) -> None:
    root = os.path.abspath(root)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for ev_dir in discover_event_dirs(root, recursive):
        evid = os.path.basename(ev_dir)
        outfile = os.path.join(out_dir, evid + ".png") if out_dir else None
        if outfile and (not overwrite) and os.path.exists(outfile):
            print(f"[skip] {evid} (png exists)")
            continue

        print(f"[plot] {evid}")
        try:
            plot_event(
                ev_dir,
                show=show_when_saving if out_dir else True,
                savefile=outfile,
            )
        except ValueError as ve:
            # benign skip conditions
            print(f"  -> skip ({ve})")
        except Exception as exc:
            print(f"  -> failed: {exc}")
            if debug:
                import traceback
                traceback.print_exc()


# -----------------------------------------------------------------------------#
# ----------------------------------- CLI ------------------------------------ #
# -----------------------------------------------------------------------------#
def _cli() -> None:
    p = argparse.ArgumentParser(description="Plot pick comparison.")
    p.add_argument("root", help="Root directory with waveform files + subdirs")
    p.add_argument("--recursive", action="store_true", help="search recursively")
    p.add_argument(
        "--out", dest="out_dir", default=None,
        help="output directory for PNGs (omit for interactive display)",
    )
    p.add_argument("--no-overwrite", action="store_true", help="skip existing PNGs")
    p.add_argument(
        "--also-show", action="store_true",
        help="when writing PNGs still pop up the figure window",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="print full traceback when an event fails",
    )
    args = p.parse_args()

    plot_all_events(
        root=args.root,
        recursive=args.recursive,
        out_dir=args.out_dir,
        overwrite=not args.no_overwrite,
        show_when_saving=args.also_show,
        debug=args.debug,
    )


if __name__ == "__main__":
    _cli()