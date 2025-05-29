#!/usr/bin/env python3
"""
Quick-look comparison plots:
AI (PhaseNet) picks vs. standard autopicks for *all* events
found beneath a given root folder.

Directory layout expected:

    ROOT/
        lbnl2024eroy.ms               ← waveform file (parent folder)
        lbnl2024eroy/                 ← event directory with picks
            └── xml/
                  ├─ dlpicks_phasenet_instance.xml
                  └─ ...autopicks.xml
        20240501_1234.mseed
        20240501_1234/
            └── xml/…

A figure lbnl2024eroy.png is produced for each event.
"""

from __future__ import annotations
import argparse
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Optional

import matplotlib.pyplot as plt
from obspy import read, read_events, Catalog

# -----------------------------------------------------------------------------#
# -------------------------- helper: picks to dict ----------------------------#
# -----------------------------------------------------------------------------#
def _picks_to_dict(cat: Optional[Catalog]) -> Dict[Tuple[str, str, str, str], List]:
    """
    Convert an ObsPy Catalog into a dictionary keyed by
    (net, sta, loc, cha) for fast lookup when plotting.
    """
    out: Dict[Tuple[str, str, str, str], List] = defaultdict(list)
    if cat is None:
        return out

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


# -----------------------------------------------------------------------------#
# ----------------------------- single event plot -----------------------------#
# -----------------------------------------------------------------------------#
def plot_event(
    event_dir: str,
    phasenet_fname: str = "xml/dlpicks_phasenet_instance.xml",
    autopick_glob: str = "xml/*autopicks.xml",
    show: bool = True,
    savefile: Optional[str] = None,
    fig_dpi: int = 200,
) -> None:
    """
    Produce one figure for *event_dir*.

    The waveform file is searched in the *parent* directory and has to be
    called  <event_name>.{ms|mseed|miniseed}.
    """
    event_dir = os.path.abspath(event_dir)
    evid = os.path.basename(event_dir)
    parent = os.path.dirname(event_dir)

    # 1 — waveform file --------------------------------------------------------
    mseed_file = None
    for ext in (".ms", ".mseed", ".miniseed"):
        cand = os.path.join(parent, evid + ext)
        if os.path.isfile(cand):
            mseed_file = cand
            break
    if mseed_file is None:
        raise FileNotFoundError(
            f"Waveform file {evid}.(ms|mseed|miniseed) not found in {parent}"
        )

    st = read(mseed_file)
    st.sort(keys=("network", "station", "channel"))

    # 2 — pick files -----------------------------------------------------------
    ph_xml = os.path.join(event_dir, phasenet_fname)
    auto_xmls = glob.glob(os.path.join(event_dir, autopick_glob))

    cat_ph = read_events(ph_xml) if os.path.isfile(ph_xml) else None
    cat_auto = read_events(auto_xmls[0]) if auto_xmls else None

    picks_ph = _picks_to_dict(cat_ph)
    picks_auto = _picks_to_dict(cat_auto)

    # 3 — plotting -------------------------------------------------------------
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

        # PhaseNet picks
        for pick in picks_ph.get(key, []):
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

        # Autopicks
        for pick in picks_auto.get(key, []):
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
    fig.suptitle(evid)
    fig.autofmt_xdate()

    if savefile:
        fig.savefig(savefile, dpi=fig_dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------#
# ---------------------------- event discovery --------------------------------#
# -----------------------------------------------------------------------------#
def discover_event_dirs(root: str, recursive: bool = False) -> Iterable[str]:
    """
    Yield paths to event directories.

    Criterion:
        • path is a directory
        • its parent folder contains a waveform file called
          <dirname>.{ms|mseed|miniseed}
    """
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
            parent = os.path.dirname(ev_dir)
            # look for waveform file in parent directory
            for ext in (".ms", ".mseed", ".miniseed"):
                if os.path.isfile(os.path.join(parent, d + ext)):
                    yield ev_dir
                    break


# -----------------------------------------------------------------------------#
# ------------------------- bulk processing helper ----------------------------#
# -----------------------------------------------------------------------------#
def plot_all_events(
    root: str,
    recursive: bool = False,
    out_dir: Optional[str] = None,
    overwrite: bool = True,
    show_when_saving: bool = False,
) -> None:
    """
    Loop through all event directories beneath *root* and plot them.
    """
    root = os.path.abspath(root)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for ev_dir in discover_event_dirs(root, recursive=recursive):
        evid = os.path.basename(ev_dir)
        outfile = os.path.join(out_dir, evid + ".png") if out_dir else None

        if outfile and (not overwrite) and os.path.exists(outfile):
            print(f"[skip] {evid} (figure exists)")
            continue

        print(f"[plot] {evid}")
        try:
            plot_event(
                ev_dir,
                show=show_when_saving if out_dir else True,
                savefile=outfile,
            )
        except Exception as exc:
            print(f"  -> failed: {exc}")


# -----------------------------------------------------------------------------#
# ------------------------------- CLI -----------------------------------------#
# -----------------------------------------------------------------------------#
def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Plot PhaseNet vs. autopicks for many events."
    )
    p.add_argument("root", help="Root directory containing waveform files and pick sub-folders")
    p.add_argument(
        "--recursive", action="store_true", help="search for events recursively"
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        default=None,
        help="output directory for PNGs (omit for interactive display)",
    )
    p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="skip an event when the PNG already exists",
    )
    p.add_argument(
        "--also-show",
        action="store_true",
        help="if --out is used, still pop up figure windows",
    )
    args = p.parse_args()

    plot_all_events(
        root=args.root,
        recursive=args.recursive,
        out_dir=args.out_dir,
        overwrite=not args.no_overwrite,
        show_when_saving=args.also_show,
    )


if __name__ == "__main__":
    _cli()