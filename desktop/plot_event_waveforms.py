#!/usr/bin/env python3
"""
plot_event_waveforms.py

Per-event waveform + pick plots for CASS-DTS active source shots.

For each event the waveforms are trimmed to:
  start : 2 ms before the first P pick
  end   : CODA_S seconds after the theoretical last S arrival
          (S-P estimated from Vp/Vs applied to each station's P travel time)

Usage:
    python plot_event_waveforms.py [--outdir /tmp/plots]
    python plot_event_waveforms.py --events-dir /tmp --waves-dir /tmp --outdir /tmp/plots
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import obspy
from obspy import UTCDateTime

# ---------------------------------------------------------------------------
# Physical constants / trimming parameters
# ---------------------------------------------------------------------------
VP = 6900.0       # P velocity (m/s)
VS = 3990.0       # S velocity (m/s)
VP_VS = VP / VS   # ~1.73

PRE_ORIGIN_S = 0.002   # seconds of pre-P buffer before origin time
CODA_S       = 0.020   # seconds of coda after theoretical last S arrival


def theoretical_last_s(arrivals, pick_by_id, origin_time):
    """Return UTCDateTime of the latest theoretical S arrival across all P picks."""
    t_last = None
    for arr in arrivals:
        if arr.phase != "P":
            continue
        pk = pick_by_id.get(str(arr.pick_id))
        if pk is None:
            continue
        tp = pk.time - origin_time          # P travel time (s)
        ts = origin_time + tp * VP_VS       # theoretical S = origin + tp * Vp/Vs
        if t_last is None or ts > t_last:
            t_last = ts
    return t_last


def plot_event(ev, st_full, out_path, event_label):
    o = ev.preferred_origin() or ev.origins[0]
    origin_time = o.time

    # Build pick_id → Pick lookup for this event
    pick_by_id = {str(pk.resource_id): pk for pk in ev.picks}

    # Pair arrivals with picks; sort by epicentral distance (nearest first)
    phase_picks = []
    for arr in o.arrivals:
        pk = pick_by_id.get(str(arr.pick_id))
        if pk is None:
            continue
        # distance in degrees → rough metres; only used for trace ordering
        dist_m = (arr.distance or 0.0) * 111320.0
        phase_picks.append((dist_m, arr.phase, pk))
    phase_picks.sort(key=lambda x: x[0])

    if not phase_picks:
        print(f"    {event_label}: no matched arrivals, skipping")
        return

    # Determine trim window
    t_last_s = theoretical_last_s(o.arrivals, pick_by_id, origin_time)
    t_start  = origin_time - PRE_ORIGIN_S
    t_end    = (t_last_s if t_last_s else origin_time) + CODA_S

    # Group by station so we can cross-mark S on P traces and vice-versa
    picks_by_sta = {}
    for _, phase, pk in phase_picks:
        sta = pk.waveform_id.station_code
        picks_by_sta.setdefault(sta, []).append((phase, pk))

    # Build list of (dist_m, phase, pick, trace) — one row per picked channel
    rows = []
    for dist_m, phase, pk in phase_picks:
        wid = pk.waveform_id
        sel = st_full.select(
            network=wid.network_code or "",
            station=wid.station_code or "",
            channel=wid.channel_code or "",
            location=wid.location_code or "",
        )
        if not sel:
            continue
        tr = sel[0].copy()
        tr.trim(t_start, t_end)
        if len(tr.data) == 0:
            continue
        rows.append((dist_m, phase, pk, tr))

    if not rows:
        print(f"    {event_label}: no matching traces in stream, skipping")
        return

    n = len(rows)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(12, max(4, n * 0.75)),
        sharex=True,
        squeeze=False,
    )
    window_ms = (t_end - t_start) * 1000.0
    fig.suptitle(
        f"{event_label}  |  {origin_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]} UTC  |  "
        f"elev {-o.depth:.0f} m  |  {len(o.arrivals)} arrivals  |  "
        f"window {window_ms:.0f} ms",
        fontsize=8,
    )

    for row, (dist_m, phase, pk, tr) in enumerate(rows):
        ax = axes[row, 0]

        # x-axis: ms relative to origin time
        x_ms = (tr.times("timestamp") - float(origin_time)) * 1000.0
        data = tr.data.astype(float)
        amp = np.max(np.abs(data))
        if amp > 0:
            data = data / amp

        ax.plot(x_ms, data, color="0.2", lw=0.4, rasterized=True)

        # Origin time marker (t = 0)
        ax.axvline(0.0, color="0.6", lw=0.6, ls="--", zorder=2)

        # Main pick for this row
        pk_ms = (pk.time - origin_time) * 1000.0
        color = "red" if phase == "P" else "blue"
        ax.axvline(pk_ms, color=color, lw=1.0, zorder=3)

        # Cross-phase picks on the same station (dotted)
        for ph2, pk2 in picks_by_sta.get(pk.waveform_id.station_code, []):
            if str(pk2.resource_id) == str(pk.resource_id):
                continue
            c2 = "blue" if ph2 == "S" else "red"
            ax.axvline((pk2.time - origin_time) * 1000.0,
                       color=c2, lw=0.8, ls=":", zorder=3)

        sta = pk.waveform_id.station_code
        cha = pk.waveform_id.channel_code
        res = next(
            (arr.time_residual for arr in o.arrivals
             if str(arr.pick_id) == str(pk.resource_id)),
            None,
        )
        res_str = f"  res={res*1000:.2f}ms" if res is not None else ""
        ax.set_ylabel(
            f"{sta}.{cha}\n{dist_m:.0f} m{res_str}",
            fontsize=6,
            rotation=0,
            labelpad=65,
            va="center",
        )
        ax.set_ylim(-1.3, 1.3)
        ax.tick_params(axis="y", labelleft=False, left=False)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        if row < n - 1:
            ax.tick_params(axis="x", labelbottom=False)

    axes[-1, 0].set_xlabel("Time relative to origin (ms)", fontsize=8)

    # Legend
    for label, color, ls in [
        ("P pick", "red", "-"),
        ("S pick", "blue", "-"),
        ("origin", "0.6", "--"),
    ]:
        axes[0, 0].axvline(np.nan, color=color, lw=1.0, ls=ls, label=label)
    axes[0, 0].legend(fontsize=6, loc="upper right", framealpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir",     default="/tmp/plots",
                    help="Output directory for PNG files")
    ap.add_argument("--events-dir", default="/tmp",
                    help="Directory containing pbevents_w*.xml")
    ap.add_argument("--waves-dir",  default="/tmp",
                    help="Directory containing shots_w*.mseed")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for w in [1, 2, 3]:
        ev_file = Path(args.events_dir) / f"pbevents_w{w}.xml"
        ms_file = Path(args.waves_dir)  / f"shots_w{w}.mseed"
        if not ev_file.exists():
            print(f"w{w}: {ev_file} not found, skipping")
            continue
        if not ms_file.exists():
            print(f"w{w}: {ms_file} not found, skipping")
            continue

        print(f"\nw{w}: loading waveforms ({ms_file.stat().st_size // 1_000_000} MB)…")
        st  = obspy.read(str(ms_file))
        cat = obspy.read_events(str(ev_file))
        print(f"w{w}: {len(cat)} events, {len(st)} traces")

        for i, ev in enumerate(cat):
            label = f"w{w}_ev{i+1:02d}"
            out_path = outdir / f"{label}_waveforms.png"
            print(f"  {label}:")
            plot_event(ev, st, out_path, label)

    print("\nDone.")


if __name__ == "__main__":
    main()
