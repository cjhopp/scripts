#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
from obspy import read, read_events, UTCDateTime
from collections import defaultdict


def parse_picks_from_xml(pick_xml_path):
    # Use lxml for namespace support
    from lxml import etree
    picks = []
    tree = etree.parse(pick_xml_path)
    ns = {'sc': 'http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.13'}
    for pick in tree.xpath('//sc:pick', namespaces=ns):
        time_str = pick.find('sc:time/sc:value', namespaces=ns).text
        station = pick.find('sc:waveformID', namespaces=ns).get('stationCode')
        channel = pick.find('sc:waveformID', namespaces=ns).get('channelCode')
        phase = pick.find('sc:phaseHint', namespaces=ns).text
        t = UTCDateTime(time_str)
        picks.append({
            'time': t,
            'station': station,
            'channel': channel,
            'phase': phase,
        })
    return picks


def plot_event_waveforms_with_picks_singletrace(
    event_xml_path,
    waveform_path,
    picks_ml_path,
    picks_auto_path,
    plot_dir,
    window_before=1.0,
    window_after=2.0
):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    import matplotlib.dates as mdates

    # 1. Read event and get arrivals
    event = read_events(event_xml_path)[0]
    origin = event.preferred_origin() or event.origins[0]

    # Group arrivals by station/channel
    arrivals_by_sta_cha = defaultdict(list)
    for arrival in origin.arrivals:
        pick = arrival.pick_id.get_referred_object()
        if pick is None:
            continue
        station = pick.waveform_id.station_code
        channel = pick.waveform_id.channel_code
        arrivals_by_sta_cha[(station, channel)].append(pick.time)

    if not arrivals_by_sta_cha:
        raise ValueError("No arrivals found in event XML.")

    # 2. Parse picks directly from XML
    ml_picks = parse_picks_from_xml(picks_ml_path)
    auto_picks = parse_picks_from_xml(picks_auto_path)

    # 3. Read waveform
    st = read(waveform_path)

    for tr in st:
        station = tr.stats.station
        channel = tr.stats.channel
        key = (station, channel)
        if key not in arrivals_by_sta_cha:
            continue

        # Get time window for this trace
        pick_times = arrivals_by_sta_cha[key]
        tmin = min(pick_times) - window_before
        tmax = max(pick_times) + window_after

        tr_trim = tr.copy().trim(tmin, tmax)
        times = tr_trim.times("utcdatetime")

        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot([ti.datetime for ti in times], tr_trim.data, label=f"{station}.{channel}", linewidth=0.75, color='k')

        legend_handles = [Line2D([0], [0], color='k', label=f"{station}.{channel}")]

        # Overlay ML picks for this trace
        for pick in ml_picks:
            if pick['station'] == station and pick['channel'] == channel:
                if tmin <= pick['time'] <= tmax:
                    label = f"ML Pick ({pick['phase']})"
                    ax.axvline(
                        pick['time'].datetime, ymin=0, ymax=1, linewidth=0.75,
                        color='r', alpha=0.5, label=label
                    )
                    legend_handles.append(Line2D([0], [0], color='r', linewidth=0.75, alpha=0.5, label=label))

        # Overlay Auto picks for this trace
        for pick in auto_picks:
            if pick['station'] == station and pick['channel'] == channel:
                if tmin <= pick['time'] <= tmax:
                    label = f"Auto Pick ({pick['phase']})"
                    ax.axvline(
                        pick['time'].datetime, ymin=0, ymax=1, linewidth=0.75,
                        color='b', alpha=0.5, label=label
                    )
                    legend_handles.append(Line2D([0], [0], color='b', linewidth=0.75, alpha=0.5, label=label))

        ax.set_ylabel(f"{station}.{channel}")
        ax.set_xlabel("Time (UTC)")
        ax.set_title(f"{station}.{channel}")

        # Format x-axis datetime with 4 decimal places for seconds
        def format_datetime(x, pos=None):
            dt = mdates.num2date(x)
            return dt.strftime('%H:%M:%S.') + f"{dt.microsecond/1e6:.3f}"[2:6]

        ax.xaxis.set_major_formatter(FuncFormatter(format_datetime))

        # Remove duplicate legend entries
        seen = set()
        unique_handles = []
        for h in legend_handles:
            if h.get_label() not in seen:
                unique_handles.append(h)
                seen.add(h.get_label())
        ax.legend(handles=unique_handles, loc='upper right', fontsize=8)
        plt.tight_layout()

        # --- Save the figure ---
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{station}.{channel}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    # === SET YOUR ROOT DIRECTORY HERE ===
    root_dir = "/media/chopp/HDD1/chet-meq/cape_modern/seiscomp_output/dlpick_testing/event-wise_test"

    for subdir in sorted(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        event_xml_path = os.path.join(subdir_path, f"{subdir}.xml")
        waveform_path = os.path.join(subdir_path, f"{subdir}.ms")
        picks_ml_path = os.path.join(subdir_path, "tmp/picks-merged.xml")
        picks_auto_path = os.path.join(subdir_path, "auto-picks.xml")
        plot_dir = os.path.join(subdir_path, "plots")

        # Check if required files exist
        if not (os.path.exists(event_xml_path) and os.path.exists(waveform_path) and
                os.path.exists(picks_ml_path) and os.path.exists(picks_auto_path)):
            print(f"Skipping {subdir}: missing required files.")
            continue

        print(f"Processing {subdir}...")
        plot_event_waveforms_with_picks_singletrace(
            event_xml_path,
            waveform_path,
            picks_ml_path,
            picks_auto_path,
            plot_dir=plot_dir,
            window_before=1.0,
            window_after=2.0
        )