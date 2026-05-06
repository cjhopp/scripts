#!/usr/bin/env bash
# =============================================================================
# cass_playback.sh  —  Full CASS shot-window playback pipeline
# =============================================================================
# Runs scautopick (all 27 aliases in parallel) on each shot window mseed,
# merges picks, runs scanloc + scevent, and reports event counts.
#
# Prerequisites:
#   /tmp/shots_w{1,2,3}.mseed        — raw waveforms (from DUGseis acquisition)
#   /tmp/inventory.xml               — station inventory
#   /tmp/aliases.txt                 — list of scautopick alias names (one per line)
#                                      regenerate with:
#   sudo -u sysop bash -c 'ls /home/sysop/seiscomp/etc/scautopick_*.cfg' \
#     | xargs -I{} basename {} .cfg | grep -v CTrig | sort > /tmp/aliases.txt
#
# Outputs (per window W in w1 w2 w3):
#   /tmp/shots_${W}_trim.mseed — trimmed to shot window only (avoids spurious events)
#   /tmp/picks_s_${W}/  — per-alias pick XMLs + logs
#   /tmp/fpicks_${W}.xml — merged picks (P+S, all phases passed to scanloc)
#   /tmp/pborigins_${W}.xml
#   /tmp/pbevents_${W}.xml  (split from combined)
#   /tmp/pbevents_all.xml   (combined, used for plots)
#   /tmp/plots/          — waveform + location plots (via plot_event_*.py)
#
# Pipeline structure (two phases):
#   Phase 1 (per window): trim → scautopick → merge picks → scanloc
#   Phase 2 (combined):   merge origins → scamp → scmag → scevent (once)
#                         Scamp's station-response cache is populated once
#                         and reused for all three windows (~9 s startup
#                         vs ~55 s if called per-window).
# =============================================================================

set -euo pipefail

SEISCOMP_EXEC="sudo -u sysop /home/sysop/seiscomp/bin/seiscomp exec"
SC_NS="http://geofon.gfz.de/ns/seiscomp-schema/0.14"
SCEVENT_CFG=/tmp/scevent_tmp.cfg
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_DIR=/tmp/plots

# scevent_tmp.cfg — looser association for shot clusters
if [[ ! -f "$SCEVENT_CFG" ]]; then
  cat > "$SCEVENT_CFG" <<EOF
eventIDPrefix = lbnl
eventAssociation.maximumTimeSpan = 0.5
eventAssociation.minimumDefiningPhases = 7
EOF
fi

mkdir -p "$PLOT_DIR"

# ── Step 0: Temporarily tighten scanloc pickKeep for playback ──────────────
echo "=== Patching scanloc.cfg (pickKeep 2.0→0.5) ==="
sudo sed -i 's/buffer\.pickKeep\s*=.*/buffer.pickKeep   = 0.5    # s/' \
    /home/sysop/seiscomp/etc/scanloc.cfg

cleanup() {
  echo "=== Restoring scanloc.cfg (pickKeep 0.5→2.0) ==="
  sudo sed -i 's/buffer\.pickKeep\s*=.*/buffer.pickKeep   = 2.0    # s/' \
      /home/sysop/seiscomp/etc/scanloc.cfg
  echo "buffer.pickKeep restored"
}
trap cleanup EXIT

# Shot window trim times — restrict scautopick to actual shot intervals only.
# Pre-shot and post-shot noise in the mseeds causes spurious events otherwise.
declare -A WIN_BEGIN WIN_END
WIN_BEGIN[w1]="2026-05-01T19:22:25" ; WIN_END[w1]="2026-05-01T19:22:29.5"
WIN_BEGIN[w2]="2026-05-01T19:33:36" ; WIN_END[w2]="2026-05-01T19:33:44"
WIN_BEGIN[w3]="2026-05-01T19:54:50" ; WIN_END[w3]="2026-05-01T19:55:00"

# ── Main loop ───────────────────────────────────────────────────────────────
for W in w1 w2 w3; do
  MSEED_FULL="/tmp/shots_${W}.mseed"
  MSEED="/tmp/shots_${W}_trim.mseed"
  PICKDIR="/tmp/picks_s_${W}"
  mkdir -p "$PICKDIR"

  echo ""
  echo "===== Window ${W} ($(date -u +%T)) ====="

  # Step 0: Trim mseed to shot window only
  echo "  [0/4] Trimming mseed to ${WIN_BEGIN[$W]} → ${WIN_END[$W]}…"
  python3 - <<PYEOF
import obspy
st = obspy.read('${MSEED_FULL}')
st.trim(obspy.UTCDateTime('${WIN_BEGIN[$W]}'), obspy.UTCDateTime('${WIN_END[$W]}'))
st.write('${MSEED}', format='MSEED')
print(f'  Trimmed: {len(st)} traces, {st[0].stats.starttime} → {st[0].stats.endtime}')
PYEOF

  # Step 1: Run all aliases in parallel
  echo "  [1/4] scautopick ($(wc -l < /tmp/aliases.txt) aliases in parallel)…"
  PIDS=()
  while IFS= read -r alias; do
    $SEISCOMP_EXEC "$alias" \
        -d localhost --ep --playback \
        -I "file://${MSEED}" \
        > "${PICKDIR}/${alias}.xml" \
        2>"${PICKDIR}/${alias}.log" &
    PIDS+=($!)
  done < /tmp/aliases.txt
  for pid in "${PIDS[@]}"; do wait "$pid"; done

  # Count raw picks
  PPICKS=$(grep -h 'phaseHint>P' "${PICKDIR}"/*.xml 2>/dev/null | wc -l || echo 0)
  SPICKS=$(grep -h 'phaseHint>S' "${PICKDIR}"/*.xml 2>/dev/null | wc -l || echo 0)
  echo "  Raw picks: P=${PPICKS}  S=${SPICKS}"

  # Step 2: Merge all per-alias XMLs — all phases (P+S) passed to scanloc
  echo "  [2/4] Merging picks → /tmp/fpicks_${W}.xml…"
  python3 - <<PYEOF
import xml.etree.ElementTree as ET, glob, sys

SC_NS = "${SC_NS}"
N = lambda t: '{' + SC_NS + '}' + t

root_out = ET.Element(N('seiscomp'))
ep_out   = ET.SubElement(root_out, N('EventParameters'))

seen = set()
nP = nS = 0
for f in sorted(glob.glob('${PICKDIR}/*.xml')):
    try:
        root = ET.parse(f).getroot()
        ep   = root.find(N('EventParameters'))
        if ep is None:
            continue
        for pick in ep.findall(N('pick')):
            pid = pick.get('publicID', '')
            if pid in seen:
                continue
            seen.add(pid)
            ep_out.append(pick)
            ph_el = pick.find(N('phaseHint'))
            ph = ph_el.text.strip() if ph_el is not None and ph_el.text else 'P'
            if ph.startswith('S'): nS += 1
            else: nP += 1
    except Exception as e:
        print(f'  WARN: {f}: {e}', file=sys.stderr)

ET.ElementTree(root_out).write(
    '/tmp/fpicks_${W}.xml',
    xml_declaration=True, encoding='UTF-8')
print(f'  Merged P={nP} S={nS} → /tmp/fpicks_${W}.xml')
PYEOF

  # Step 3: scanloc
  echo "  [3/4] scanloc… ($(date -u +%T))"
  $SEISCOMP_EXEC scanloc \
      -d localhost \
      --ep "/tmp/fpicks_${W}.xml" \
      > "/tmp/pborigins_${W}.xml" \
      2>"/tmp/fsscanloc_${W}.log"
  NORIGS=$(grep -o '<origin ' "/tmp/pborigins_${W}.xml" 2>/dev/null | wc -l || echo 0)
  echo "  Origins: ${NORIGS}  ($(date -u +%T))"

done  # ── end of Phase 1 per-window loop (trim/pick/scanloc)

# =============================================================================
# Phase 2: merge all windows and run scamp + scmag + scevent in a single pass.
# Scamp fetches station responses from the DB once per stream and caches them
# for the lifetime of the process, so a single invocation pays the ~9 s
# warm-up cost once instead of once per window.
# =============================================================================
echo ""
echo "===== Phase 2: scamp / scmag / scevent (combined, $(date -u +%T)) ====="

# ── Merge per-window origins XMLs into one EP XML ────────────────────────────
echo "  Merging pborigins_w*.xml → /tmp/pborigins_all.xml…"
python3 - <<'PYEOF'
import xml.etree.ElementTree as ET, sys
SC_NS = "http://geofon.gfz.de/ns/seiscomp-schema/0.14"
N = lambda t: '{' + SC_NS + '}' + t
root_out = ET.Element(N('seiscomp'))
ep_out   = ET.SubElement(root_out, N('EventParameters'))
seen = set()
total = 0
for w in ('w1', 'w2', 'w3'):
    try:
        root = ET.parse(f'/tmp/pborigins_{w}.xml').getroot()
        ep   = root.find(N('EventParameters'))
        if ep is None:
            continue
        for child in list(ep):
            pid = child.get('publicID', id(child))
            if pid in seen:
                continue
            seen.add(pid)
            ep_out.append(child)
            total += 1
    except Exception as e:
        print(f'  WARN {w}: {e}', file=sys.stderr)
ET.ElementTree(root_out).write(
    '/tmp/pborigins_all.xml', xml_declaration=True, encoding='UTF-8')
print(f'  Merged {total} objects into /tmp/pborigins_all.xml')
PYEOF

# ── Concatenate trimmed MSEEDs (MSEED is record-based; binary concat is valid) ──
echo "  Concatenating trimmed MSEEDs → /tmp/shots_all_trim.mseed…"
cat /tmp/shots_w1_trim.mseed \
    /tmp/shots_w2_trim.mseed \
    /tmp/shots_w3_trim.mseed \
    > /tmp/shots_all_trim.mseed

# ── scamp — single invocation over all three windows ────────────────────────
echo "  [scamp] MLv amplitudes (all windows)… ($(date -u +%T))"
$SEISCOMP_EXEC scamp \
    -d localhost \
    --ep "/tmp/pborigins_all.xml" \
    -I "file:///tmp/shots_all_trim.mseed" \
    > "/tmp/pbamp_all.xml" \
    2>"/tmp/pbamp_all.log"
NAMPS=$(grep -o '<amplitude ' "/tmp/pbamp_all.xml" 2>/dev/null | wc -l || echo 0)
echo "  Amplitudes: ${NAMPS}  ($(date -u +%T))"

# ── scmag — single invocation ────────────────────────────────────────────────
echo "  [scmag] MLv magnitudes (all windows)… ($(date -u +%T))"
$SEISCOMP_EXEC scmag \
    -d localhost \
    --ep "/tmp/pbamp_all.xml" \
    > "/tmp/pbmag_all.xml" \
    2>"/tmp/pbmag_all.log"

# ── scevent — single invocation ──────────────────────────────────────────────
echo "  [scevent] event association (all windows)… ($(date -u +%T))"
$SEISCOMP_EXEC scevent \
    --ep "/tmp/pbmag_all.xml" \
    --config-file "$SCEVENT_CFG" \
    > "/tmp/pbevents_all.xml" \
    2>/dev/null

# ── Split combined output back into per-window XMLs (for plots / compatibility) ──
echo "  Splitting pbevents_all.xml → pbevents_w*.xml…"
python3 - <<'PYEOF'
import xml.etree.ElementTree as ET, sys
from datetime import datetime

SC_NS = "http://geofon.gfz.de/ns/seiscomp-schema/0.14"
N = lambda t: '{' + SC_NS + '}' + t

root  = ET.parse('/tmp/pbevents_all.xml').getroot()
ep    = root.find(N('EventParameters'))

# Build lookup tables from the combined EP
origs    = {o.get('publicID'): o for o in ep.iter(N('origin'))}
netmags  = {m.get('publicID'): m for m in ep.iter(N('magnitude'))}
stamags  = {m.get('publicID'): m for m in ep.iter(N('stationMagnitude'))}
amps     = {a.get('publicID'): a for a in ep.iter(N('amplitude'))}
picks    = {p.get('publicID'): p for p in ep.iter(N('pick'))}

windows = {
    'w1': ('2026-05-01T19:22:25',   '2026-05-01T19:22:29.5'),
    'w2': ('2026-05-01T19:33:36',   '2026-05-01T19:33:44'),
    'w3': ('2026-05-01T19:54:50',   '2026-05-01T19:55:00'),
}

# Strip timezone so all datetimes are naive UTC (avoids TypeError when
# comparing offset-naive window bounds against offset-aware origin times)
def parse_iso(s):
    return datetime.fromisoformat(s.replace('Z', '').replace('+00:00', ''))

for w, (t0s, t1s) in windows.items():
    t0, t1 = parse_iso(t0s), parse_iso(t1s)
    root_w = ET.Element(N('seiscomp'))
    ep_w   = ET.SubElement(root_w, N('EventParameters'))
    nevt = 0
    for evt in ep.findall(N('event')):
        oid_el = evt.find(N('preferredOriginID'))
        if oid_el is None:
            continue
        o = origs.get(oid_el.text)
        if o is None:
            continue
        v_el = o.find(N('time'))
        if v_el is None:
            continue
        v_el = v_el.find(N('value'))
        if v_el is None:
            continue
        ot = parse_iso(v_el.text)
        if not (t0 <= ot <= t1):
            continue
        # Add event + its preferred origin
        ep_w.append(evt)
        ep_w.append(o)
        # Add preferred magnitude
        mid_el = evt.find(N('preferredMagnitudeID'))
        if mid_el is not None and mid_el.text in netmags:
            ep_w.append(netmags[mid_el.text])
        nevt += 1
    ET.ElementTree(root_w).write(
        f'/tmp/pbevents_{w}.xml', xml_declaration=True, encoding='UTF-8')
    print(f'  {w}: {nevt} events → /tmp/pbevents_{w}.xml')
PYEOF

# ── Per-window event report ───────────────────────────────────────────────────
echo ""
python3 - <<'PYEOF'
import xml.etree.ElementTree as ET
from datetime import datetime

SC_NS = "http://geofon.gfz.de/ns/seiscomp-schema/0.14"
N = lambda t: '{' + SC_NS + '}' + t

root    = ET.parse('/tmp/pbevents_all.xml').getroot()
ep      = root.find(N('EventParameters'))
origs   = {o.get('publicID'): o for o in ep.iter(N('origin'))}
netmags = {m.get('publicID'): m for m in ep.iter(N('magnitude'))}

windows = {
    'w1': ('2026-05-01T19:22:25',   '2026-05-01T19:22:29.5'),
    'w2': ('2026-05-01T19:33:36',   '2026-05-01T19:33:44'),
    'w3': ('2026-05-01T19:54:50',   '2026-05-01T19:55:00'),
}

# Strip timezone so all datetimes are naive UTC
def parse_iso(s):
    return datetime.fromisoformat(s.replace('Z', '').replace('+00:00', ''))

for w, (t0s, t1s) in windows.items():
    t0, t1 = parse_iso(t0s), parse_iso(t1s)
    evts_w = []
    for evt in ep.findall(N('event')):
        oid_el = evt.find(N('preferredOriginID'))
        if not oid_el:
            continue
        o = origs.get(oid_el.text)
        if not o:
            continue
        v_el = o.find(N('time'))
        if not v_el:
            continue
        v_el = v_el.find(N('value'))
        if not v_el:
            continue
        ot = parse_iso(v_el.text)
        if t0 <= ot <= t1:
            evts_w.append((evt, o))
    print(f'  ===== Window {w} =====')
    for evt, o in evts_w:
        ot_str = o.find(N('time')).find(N('value')).text[11:21]
        mid_el = evt.find(N('preferredMagnitudeID'))
        m = netmags.get(mid_el.text) if mid_el is not None else None
        if m is not None:
            mv  = m.find(N('magnitude')).find(N('value'))
            mtp = m.find(N('type'))
            mval = f'{float(mv.text):.2f}' if mv is not None else '?'
            mtyp = mtp.text if mtp is not None else '?'
            print(f'    event {ot_str}  {mtyp} = {mval}  (uncalibrated)')
        else:
            print(f'    event {ot_str}  no magnitude')
    print(f'  {w}: {len(evts_w)} events')
PYEOF

# ── Plots ──────────────────────────────────────────────────────────────────
echo ""
echo "===== Generating plots ====="
python3 "${SCRIPT_DIR}/../desktop/plot_event_waveforms.py" \
    --events-dir /tmp --waves-dir /tmp --outdir "$PLOT_DIR" 2>&1 | grep -E 'Saved|Error|event'
python3 "${SCRIPT_DIR}/../desktop/plot_event_locations.py" \
    --outdir "$PLOT_DIR" 2>&1 | grep -E 'Saved|Error|hmc_x|W '

echo ""
echo "===== Done. Outputs in /tmp/pbevents_w*.xml and ${PLOT_DIR}/ ====="
