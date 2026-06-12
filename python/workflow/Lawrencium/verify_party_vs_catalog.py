#!/usr/bin/env python
"""
Compare Party file vs XML catalog vs plot_smackover_detections.py results.
"""

import logging
import pandas as pd
from obspy import read_events
from eqcorrscan import Party

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PARTY_PATH = (
    "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
    "/Smackover_analyzed_raw.tgz"
)
CATALOG_PATH = (
    "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
    "/Smackover_analyzed_raw.xml"
)

MIN_CHANS = 3
TEMPLATE_EXCLUSIONS = [
    "us2000h85v", "tx2025qlwgec", "us70003tzm", "us6000e1q3",
    "tx2023zock", "tx2024ywip", "tx2024zbdb", "tx2024zocv",
    "tx2024yvww", "us6000pi49", "us70008ee1",
]
SPIKE_DAY_EXCLUSIONS = {
    "nm60081223": ["2015-03-08", "2013-02-14"],
    "nm60351847": ["2015-03-08", "2013-02-14", "2013-02-07"],
    "nm60120628": ["2023-06-26", "2023-08-08"],
    "nm60080523": ["2012-12-25", "2013-02-07", "2013-02-14", "2020-09-14"],
    "us70003tzm": ["2012-12-25", "2013-02-07"],
    "us7000rfpr": ["2012-12-25"],
    "us6000m33c": ["2012-12-25"],
    "us6000pkzk": ["2012-12-25", "2013-02-07"],
    "nm60163943": ["2020-09-14"],
    "tx2024ywip": ["2012-12-25"],
    "us70008ee1": ["2012-12-25"],
    "us6000e1z3": ["2013-02-07"],
    "us6000dy5c": ["2013-02-07"],
}

# ─ Load Party ─────────────────────────────────────────────────────────────
log.info("=" * 80)
log.info("LOADING PARTY")
log.info("=" * 80)
party = Party().read(PARTY_PATH, read_detection_catalog=False)
log.info(f"Families in party: {len(party.families)}")

records = []
for fam in party.families:
    tmpl_name = fam.template.name
    for d in fam.detections:
        records.append({
            "template_name": tmpl_name,
            "detect_time": d.detect_time.datetime,
            "detect_val": d.detect_val,
            "no_chans": d.no_chans,
            "threshold": d.threshold,
        })

df_party = pd.DataFrame(records)
df_party["detect_time"] = pd.to_datetime(df_party["detect_time"], utc=True)
log.info(f"Total detections in Party: {len(df_party)}")
log.info(f"no_chans distribution: min={df_party['no_chans'].min()}, max={df_party['no_chans'].max()}, median={df_party['no_chans'].median()}")

# ─ Apply filters (plot_smackover_detections.py logic) ─────────────────────
log.info("\n" + "=" * 80)
log.info("FILTERING PARTY (plot_smackover_detections.py logic)")
log.info("=" * 80)

n_before = len(df_party)
df_party = df_party[~df_party["template_name"].isin(TEMPLATE_EXCLUSIONS)]
log.info(f"After template exclusions: {len(df_party)} (dropped {n_before - len(df_party)})")

n_before = len(df_party)
date_col = df_party["detect_time"].dt.tz_convert(None).dt.normalize()
mask_keep = pd.Series(True, index=df_party.index)
for tmpl, bad_days in SPIKE_DAY_EXCLUSIONS.items():
    bad_ts = pd.to_datetime(bad_days).normalize()
    in_tmpl = df_party["template_name"] == tmpl
    on_bad_day = date_col.isin(bad_ts)
    mask_keep &= ~(in_tmpl & on_bad_day)
df_party = df_party[mask_keep]
log.info(f"After spike-day exclusions: {len(df_party)} (dropped {n_before - len(df_party)})")

n_before = len(df_party)
df_party = df_party[df_party["no_chans"] >= MIN_CHANS]
log.info(f"After MIN_CHANS >= {MIN_CHANS}: {len(df_party)} (dropped {n_before - len(df_party)})")

# ─ Load XML Catalog ────────────────────────────────────────────────────────
log.info("\n" + "=" * 80)
log.info("LOADING XML CATALOG")
log.info("=" * 80)

catalog = read_events(CATALOG_PATH)
log.info(f"Total events in catalog: {len(catalog)}")

# Check pick distribution
pick_counts = [len(e.picks) if e.picks else 0 for e in catalog]
log.info(f"Pick distribution: min={min(pick_counts)}, max={max(pick_counts)}, median={int(__import__('numpy').median(pick_counts))}")

# ─ Summary ─────────────────────────────────────────────────────────────────
log.info("\n" + "=" * 80)
log.info("SUMMARY")
log.info("=" * 80)
log.info(f"Party (filtered):     {len(df_party)} detections")
log.info(f"XML Catalog (raw):    {len(catalog)} events")
log.info(f"Match: {len(df_party) == len(catalog)}")

log.info(f"\nExpected from plot (650 filtered): {'✓ Party matches!' if len(df_party) >= 600 else '✗ Party differs'}")
log.info(f"Expected from generate_hypoDD (184 from catalog): {'✓' if len(catalog) >= 180 else '✗'}")

# Dig deeper: XML catalog pick distribution
log.info(f"\nXML catalog events with 3+ picks: {sum(1 for c in pick_counts if c >= 3)}")
