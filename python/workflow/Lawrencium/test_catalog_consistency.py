#!/usr/bin/env python
"""
Quick test to verify the XML catalog has the right structure
and filtering logic matches plot_smackover_detections.py
"""

import logging
from obspy import read_events

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CATALOG_PATH = (
    "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
    "/Smackover_analyzed_raw.xml"
)

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

MIN_CHANS = 3

def extract_template_name(event):
    """Extract template name from resource_id."""
    try:
        event_id_full = str(event.resource_id.id)
        if "/" in event_id_full:
            event_id_short = event_id_full.split("/", 1)[1]
        else:
            event_id_short = event_id_full
        
        if "_" not in event_id_short:
            return ""
        
        parts = event_id_short.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0]
    except Exception:
        pass
    return ""


# Load catalog
log.info(f"Loading catalog from: {CATALOG_PATH}")
catalog = read_events(CATALOG_PATH)
log.info(f"Loaded {len(catalog)} events")

# Check structure
events = list(catalog.events) if catalog.events else []
log.info(f"\n1. Sample events (first 3):")
for i, event in enumerate(events[:3]):
    template = extract_template_name(event)
    n_picks = len(event.picks) if event.picks else 0
    origin = event.origins[0] if event.origins else None
    origin_time = origin.time if origin else None
    log.info(f"   Event {i}: template={template}, n_picks={n_picks}, origin_time={origin_time}")

# Apply filters in same order as plot_smackover_detections.py
log.info(f"\n2. Applying filters in order:")

# Filter 1: Template exclusions
n_before = len(events)
events = [e for e in events if extract_template_name(e) not in TEMPLATE_EXCLUSIONS]
n_excluded_template = n_before - len(events)
log.info(f"   After template exclusions: {len(events)} (dropped {n_excluded_template})")

# Filter 2: Spike days
n_before = len(events)
events_keep = []
for event in events:
    template_name = extract_template_name(event)
    if template_name in SPIKE_DAY_EXCLUSIONS and len(event.origins) > 0:
        origin_time = event.origins[0].time
        event_date_str = origin_time.datetime.strftime("%Y-%m-%d")
        if event_date_str in SPIKE_DAY_EXCLUSIONS[template_name]:
            continue
    events_keep.append(event)

n_excluded_spike = n_before - len(events_keep)
events = events_keep
log.info(f"   After spike-day exclusions: {len(events)} (dropped {n_excluded_spike})")

# Filter 3: MIN_CHANS (before this is raw_catalog)
raw_count = len(events)
events_filtered = [e for e in events if len(e.picks) >= MIN_CHANS]
n_excluded_minchan = len(events) - len(events_filtered)
log.info(f"   After MIN_CHANS >= {MIN_CHANS}: {len(events_filtered)} (dropped {n_excluded_minchan})")

log.info(f"\n3. Summary:")
log.info(f"   Original:              {len(catalog)}")
log.info(f"   Raw (after excl):      {raw_count}")
log.info(f"   Filtered (final):      {len(events_filtered)}")
log.info(f"   Total excluded:        {len(catalog) - len(events_filtered)}")
log.info(f"     - templates:         {n_excluded_template}")
log.info(f"     - spike days:        {n_excluded_spike}")
log.info(f"     - min_chans:         {n_excluded_minchan}")
