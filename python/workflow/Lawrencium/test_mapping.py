#!/usr/bin/env python
"""Quick test of event ID to waveform filename mapping"""

from obspy import read_events
from pathlib import Path
import os

catalog_path = "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium/Smackover_analyzed_raw.xml"
waveform_dir = "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium/waveforms/smackover_north_analyzed/MAD12_2hr"

print(f"Loading catalog...")
catalog = read_events(catalog_path)
print(f"Loaded {len(catalog)} events")

# Sample first 5 events
print("\nFirst 5 events and their expected waveform files:")
for i, event in enumerate(catalog[:5]):
    event_id_full = str(event.resource_id.id)
    if "/" in event_id_full:
        event_id = event_id_full.split("/")[1]
    else:
        event_id = event_id_full
    
    parts = event_id.rsplit("_", 1)
    if len(parts) == 2:
        template_name, iso_timestamp = parts
        # Convert ISO to waveform format
        if "T" in iso_timestamp:
            date_part, time_part = iso_timestamp.split("T")
            if "." in time_part:
                time_str, microsecond_str = time_part.split(".")
                microseconds = microsecond_str[:6].ljust(6, "0")
            else:
                time_str = time_part
                microseconds = "000000"
            waveform_timestamp = f"{date_part}_{time_str}{microseconds}"
            expected_basename = f"{template_name}_{waveform_timestamp}.mseed"
            waveform_path = os.path.join(waveform_dir, expected_basename)
            exists = os.path.exists(waveform_path)
            print(f"\n  Event {i}: {event_id}")
            print(f"    Expected file: {expected_basename}")
            print(f"    Exists: {exists}")

# Check unique templates in catalog
print("\n\nUnique templates in catalog (first 20):")
templates_cat = set()
for event in catalog:
    event_id_full = str(event.resource_id.id)
    if "/" in event_id_full:
        event_id = event_id_full.split("/")[1]
    else:
        event_id = event_id_full
    
    if "_" in event_id:
        template = event_id.rsplit("_", 1)[0]
        templates_cat.add(template)

print(sorted(list(templates_cat))[:20])

# Check what templates are in waveform directory
print("\n\nTemplates in waveforms directory (extracted as template only):")
templates_wf = set()
for f in os.listdir(waveform_dir):
    if f.endswith(".mseed"):
        # Extract only the template part (first segment before first date pattern)
        basename = f.replace(".mseed", "")
        parts = basename.split("_")
        # First part before any 8-digit date
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():
                template = "_".join(parts[:i])
                templates_wf.add(template)
                break
        else:
            # No date found, use rsplit method
            template = basename.rsplit("_", 1)[0]
            templates_wf.add(template)

print(sorted(list(templates_wf))[:20])

# Find overlap
overlap = templates_cat & templates_wf
print(f"\n\nOverlap: {len(overlap)} templates")
print(f"  Overlap templates: {sorted(list(overlap))[:10]}")
print(f"  Only in catalog: {len(templates_cat - templates_wf)}")
print(f"  Only in waveforms: {len(templates_wf - templates_cat)}")
