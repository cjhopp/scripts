#!/usr/bin/env python
"""
Quick test: verify generate_hypoDD_inputs.py loads Party correctly and filters to ~624 events.
"""

import sys
sys.path.insert(0, '/home/chopp/scripts/python/workflow/Lawrencium')

from generate_hypoDD_inputs import (
    load_party_as_catalog,
    apply_quality_filters,
    PARTY_PATH
)

print("=" * 80)
print("TEST: Load Party and apply filters")
print("=" * 80)

catalog = load_party_as_catalog(PARTY_PATH)
print(f"Loaded: {len(catalog)} events from Party")

filtered_catalog, raw_catalog = apply_quality_filters(catalog)
print(f"Filtered: {len(filtered_catalog)} events (expected ~624)")

if len(filtered_catalog) >= 600:
    print("✓ SUCCESS: Got expected number of filtered events (~624)")
else:
    print(f"✗ FAIL: Expected ~624, got {len(filtered_catalog)}")
