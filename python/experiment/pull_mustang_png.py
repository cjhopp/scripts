#!/usr/bin/env python3
"""
Download a month–at–a-time PSD for every month from 2015-01 through today
for UW.NN19 using the IRIS MUSTANG noise-psd service.
"""

import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ------------------------------------------------------------
# 1. User-editable parameters
# ------------------------------------------------------------
BASE_URL   = "https://service.iris.edu/mustang/noise-pdf/1/query?"
NETWORK    = "UW"
STATION    = "NN21"
LOCATION   = "01"      # "--" or "*" means ‘any / blank location code’
CHANNEL    = "EHZ"     # grab every BH? channel; change if needed
OUT_DIR    = Path("/media/chopp/Data1/chet-meq/pdf_NN21")   # where to save files
SAVE_FILES = True      # set False if you only want to see the URLs
FORMAT     = "plot"    # plot | text | xml
# ------------------------------------------------------------

def week_edges(start, stop):
    """
    Yield consecutive (week_start, week_end) tuples between
    start <= t < stop. week_end is the start of the next week
    (i.e., suitable as an exclusive endtime value).
    Weeks start on the same weekday as 'start'.
    """
    ws = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    while ws < stop:
        we = ws + timedelta(days=7)
        yield ws, we
        ws = we

def build_query(start, end):
    """Return the complete query URL for one month."""
    q = (
        f"target={NETWORK}.{STATION}.{LOCATION}.{CHANNEL}.M"
        f"&starttime={start.isoformat(timespec='seconds')}"
        f"&endtime={end.isoformat(timespec='seconds')}"
        f"&format={FORMAT}&plot.power.min=-180&plot.power.max=-50"
        f"&plot.period.min=0.004&plot.period.max=100"
        #  add extra options here if desired, e.g.  correct=true
    )
    return BASE_URL + q

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc)

    for ms, me in week_edges(datetime(2015, 1, 1, tzinfo=timezone.utc), today):
        url = build_query(ms, me)
        fname = OUT_DIR / f"{STATION}_{ms:%Y_%m_%j}.{FORMAT}"

        print(url)                           # always show the URL
        if SAVE_FILES:
            try:
                r = requests.get(url, timeout=300)
                r.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print(f"  ↳ HTTP error: {e}")
                continue

            fname.write_bytes(r.content)
            print(f"  ↳ saved to {fname} ({len(r.content):,} bytes)")

if __name__ == "__main__":
    main()