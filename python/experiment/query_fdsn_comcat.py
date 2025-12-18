#!/usr/bin/bash

"""Example function for querying both FDSN event info and usgs COMCAT"""

import io
import os
from obspy import UTCDateTime, read_events
from obspy.core.event import Pick, Arrival, WaveformStreamID
from obspy.clients.fdsn import Client

from libcomcat.search import get_event_by_id
from libcomcat.dataframes import get_phase_dataframe



def retrieve_usgs_catalog(output_dir, **kwargs):
    """
    Wrapper on obspy.clients.fdsn.Client and libcomcat (usgs) to retrieve a full
    catalog, including phase picks (that otherwise are not supported by the usgs
    fdsn implementation). Writes each event to a file and skips already processed events.

    :param output_dir: Directory to save individual event files.
    :param kwargs: Will be passed to the Client (e.g. minlongitude, maxmagnitude, etc...).
    :return: obspy.core.events.Catalog
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    cli = Client('https://earthquake.usgs.gov')
    cat = cli.get_events(**kwargs)

    # Loop over each event and process it
    for ev in cat:
        event_id = ev.resource_id.id.split('=')[-2].split('&')[0]
        event_file = os.path.join(output_dir, f"{event_id}.xml")

        # Skip the event if it has already been processed
        if os.path.exists(event_file):
            print(f"Event {event_id} already processed. Skipping...")
            continue

        print(f"Processing event {event_id}...")
        detail = get_event_by_id(event_id, includesuperseded=True)

        try:
            phase_df = get_phase_dataframe(detail)
        except Exception as e:
            print(f"Error getting phase dataframe for event {event_id}: {e}")
            continue

        if phase_df is None:
            print(f"Phase dataframe is None for event {event_id}. Skipping...")
            continue

        # Add picks and arrivals to the event
        o = ev.preferred_origin()
        for i, phase_info in phase_df.iterrows():
            seed_id = phase_info['Channel'].split('.')
            loc = seed_id[-1]
            if loc == '--':
                loc = ''
            wf_id = WaveformStreamID(network_code=seed_id[0],
                                     station_code=seed_id[1],
                                     location_code=loc,
                                     channel_code=seed_id[2])
            pk = Pick(time=UTCDateTime(phase_info['Arrival Time']),
                      method=phase_info['Status'], waveform_id=wf_id,
                      phase_hint=phase_info['Phase'])
            ev.picks.append(pk)
            try:
                arr = Arrival(pick_id=pk.resource_id.id, phase=pk.phase_hint,
                              azimuth=phase_info['Azimuth'],
                              distance=phase_info['Distance'],
                              time_residual=phase_info['Residual'],
                              time_weight=phase_info['Weight'])
                o.arrivals.append(arr)
            except ValueError as e:
                continue

        # Try to read focal mechanisms/moment tensors
        if 'moment-tensor' in detail.products:
            # Always take MT where available
            mt_xml = detail.getProducts(
                'moment-tensor')[0].getContentBytes('quakeml.xml')[0]
        elif 'focal-mechanism' in detail.products:
            mt_xml = detail.getProducts(
                'focal-mechanism')[0].getContentBytes('quakeml.xml')[0]
        else:
            mt_xml = None

        if mt_xml:
            mt_ev = read_events(io.TextIOWrapper(io.BytesIO(mt_xml),
                                                 encoding='utf-8'))
            FM = mt_ev[0].focal_mechanisms[0]
            FM.triggering_origin_id = ev.preferred_origin().resource_id.id
            ev.focal_mechanisms = [FM]

        # Write the event to file
        try:
            ev.write(event_file, format="QUAKEML")
            print(f"Saved event {event_id} to {event_file}")
        except Exception as e:
            print(f"Error saving event {event_id} to file: {e}")

    print("Catalog processing complete.")
    return cat

