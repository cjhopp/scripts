#!/usr/bin/bash

"""Example function for querying both FDSN event info and usgs COMCAT"""

import io
from obspy import UTCDateTime, read_events
from obspy.core.event import Pick, Arrival, WaveformStreamID
from obspy.clients.fdsn import Client

from libcomcat.search import get_event_by_id
from libcomcat.dataframes import get_phase_dataframe



def retrieve_usgs_catalog(**kwargs):
    """
    Wrapper on obspy.clients.fdsn.Client and libcomcat (usgs) to retrieve a full
    catalog, including phase picks (that otherwise are not supported by the usgs
    fdsn implementation)

    :param kwargs: Will be passed to the Client (e.g. minlongitude, maxmagnitude
        etc...)
    :return: obspy.core.events.Catalog
    """
    cli = Client('https://earthquake.usgs.gov')
    cat = cli.get_events(**kwargs)
    # Now loop over each event and grab the phase dataframe using libcomcat
    for ev in cat:
        print(ev.resource_id.id)
        eid = ev.resource_id.id.split('=')[-2].split('&')[0]
        detail = get_event_by_id(eid, includesuperseded=True)
        phase_df = get_phase_dataframe(detail)
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
            arr = Arrival(pick_id=pk.resource_id.id, phase=pk.phase_hint,
                          azimuth=phase_info['Azimuth'],
                          distance=phase_info['Distance'],
                          time_residual=phase_info['Residual'],
                          time_weight=phase_info['Weight'])
            o.arrivals.append(arr)
        # Try to read focal mechanisms/moment tensors
        if 'moment-tensor' in detail.products:
            # Always take MT where available
            mt_xml = detail.getProducts(
                'moment-tensor')[0].getContentBytes('quakeml.xml')[0]
        elif 'focal-mechanism' in detail.products:
            mt_xml = detail.getProducts(
                'focal-mechanism')[0].getContentBytes('quakeml.xml')[0]
        else:
            continue
        mt_ev = read_events(io.TextIOWrapper(io.BytesIO(mt_xml),
                                             encoding='utf-8'))
        FM = mt_ev[0].focal_mechanisms[0]
        FM.triggering_origin_id = ev.preferred_origin().resource_id.id
        ev.focal_mechanisms = [FM]
    return cat

