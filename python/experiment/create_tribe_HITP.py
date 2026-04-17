#!/usr/bin/python

from eqcorrscan.core.match_filter import template_gen
from eqcorrscan.core.match_filter import Template, Tribe


def tribe_from_client_HITP(cat, client, parameters, seeds=None):
    """
    Efficiently create a Tribe by pulling each day's data once and looping over events in that day.
    Pulls exactly 86400s for each UTC day.
    Calls template_gen once on the entire catalog after filtering picks.
    """
    tribe = Tribe()
    catalog = cat.copy()
    catalog.events.sort(key=lambda x: x.preferred_origin().time)
    # Remove picks not in seeds list
    if seeds is not None:
        for ev in catalog:
            # arr_picks = [arr.pick_id.get_referred_object() for arr in ev.preferred_origin().arrivals]
            filt_picks = [p for p in ev.picks if p.waveform_id.get_seed_string() in seeds]
            ev.picks = filt_picks
        # Remove events with no picks left
    catalog.events = [ev for ev in catalog if len(ev.picks) > 0]
    if len(catalog) == 0:
        print("No events left after filtering picks with seeds.")
        return tribe

    # Gather all unique seed ids for the filtered catalog
    all_picks = [pk for ev in catalog for pk in ev.picks]
    ids = list({p.waveform_id.get_seed_string() for p in all_picks})
    if not ids:
        print("No picks for any events in catalog after filtering. Skipping.")
        return tribe

    # Call template_gen once on the entire catalog
    print("Generating templates for filtered catalog")
    temp_list = template_gen(
        method='from_client',
        client_id=client,
        catalog=catalog,
        data_pad=20,
        **parameters
    )
    for i, temp in enumerate(temp_list):
        ev = catalog[i]
        eid = ev.resource_id.id.split('=')[-2].split('&')[0]
        temp_new = Template(
            name=eid, event=ev, st=temp,
            lowcut=parameters['lowcut'], highcut=parameters['highcut'],
            samp_rate=parameters['samp_rate'], filt_order=parameters['filt_order'],
            prepick=parameters['prepick'], process_length=parameters['process_len']
        )
        tribe.templates.append(temp_new)
    return tribe

