#!/usr/bin/python
import os
import sys
sys.path.insert(0, '/home/chopp/scripts/python')

from obspy import read, read_events, Stream, Catalog, UTCDateTime
from obspy.clients.fdsn import Client
from eqcorrscan import Tribe, Template
from eqcorrscan.core.template_gen import template_gen

from lbnl.denoiser import remove_HITP_spikes


def build_tribe_with_despike(catalog,
                             waveform_cache_root,
                             spike_template_paths,
                             parameters,
                             out_tribe_path,
                             client_url,
                             network,
                             station,
                             channels,
                             chunk_length_sec=3600,
                             plot=False,
                             plot_output_dir='.'):
    """
    Build a Tribe from event XMLs + daylong MSEEDs, with despiking applied
    to daylong data before template generation.

    Daylong mseed cache files are stored as:
    {waveform_cache_root}/{NETWORK}.{STATION}.{YYYYJJJ}.mseed
    """
    os.makedirs(waveform_cache_root, exist_ok=True)
    client = Client(client_url)

    cat = read_events(catalog)

    cat.events.sort(key=lambda x: x.preferred_origin().time)
    tribe = Tribe()

    for ev in cat:
        # Keep only preferred-origin picks
        ev_cat = Catalog(events=[ev])

        t0 = ev.preferred_origin().time
        day_start = UTCDateTime(t0.date)
        day_end = day_start + 86400
        day_key = t0.strftime("%Y%j")

        cache_fname = f"{network}.{station}.{day_key}.mseed"
        cache_path = os.path.join(waveform_cache_root, cache_fname)

        if os.path.exists(cache_path):
            print(f"Reading cached daily data: {cache_path}")
            st = read(cache_path)
        else:
            print(f"Downloading daily data for {day_key}...")
            st = client.get_waveforms(network, station, '*', ",".join(channels), day_start, day_end)
            st.write(cache_path, format="MSEED")
            print(f"Wrote daily data to cache: {cache_path}")

        st.merge(fill_value='interpolate')

        # Despike in chunks
        st_denoised = Stream()
        chunk_start = day_start
        while chunk_start < day_end:
            chunk_end = min(chunk_start + chunk_length_sec, day_end)
            chunk = st.slice(chunk_start, chunk_end)

            remove_HITP_spikes(
                stream=chunk,
                spike_template_path=spike_template_paths,
                geophone_chans=parameters.get('geophone_chans', ['GPZ']),
                plot=plot,
                plot_output_dir=plot_output_dir,
                chunk_start=chunk_start,
            )

            st_denoised += chunk
            chunk_start = chunk_end

        st_denoised.merge(fill_value='interpolate')

        # Generate template from despiked stream
        eid = ev.resource_id.id.split('/')[-1]
        print(f"Generating template {eid}")
        temp = template_gen(
            method='from_meta_file',
            name=eid,
            st=st_denoised,
            process=True,
            meta_file=ev_cat,
            **parameters
        )

        temp_new = Template(
            name=eid,
            event=ev,
            st=temp[0],
            lowcut=parameters['lowcut'],
            highcut=parameters['highcut'],
            samp_rate=parameters['samp_rate'],
            filt_order=parameters['filt_order'],
            prepick=parameters['prepick'],
            process_length=parameters['process_length']
        )
        tribe.templates.append(temp_new)

    tribe.write(out_tribe_path)
    print(f"Wrote Tribe to: {out_tribe_path}")
    return tribe


if __name__ == "__main__":
    # --- USER PARAMETERS ---
    CATALOG = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/Template_catalog_clustered.xml"
    WAVEFORM_CACHE_ROOT = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/waveform_cache"
    SPIKE_TEMPLATE_PATHS = [
        "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
        "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
    ]
    OUT_TRIBE = "/media/chopp/HDD1/chet-meq/cape_modern/templates/eqcorrscan/HITP_despiked_templates_clustered.tgz"

    CLIENT_URL = "http://131.243.224.19:8085"
    NETWORK = "6K"
    STATION = "HITP,HITP2"
    CHANNELS = ["GK1", "GPZ"]

    TEMPLATE_PARAMS = dict(
        lowcut=3.0,
        highcut=100.0,
        samp_rate=250.0,
        filt_order=3,
        prepick=0.05,
        length=1.2,
        process_length=86400.0,
        geophone_chans=['GPZ'],
    )

    build_tribe_with_despike(
        catalog=CATALOG,
        waveform_cache_root=WAVEFORM_CACHE_ROOT,
        spike_template_paths=SPIKE_TEMPLATE_PATHS,
        parameters=TEMPLATE_PARAMS,
        out_tribe_path=OUT_TRIBE,
        client_url=CLIENT_URL,
        network=NETWORK,
        station=STATION,
        channels=CHANNELS,
        chunk_length_sec=3600,
        plot=False,
        plot_output_dir=".",
    )