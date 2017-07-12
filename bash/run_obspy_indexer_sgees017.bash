#!/bin/bash

DB=postgresql://mercury:MercuryIsThePlanet@127.0.0.1:5432/mercury
DATA=/Volumes/Taranaki_01/data/civilfra/Rotokawa_Seismic/raw_data/RT-NM_mseed,/Volumes/Taranaki_01/data/hoppche/waveform_data,/Volumes/GeoPhysics_07/users-data/hoppche/geonet_waveforms
LOG=./indexer.log

#Run only once and remove duplicates:
obspy-indexer -v -i0.0 --run-once --check-duplicates -n2 -u$DB -d$DATA
