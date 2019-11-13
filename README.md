# PhD scripts and functions
## Description
This directory contains all of the scripts used in completing my PhD in
GeoPhysics at Victoria University of Wellington from 2015-2019.

**PhD title:** Characterizing microseismicity at the Rotokawa and Ngatamariki
geothermal fields, North Island, New Zealand

## Usage
The *python/workflow* directory contains all of the relevant python files used
through the course of this work. Other directories contain older versions of
the scripts contained under *workflow* but are left as is for posterity.

Each *python/workflow* script detailed below is only a collection of functions.
They cannot be executed on their own and the functions within are meant to be
imported and used from within a python interpreter. In order to
use a particular function from a file, make sure the file is on your python
path before using: `from plot_mags import plot_mags`, for example.

The most important files are explained below.

## Workflow functions:

* *data_prep.py*: Functions to take raw miniseed and sc3ml files and output
usable QML and pyasdf files for use in main workflows. This includes
functionality for creating templates.

* *relocate.py*: Functions for relocating events in catalogs, including NLLoc,
HypoDD and GrowClust.

* *magnitudes.py*: Wrappers on magnitude calculation functions in EQcorrscan for
matched filter detections.

* *obspyck_util*: Wrapper functions on the python-based picker GUI, Obspyck.
These were used for making all polarity picks.

* *focal_mecs*: Focal mechanism calculation, plotting and file parsing for use
with a number of packages including: MTFit, Richard Arnold's R codes, HybridMT,
etc...

* *shelly_focmecs.py*: Python implementation functions for David Shelly's
polarity clustering routines.
Read the paper [here](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JB013437)

* *shelly_mags.py*: Python implementation functions for David Shelly's
relative-amplitude calculation workflow.
Read the paper [here](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JB012719)

## Plotting functions:
* *plot_detections.py*: A large number of plotting functions for catalog
locations, time series, cumulative detections and much more

* *plot_mags.py*: Various magnitude plotting functions (i.e. b-values,
magnitude time series, etc...)

* *plot_stresses.py*: Plotting mostly of the output of Richard and John's
focal mechanism inversion codes, including a python version of the principal
stress axes and pdfs.

* *plot_well_data.py*: Large number of functions for plotting most of the
Mercury datasets including PTS runs, flow rates, pressures, well locations, etc


# Data location and processing
**This section serves as reference for any VUW students and staff needing to
locate and access the data files used in this project**

## Waveform files:
The raw day-long miniseed files live in two locations on the network:

For 2012-2013 data:
* **/Volumes/Taranaki_01/data/civilfra/Rotokawa_Seismic/raw_data/RT-NM_mseed**

For 2014-2015 data:
* **/Volumes/Taranaki_01/data/hoppche/waveform_data**

For a local archive of all GeoNet waveforms relevant to our network:
* **/Volumes/GeoPhysics_07/users-data/hoppche/geonet_waveforms**

## Earthquake catalogs
The final earthquake catalogs and station information can be accessed at:
* [This project's OSF repository](10.17605/OSF.IO/C2M6U)
* On the VUW network: **/Volumes/GeoPhysics_07/users-data/hoppche/FINAL_FILES_29-3-19**

## Mercury datasets
The Mercury flow rate and pressure datasets used throughout the project are
found in the following directory on the network:
* **/Volumes/GeoPhysics_07/users-data/hoppche/MERCURY_FILES**

I will retain these locally, but will not make them publically available via
the above repo, as they are proprietary.

## Acknowledgments
These scripts rely heavily upon other Python packages including, but not
limited to:
* [EQcorrscan](https://eqcorrscan.github.io/)
* [Obspy](https://docs.obspy.org/)
* [Obspyck](https://github.com/megies/obspyck)
These scripts assume that the above are installed properly and are accessible
on the users machine.