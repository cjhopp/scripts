#!/bin/bash

"""
Simple script to run obspyck with a number of inputs

Working command (for the moment will only work with --nometadata)
Metadata must be suplied in dataless seed or xseed (which should be available
via the db??). Perhaps another command line option?

-m: merge input waveforms if multiple for time range
-d: duration to plot (in seconds)
Other options self explanatory
"""
obspyck.py -t 2012-06-25T13:30:00 -d 7200 -i NZ.*.*.* --seishub-servername sgees017.geo.vuw.ac.nz --seishub-port 8180 -m ''
