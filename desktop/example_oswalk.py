import pdb, sys, os, fnmatch
import matplotlib.pyplot as plt
from obspy import read, Stream, readEvents, UTCDateTime
from core import template_gen, match_filter
from utils.Sfile_util import PICK
from utils.EQcorrscan_plotting import pretty_template_plot as tplot#import numpy as np

wav_files = []
for root, dirnames, filenames in os.walk('/Volumes/GeoPhysics_07/users-data/hoppche/gabe_backup/'):
    for filename in fnmatch.filter(filenames, '*.159*'):
        wav_files.append(os.path.join(root, filename))

#Read in all waveforms from list generated above
for wavefile in wav_files:
    if not 'st' in locals():
        st = read(wavefile)
    else:
        st += read(wavefile)
