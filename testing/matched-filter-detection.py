# Matched filter detection practice

import sys
sys.path.insert(0, '/Users/home/taylorte/EQcorrscan')

from eqcorrscan.core import match_filter
from eqcorrscan.utils import pre_processing
from obspy import read

st = read('daylong-data.ms')
st_processed = pre_processing.dayproc(st=st, lowcut=2.0, highcut=9.0,
                                      filt_order=4,
                                      samp_rate=20.0,
                                      starttime=st[0].stats.starttime.date)

# Reading the templates

templates = []
template_names = ['kaik_eq-WEL.ms', 'kaik_eq-WEL2.ms', 'kaik_eq-WEL3.ms',
                  'kaik_eq-WEL4.ms']

for template_file in template_names:
    templates.append(read(template_file))

detections = match_filter.match_filter(template_names=template_names,
                                       template_list=templates,
                                       st=st_processed,  threshold=8,
                                       threshold_type='MAD', trig_int=6,
                                       plotvar=False, cores=4, debug=1)

for detection in detections:
    detection.write('detection-attempt.csv', append=True)

# code only came up with a single detection, and it falls
# outside of the start and end times which is strange"
