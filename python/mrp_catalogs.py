#!/user/bin/python

from obspy import Catalog
from obspy.core.event import Event

for event in cat_ngatamariki:
    if not 'times_nga' in locals():
        times_nga = [event.origins[0].time]
    else:
        times_nga.append(event.origins[0].time)
