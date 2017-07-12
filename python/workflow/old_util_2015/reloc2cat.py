#!/usr/bin/env python

"""
This script will lay out taking hypoDD.reloc files and reading those locations back into the catalog
from which the input file were created.
"""

# TODO This can probably be added to catalog_to_dd.py in EQcorrscan at some point

def reloc2cat(reloc_file, cat):
    from obspy import UTCDateTime
    from obspy.core.event import Origin, ResourceIdentifier
    with open(reloc_file, 'rb') as f:
        for row in f:
            ev_id = int(row.split()[0])
            row_lst = row.split()
            # Catch stupid 60 entries for seconds
            if row_lst[15].split('.')[0] == '60':
                row_lst[14] = int(row_lst[14]) + 1
                row_lst[15] = '00.000'
            cat[ev_id].origins.append(Origin(latitude=float(row_lst[1]), longitude=float(row_lst[2]),
                                             depth=float(row_lst[3]) * 1000,
                                             time=UTCDateTime(year=int(row_lst[10]), month=int(row_lst[11]),
                                                              day=int(row_lst[12]), hour=int(row_lst[13]),
                                                              minute=int(row_lst[14]),
                                                              second=int(row_lst[15].split('.')[0]),
                                                              microsecond=(int(row_lst[15].split('.')[1]))),
                                             method_id=ResourceIdentifier(id='HypoDD')))
            cat[ev_id].preferred_origin_id = str(cat[ev_id].origins[-1].resource_id)
    return cat