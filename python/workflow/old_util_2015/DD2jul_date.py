#!/usr/bin/env python

"""
Script for converting HypoDD reloc file times to Julian Days and rewrite them to new file
"""

import csv
from obspy import UTCDateTime

def hypoDD_2_juldate(infile, outfile):
    with open(infile, 'rb') as S_in:
        with open(outfile, 'wb') as S_out:
            writer = csv.writer(S_out, delimiter=' ')
            for line in S_in:
                row = line.split()
                if len(row[11]) < 2:
                    month = '0' + row[11]
                else:
                    month = row[11]
                if len(row[12]) < 2:
                    day = '0' + row[12]
                else:
                    day = row[12]
                julday = UTCDateTime(row[10] + month + day).julday
                writer.writerow([row[0], row[1], row[2], row[3], julday])
    return