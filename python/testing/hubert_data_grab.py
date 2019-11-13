#!/usr/bin/python

"""
Functions for grabbing IRIS waveforms for all events in a text file
"""

def grab_ev_wavs(evfile, pickfile, outdir):
    import csv
    import os
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client

    cli = Client('IRIS', user='martha.savage@vuw.ac.nz',
                 password='erJkla3i9eMa')
    with open(evfile, 'rb') as evs:
        with open(pickfile, 'rb') as picks:
            ev_reader = csv.reader(evs, delimiter=' ')
            pk_reader = csv.reader(picks, delimiter=' ')
            for ev in ev_reader:
                t1 = UTCDateTime(ev[4])
                t2 = t1 + 200.
                bulk_info = [("YH", "LOBS1", "*", "HH*", t1, t2),
                             ("YH", "LOBS3", "*", "HH*", t1, t2),
                             ("YH", "LOBS7", "*", "HH*", t1, t2),
                             ("YH", "LOBS8", "*", "HH*", t1, t2),
                             ("YH", "LOBS9", "*", "HH*", t1, t2),
                             ("YH", "LOB10", "*", "HH*", t1, t2)]
                st = cli.get_waveforms_bulk(bulk_info, attach_response=True)
                inv = cli.get_stations_bulk(bulk_info)
                st.merge(fill_value='interpolate').detrend()
                for tr in st:
                    sta_inv = inv.select(station=tr.stats.station)
                    pk_P = [pk for pk in pk_reader if pk[0] == ev[0]
                            and pk[1] == tr.stats.station and pk[3] == 'P']
                    pk_S = [pk for pk in pk_reader if pk[0] == ev[0]
                            and pk[1] == tr.stats.station and pk[3] == 'S']
                    tr.stats['sac']['evla'] = ev[1]
                    tr.stats['sac']['evlo'] = ev[2]
                    tr.stats['sac']['evdp'] = ev[3]
                    if len(pk_P) != 0:
                        tr.stats['sac']['t1'] = UTCDateTime(pk_P[0][2])
                    if len(pk_S) != 0:
                        tr.stats['sac']['t2'] = UTCDateTime(pk_S[0][2])
                    tr.stats['sac']['o'] = t1
                    tr.stats['sac']['knetwk'] = sta_inv[0].code
                    tr.stats['sac']['kstnm'] = sta_inv[0][0].code
                    tr.stats['sac']['stla'] = sta_inv[0][0].latitude
                    tr.stats['sac']['stlo'] = sta_inv[0][0].longitude
                    tr.stats['sac']['stel'] = sta_inv[0][0].elevation
                    for chan in sta_inv[0][0]:
                        if chan.code == tr.stats.channel:
                            tr.stats['sac']['stdp'] = chan.depth
                            tr.stats['sac']['cmpaz'] = chan.azimuth
                            tr.stats['sac']['kcmpnm'] = chan.code
                            # SAC cmpinc is deg from vertical (not horiz)
                            if chan.dip == -90.0:
                                tr.stats['sac']['cmpinc'] = 180.0
                                tr.stats['sac']['lpspol'] = False
                            elif chan.dip == 90.0:
                                tr.stats['sac']['cmpinc'] = 0.0
                                tr.stats['sac']['lpspol'] = True
                            else:
                                tr.stats['sac']['cmpinc'] = 90.0
                    if not os.path.exists('%s/%s' % (outdir,
                                                     sta_inv[0][0].code)):
                        os.mkdir('%s/%s' % (outdir, sta_inv[0][0].code))
                    if not os.path.exists('%s/%s/%s' % (outdir,
                                                        sta_inv[0][0].code,
                                                        ev[0])):
                        os.mkdir('%s/%s/%s' % (outdir, sta_inv[0][0].code,
                                               ev[0]))
                    filename = '%s.%s.%s' % (str(t1), ev[0], tr.stats.channel)
                    print('Writing event ' + filename + ' to file...')
                    tr.write(os.path.join(outdir, sta_inv[0][0], ev[0],
                                          filename), format="SAC")
    return



