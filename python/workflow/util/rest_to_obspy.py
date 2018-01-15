#!/usr/env/python
"""
Functions to convert REST formatted picks for ObsPy Events.

Note, if called as a command-line script it will convert the REST
file to a QuakeML file.

:author:	Calum J. Chamberlain
:date:		31/10/2017
:licence:	LGPL v.3.0
"""

from obspy.core.event import (
    Event, Catalog, Pick, Origin, Magnitude, QuantityError,
    ResourceIdentifier, OriginQuality, WaveformStreamID,
    Arrival)
from obspy import UTCDateTime
from obspy.geodetics import kilometer2degrees


def rest_to_obspy(filename, polfile):
    """
    Read REST formatted event info to an ObsPy Event object.

    :param filename: File to read from
    :type filename: str

    :returns: :class:`obspy.core.event.Event`
    """
    catalog = Catalog()
    with open(filename, 'r') as f:
        full_str = [line for line in f]
    with open(polfile, 'r') as pf:
        all_pol_str = [line for line in pf]
    # this test is not valid for the full file.
    # if not is_rest(event_str):
    #     raise IOError(
    #         "%s is not REST formatted (as coded)" % filename)
    event_str = []
    ev_p_str = []
    for eline, pline in zip(full_str, all_pol_str):
        if len(eline.rstrip(" \n\r")) != 0:
            event_str.append(eline)
            ev_p_str.append(pline)
        else:
            event = read_origin(event_str)
            print(event.origins[0])
            event = read_picks(event_str, event)
            print(ev_p_str)
            event = read_pol(ev_p_str, event)
            catalog.events.append(event)
            event_str = []
            ev_p_str = []
    return catalog


def is_rest(event_str):
    """
    Check if the format is as expected.

    :param event_str: Contents of file as list of str
    :type event_str: list

    :returns: bool
    """
    if len(event_str[0].rstrip()) is not 141:
        return False
    if event_str[0][0:4] != 'YEAR':
        return False
    if event_str[3][0:3] != 'STA':
        return False
    return True


def read_origin(event_str):
    """
    Read the origin information from the REST file string

    :param event_str: Contents of file as list of str
    :type event_str: list

    :returns: :class:`obspy.core.event.Event`
    """
    event = Event()

    head = event_str[0].split()
    try:
        gap = float(head[17])
    except IndexError:
        gap = None
    origin = Origin(
        time=UTCDateTime(
            year=int(head[0]), julday=int(head[1]), hour=int(head[2]),
            minute=int(head[3])) + float(head[4]),
        latitude=float(head[5]), longitude=float(head[6]),
        depth=float(head[7]) * 1000, origin_quality=OriginQuality(
            standard_error=float(head[9]),
            azimuthal_gap=gap,
            used_phase_count=int(head[17])),
        longitude_errors=QuantityError(
            uncertainty=kilometer2degrees(float(head[12]))),
        latitude_errors=QuantityError(
            uncertainty=kilometer2degrees(float(head[11]))),
        depth_errors=QuantityError(uncertainty=float(head[13]) * 1000),
        method_id=ResourceIdentifier("smi:local/REST"),
        evaluation_mode="automatic")
    event.origins.append(origin)
    try:
        event.magnitudes.append(Magnitude(
            mag=float(head[19]), magnitude_type="M"))
    except IndexError:
        pass
    return event


def read_picks(event_str, event):
    """
    Read the picks from the REST file string

    :param event_str: Contents of file as list of str
    :type event_str: list
    :param event:
        Event to assoaite the picks with. Note old picks will
        not be overwritten. Event should have only one origin.
    :type event: :class:`obspy.core.event.Event`

    :returns: :class:`obspy.core.event.Event`
    """
    for line in event_str[1:]:
        pick, arrival = read_pick(line)
        event.picks.append(pick)
        event.origins[0].arrivals.append(arrival)
    return event


def read_pick(line):
    """
    Convert REST pick string to ObsPy Pick object

    :param line: string containing pick information
    :type line: str

    :returns:
        :class:`obspy.core.event.Pick` and
        :class:`obspy.core.event.origin.Arrival`
    """
    # line = line.split()  # Cannot just split the line :(
    splits = [0, 6, 10, 15, 18, 22, 28, 29, 41, 49, -1]
    _line = []
    for split in range(len(splits) - 1):
        _line.append(line[splits[split]: splits[split + 1]].strip())
    line = _line
    pick = Pick(time=UTCDateTime(
        year=int(line[1]), julday=int(line[2]), hour=int(line[3]),
        minute=int(line[4])) + float(line[5]), phase_hint=line[7],
        evaluation_mode="automatic",
        method_id=ResourceIdentifier("smi:local/REST"),
        waveform_id=WaveformStreamID(station_code=line[0]),
        time_errors=QuantityError(uncertainty=float(line[8])))
    arrival = Arrival(
        pick_id=pick.resource_id, time_residual=float(line[9]))
    return pick, arrival

def read_pol(ev_p_str, event):
    """
    Add REST polarities to Obspy origin arrivals

    :param event_str: Event string with header and pick lines
    :type event_str: str
    :param event: Event we want to add polarities to
    :type event: obspy.core.event.Event
    :param polfile: Filename for the polarity picks
    :type polfile: str

    :returns:
        :class:'obspy.core.event.Event'
    """
    for line in ev_p_str[1:]:
        line = line.split()
        onset = line[-3]
        sta = line[0]
        if '*' in line[5]:
            pha = line[5].split('*')[-1]
        else:
            pha = line[6]
        pik = [pk for pk in event.picks if pk.waveform_id.station_code == sta
               and pha == pk.phase_hint][0]
        if float(onset) < 0.0:
            pik.polarity = "negative"
        elif float(onset) > 0.0:
            pik.polarity = "positive"
    return event

if __name__ == '__main__':
    import sys

    help_msg = (
        "Requires two arguments, the input REST pick file, the input REST pol"
        " file, and the output QuakeML filename")
    if len(sys.argv) != 4:
        print(help_msg)
    catalog = rest_to_obspy(sys.argv[1], sys.argv[2])
    catalog.write(sys.argv[3], format="QUAKEML")
