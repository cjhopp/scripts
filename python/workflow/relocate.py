#!/usr/bin/env python

"""
Script to handle pick refinement/removal and relocation of catalog earthquakes.
"""

import os
import locale
import warnings
import pyproj
import numpy as np

from glob import glob
from subprocess import call
from datetime import datetime
from obspy import UTCDateTime
from obspy.core.event import Arrival, QuantityError, ResourceIdentifier, \
    OriginUncertainty, Origin, CreationInfo, OriginQuality
from obspy.core import AttribDict
from obspy.geodetics import kilometer2degrees


"""
Now running NLLoc from subprocess call and reading new origin back into catalog
"""

# origin = [-38.3724, 175.9577]

def my_conversion(x, y, z):
    origin = [-38.3724, 175.9577]
    new_y = origin[0] + ((y * 1000) / 111111)
    new_x = origin[1] + ((x * 1000) /
                         (111111 * np.cos(origin[0] * (np.pi/180))))
    return new_x, new_y, z

def casc_xyz2latlon(x, y):
    """
    Convert from scaled surf xyz (in km) to lat lon
    :param x:
    :param y:
    :return:
    """
    pts = zip(x, y)
    orig_utm = (239200, 5117300)
    utm = pyproj.Proj(init="EPSG:32610")
    pts_utm = [(orig_utm[0] + (pt[0] * 1000), orig_utm[1] + (pt[1] * 1000))
               for pt in pts]
    utmx, utmy = zip(*pts_utm)
    lon, lat = utm(utmx, utmy, inverse=True)
    return (lon, lat)

def surf_xyz2latlon(x, y):
    """
    Convert from scaled surf xyz (in km) to lat lon
    :param x:
    :param y:
    :return:
    """
    # Descale (/10) and convert to meters
    x *= 10
    y *= 10
    pts = zip(x, y)
    orig_utm = (598420.3842806489, 4912272.275375654)
    utm = pyproj.Proj(init="EPSG:26713")
    pts_utm = [(orig_utm[0] + pt[0], orig_utm[1] + pt[1])
               for pt in pts]
    utmx, utmy = zip(*pts_utm)
    lon, lat = utm(utmx, utmy, inverse=True)
    return (lon, lat)

def fsb_xyz2latlon(x, y):
    """
    Convert from scaled surf xyz (in km) to lat lon
    :param x:
    :param y:
    :return:
    """
    # Descale (/10) and convert to meters
    x *= 10
    y *= 10
    pts = zip(x, y)
    orig_utm = (2579255., 1247501.)
    utm = pyproj.Proj(init='EPSG:2056')
    pts_utm = [(orig_utm[0] + pt[0], orig_utm[1] + pt[1])
               for pt in pts]
    utmx, utmy = zip(*pts_utm)
    lon, lat = utm(utmx, utmy, inverse=True)
    return (lon, lat)


def relocate(cat, root_name, in_file, pick_uncertainty, location='SURF'):
    """
    Run NonLinLoc relocations on a catalog.

    :type cat: obspy.Catalog
    :param cat: catalog of events with picks to relocate
    :type root_name: str
    :param root_name: String specifying where the nlloc.obs files will be
        written from the catalog
    :type in_file: str
    :param in_file: NLLoc input file
    :type pick_uncertainty: dict
    :param pick_uncertainty: Dictionary mapping uncertainties to sta/chans
    :param location: Which coordinate conversion to use

    :return: same catalog with new origins appended to each event
    """
    for ev in cat:
        if len(ev.picks) < 5:
            print('Fewer than 5 picks for {}. Will not locate.'.format(
                ev.resource_id.id))
            continue
        for pk in ev.picks:
            # Assign arrival time uncertainties if mapping provided
            if (not pk.time_errors.upper_uncertainty
                and not pk.time_errors.uncertainty) or pick_uncertainty:
                sta = pk.waveform_id.station_code[:2]
                chan = pk.waveform_id.channel_code[-1]
                try:
                    pk.time_errors.uncertainty = pick_uncertainty[sta][chan]
                except (TypeError, KeyError) as e:
                    try:
                        pk.time_errors.uncertainty = pick_uncertainty[pk.phase_hint[0]]
                    except (TypeError, KeyError) as e:
                        pk.time_errors.uncertainty = pick_uncertainty
            # For cases of specific P or S phases, just convert to P or S
            if pk.phase_hint not in ['P', 'S']:
                pk.phase_hint = pk.phase_hint[0]
        id_str = str(ev.resource_id).split('/')[-1]
        if len(id_str.split('=')) > 1 and location == 'cascadia':
            # For FDSN pulled events from USGS
            id_str = ev.resource_id.id.split('=')[-2].split('&')[0]
        filename = '{}/obs/{}.nll'.format(root_name, id_str)
        outfile = '{}/loc/{}'.format(root_name, id_str)
        # TODO This clause needs faster file existece check. Do 25-7.
        if os.path.isfile(filename):
            # if len(glob(outfile + '.????????.??????.grid0.loc.hyp')) > 0:
            print('LOC file already written, reading output to catalog')
        else:
            # Here forego obspy write func in favor of obspyck dicts2NLLocPhases
            phases = dicts2NLLocPhases(ev, location)
            with open(filename, 'w') as f:
                f.write(phases)
            # Specify awk command to edit NLLoc .in file
            # Write to unique tmp file (just in_file.bak) so as not to
            # overwrite if multiple instances running.
            cmnd = """awk '$1 == "LOCFILES" {$2 = "%s"; $5 = "%s"}1' %s > %s.bak && mv %s.bak %s""" % (
                filename, outfile, in_file, in_file, in_file, in_file)
            call(cmnd, shell=True)
            # Call to NLLoc
            call('NLLoc %s' % in_file, shell=True)
        # Now reading NLLoc output back into catalog as new origin
        # XXX BE MORE CAREFUL HERE. CANNOT GRAB BOTH SUM AND NON-SUM
        out_w_ext = glob(outfile + '.????????.??????.grid0.loc.hyp')
        try:
            loadNLLocOutput(ev=ev, infile=out_w_ext[0], location=location)
        except (ValueError, IndexError) as ve:
            print(ve)
            continue
        # ev.origins.append(new_o_obj)
        # ev.preferred_origin_id = new_o_obj.resource_id.id
    return cat

def dicts2NLLocPhases(ev, location):
    """
    *********
    CJH Stolen from obspyck to use a scaling hack for 6 decimal precision
    *********

    Returns the pick information in NonLinLoc's own phase
    file format as a string. This string can then be written to a file.
    Currently only those fields really needed in location are actually used
    in assembling the phase information string.

    Information on the file formats can be found at:
    http://alomax.free.fr/nlloc/soft6.00/formats.html#_phase_

    Quote:
    NonLinLoc Phase file format (ASCII, NLLoc obsFileType = NLLOC_OBS)

    The NonLinLoc Phase file format is intended to give a comprehensive
    phase time-pick description that is easy to write and read.

    For each event to be located, this file contains one set of records. In
    each set there is one "arrival-time" record for each phase at each seismic
    station. The final record of each set is a blank. As many events as desired can
    be included in one file.

    Each record has a fixed format, with a blank space between fields. A
    field should never be left blank - use a "?" for unused characther fields and a
    zero or invalid numeric value for numeric fields.

    The NonLinLoc Phase file record is identical to the first part of each
    phase record in the NLLoc Hypocenter-Phase file output by the program NLLoc.
    Thus the phase list output by NLLoc can be used without modification as time
    pick observations for other runs of NLLoc.

    NonLinLoc phase record:
    Fields:
    Station name (char*6)
        station name or code
    Instrument (char*4)
        instument identification for the trace for which the time pick
        corresponds (i.e. SP, BRB, VBB)
    Component (char*4)
        component identification for the trace for which the time pick
        corresponds (i.e. Z, N, E, H)
    P phase onset (char*1)
        description of P phase arrival onset; i, e
    Phase descriptor (char*6)
        Phase identification (i.e. P, S, PmP)
    First Motion (char*1)
        first motion direction of P arrival; c, C, u, U = compression;
        d, D = dilatation; +, -, Z, N; . or ? = not readable.
    Date (yyyymmdd) (int*6)
        year (with century), month, day
    Hour/minute (hhmm) (int*4)
        Hour, min
    Seconds (float*7.4)
        seconds of phase arrival
    Err (char*3)
        Error/uncertainty type; GAU
    ErrMag (expFloat*9.2)
        Error/uncertainty magnitude in seconds
    Coda duration (expFloat*9.2)
        coda duration reading
    Amplitude (expFloat*9.2)
        Maxumim peak-to-peak amplitude
    Period (expFloat*9.2)
        Period of amplitude reading
    PriorWt (expFloat*9.2)

    A-priori phase weight Currently can be 0 (do not use reading) or
    1 (use reading). (NLL_FORMAT_VER_2 - WARNING: under development)

    Example:

    GRX    ?    ?    ? P      U 19940217 2216   44.9200 GAU  2.00e-02 -1.00e+00 -1.00e+00 -1.00e+00
    GRX    ?    ?    ? S      ? 19940217 2216   48.6900 GAU  4.00e-02 -1.00e+00 -1.00e+00 -1.00e+00
    CAD    ?    ?    ? P      D 19940217 2216   46.3500 GAU  2.00e-02 -1.00e+00 -1.00e+00 -1.00e+00
    CAD    ?    ?    ? S      ? 19940217 2216   50.4000 GAU  4.00e-02 -1.00e+00 -1.00e+00 -1.00e+00
    BMT    ?    ?    ? P      U 19940217 2216   47.3500 GAU  2.00e-02 -1.00e+00 -1.00e+00 -1.00e+00
    """
    nlloc_str = ""

    for pick in ev.picks:
        if pick.waveform_id.station_code == 'NSMTC':
            sta = 'NSMT{}'.format(pick.waveform_id.location_code)
        else:
            sta = pick.waveform_id.station_code.ljust(6)
        inst = "?".ljust(4)
        comp = "?".ljust(4)
        onset = "?"
        try:
            phase = pick.phase_hint.ljust(6)
        except AttributeError:
            phase = 'P'.ljust(6)
        pol = "?"
        t = pick.time
        if location in ['SURF', 'FSB']:
            # CJH Hack to accommodate full microsecond precision...
            t = datetime.fromtimestamp(t.datetime.timestamp() * 100)
        date = t.strftime("%Y%m%d")
        hour_min = t.strftime("%H%M")
        sec = "%7.4f" % (t.second + t.microsecond / 1e6)
        error_type = "GAU"
        error = None
        # XXX check: should we take only half of the complete left-to-right error?!?
        if location == 'cascadia':
            error = pick.time_errors.uncertainty
        elif pick.time_errors.upper_uncertainty and pick.time_errors.lower_uncertainty:
            error = (pick.time_errors.upper_uncertainty + pick.time_errors.lower_uncertainty) * 100
        elif pick.time_errors.uncertainty:
            error = 200 * pick.time_errors.uncertainty
        error = "%9.2e" % error
        coda_dur = "-1.00e+00"
        ampl = "-1.00e+00"
        period = "-1.00e+00"
        fields = [sta, inst, comp, onset, phase, pol, date, hour_min,
                  sec, error_type, error, coda_dur, ampl, period]
        phase_str = " ".join(fields)
        nlloc_str += phase_str + "\n"
    return nlloc_str

def loadNLLocOutput(ev, infile, location):
    lines = open(infile, "rt").readlines()
    if not lines:
        err = "Error: NLLoc output file (%s) does not exist!" % infile
        print(err)
        return
    # goto signature info line
    try:
        line = lines.pop(0)
        while not line.startswith("SIGNATURE"):
            line = lines.pop(0)
    except:
        err = "Error: No correct location info found in NLLoc " + \
              "outputfile (%s)!" % infile
        print(err)
        return

    line = line.rstrip().split('"')[1]
    signature, nlloc_version, date, time = line.rsplit(" ", 3)
    # new NLLoc > 6.0 seems to add prefix 'run:' before date
    if date.startswith('run:'):
        date = date[4:]
    saved_locale = locale.getlocale()
    try:
        locale.setlocale(locale.LC_ALL, ('en_US', 'UTF-8'))
    except:
        creation_time = None
    else:
        creation_time = UTCDateTime().strptime(date + time,
                                               str("%d%b%Y%Hh%Mm%S"))
    finally:
        locale.setlocale(locale.LC_ALL, saved_locale)
    # goto maximum likelihood origin location info line
    try:
        line = lines.pop(0)
        while not line.startswith("HYPOCENTER"):
            line = lines.pop(0)
    except:
        err = "Error: No correct location info found in NLLoc " + \
              "outputfile (%s)!" % infile
        print(err)
        return

    line = line.split()
    x = float(line[2])
    y = float(line[4])
    depth = float(line[6]) * 1000 # depth: negative down!
    if location == 'cascadia':
        lon, lat = casc_xyz2latlon(np.array([x]), np.array([y]))
    # Convert coords
    elif location in ['SURF', 'FSB']:
        # CJH I reported depths at SURF in meters below 130 m so positive is
        # down in this case
        depth = float(line[6])
        print('Doing hypo conversion')
        # Descale first
        depth *= 10
        if location == 'SURF':
            lon, lat = surf_xyz2latlon(np.array([x]), np.array([y]))
        else:
            lon, lat = fsb_xyz2latlon(np.array([x]), np.array([y]))
    else:
        print('Location: {} not supported'.format(location))
        return
    # goto origin time info line
    try:
        line = lines.pop(0)
        while not line.startswith("GEOGRAPHIC  OT"):
            line = lines.pop(0)
    except:
        err = "Error: No correct location info found in NLLoc " + \
              "outputfile (%s)!" % infile
        print(err)
        return
    line = line.split()
    year = int(line[2])
    month = int(line[3])
    day = int(line[4])
    hour = int(line[5])
    minute = int(line[6])
    seconds = float(line[7])
    time = UTCDateTime(year, month, day, hour, minute, seconds)
    if location in ['SURF', 'FSB']:
        # Convert to actual time
        time = UTCDateTime(datetime.fromtimestamp(
            time.datetime.timestamp() / 100.
        ))
    # goto location quality info line
    try:
        line = lines.pop(0)
        while not line.startswith("QUALITY"):
            line = lines.pop(0)
    except:
        err = "Error: No correct location info found in NLLoc " + \
              "outputfile (%s)!" % infile
        print(err)
        return

    line = line.split()
    rms = float(line[8])
    gap = float(line[12])

    # goto location quality info line
    try:
        line = lines.pop(0)
        while not line.startswith("STATISTICS"):
            line = lines.pop(0)
    except:
        err = "Error: No correct location info found in NLLoc " + \
              "outputfile (%s)!" % infile
        print(err)
        return
    line = line.split()
    # # read in the error ellipsoid representation of the location error.
    # # this is given as azimuth/dip/length of axis 1 and 2 and as length
    # # of axis 3.
    # azim1 = float(line[20])
    # dip1 = float(line[22])
    # len1 = float(line[24])
    # azim2 = float(line[26])
    # dip2 = float(line[28])
    # len2 = float(line[30])
    # len3 = float(line[32])
    #
    # # XXX TODO save original nlloc error ellipse?!
    # # errX, errY, errZ = errorEllipsoid2CartesianErrors(azim1, dip1, len1,
    # #                                                   azim2, dip2, len2,
    # #                                                   len3)
    # # NLLOC uses error ellipsoid for 68% confidence interval relating to
    # # one standard deviation in the normal distribution.
    # # We multiply all errors by 2 to approximately get the 95% confidence
    # # level (two standard deviations)...
    # errX *= 2
    # errY *= 2
    # errZ *= 2
    # if location == 'SURF':
    #     # CJH Now descale to correct dimensions
    #     errX /= 100
    #     errY /= 100
    #     errZ /= 100
    # Take covariance approach from obspy
    covariance_xx = float(line[8])
    covariance_yy = float(line[14])
    covariance_zz = float(line[18])
    # determine which model was used:
    # XXX handling of path extremely hackish! to be improved!!
    dirname = os.path.dirname(infile)
    controlfile = os.path.join(dirname, "last.in")
    lines2 = open(controlfile, "rt").readlines()
    line2 = lines2.pop()
    while not line2.startswith("LOCFILES"):
        line2 = lines2.pop()
    line2 = line2.split()
    model = line2[3]
    model = model.split("/")[-1]
    event = ev
    if event.creation_info is None:
        event.creation_info = CreationInfo()
        event.creation_info.creation_time = UTCDateTime()
    o = Origin()
    event.origins = [o]
    # event.set_creation_info_username('cjhopp')
    # version field has 64 char maximum per QuakeML RNG schema
    o.creation_info = CreationInfo(creation_time=creation_time,
                                   version=nlloc_version[:64])
    # assign origin info
    o.method_id = "/".join(["smi:de.erdbeben-in-bayern", "location_method",
                            "nlloc", "7"])
    o.latitude = lat[0]
    o.longitude = lon[0]
    o.depth = depth
    if location in ['SURF', 'FSB']:
        print('Creating origin uncertainty')
        o.longitude = lon[0]
        o.latitude = lat[0]
        print('Assigning depth {}'.format(depth))
        o.depth = depth# * (-1e3)  # meters positive down!
        print('Creating extra AttribDict')
        # Attribute dict for actual hmc coords
        if location == 'FSB':
            extra = AttribDict({
                'ch1903_east': {
                    'value': 2579255. + (x * 10),
                    'namespace': 'smi:local/ch1903'
                },
                'ch1903_north': {
                    'value': 1247501. + (y * 10),
                    'namespace': 'smi:local/ch1903'
                },
                'ch1903_elev': {
                    'value': 547. - depth, # Extra attribs maintain absolute elevation
                    'namespace': 'smi:local/ch1903'
                }
            })
        else:
            extra = AttribDict({
                'hmc_east': {
                    'value': x * 10,
                    'namespace': 'smi:local/hmc'
                },
                'hmc_north': {
                    'value': y * 10,
                    'namespace': 'smi:local/hmc'
                },
                'hmc_elev': {
                    'value': 130 - depth, # Extra attribs maintain absolute elevation
                    'namespace': 'smi:local/hmc'
                }
            })
        o.extra = extra
    o.origin_uncertainty = OriginUncertainty()
    o.quality = OriginQuality()
    ou = o.origin_uncertainty
    oq = o.quality
    # negative values can appear on diagonal of covariance matrix due to a
    # precision problem in NLLoc implementation when location coordinates are
    # large compared to the covariances.
    try:
        o.longitude_errors.uncertainty = kilometer2degrees(np.sqrt(covariance_xx))
    except ValueError:
        if covariance_xx < 0:
            msg = ("Negative value in XX value of covariance matrix, not "
                   "setting longitude error (epicentral uncertainties will "
                   "still be set in origin uncertainty).")
            warnings.warn(msg)
        else:
            raise
    try:
        o.latitude_errors.uncertainty = kilometer2degrees(np.sqrt(covariance_yy))
    except ValueError:
        if covariance_yy < 0:
            msg = ("Negative value in YY value of covariance matrix, not "
                   "setting longitude error (epicentral uncertainties will "
                   "still be set in origin uncertainty).")
            warnings.warn(msg)
        else:
            raise
    o.depth_errors.uncertainty = np.sqrt(covariance_zz) * 1e3  # meters!
    o.depth_errors.confidence_level = 68
    o.depth_type = str("from location")
    # if errY > errX:
    #     ou.azimuth_max_horizontal_uncertainty = 0
    # else:
    #     ou.azimuth_max_horizontal_uncertainty = 90
    # ou.min_horizontal_uncertainty, \
    #         ou.max_horizontal_uncertainty = \
    #         sorted([errX * 1e3, errY * 1e3])
    # ou.preferred_description = "uncertainty ellipse"
    # o.depth_errors.uncertainty = errZ * 1e3
    oq.standard_error = rms #XXX stimmt diese Zuordnung!!!?!
    oq.azimuthal_gap = gap
    # o.depth_type = "from location"
    o.earth_model_id = "%s/earth_model/%s" % ("smi:de.erdbeben-in-bayern",
                                              model)
    o.time = time
    # goto synthetic phases info lines
    try:
        line = lines.pop(0)
        while not line.startswith("PHASE ID"):
            line = lines.pop(0)
    except:
        err = "Error: No correct synthetic phase info found in NLLoc " + \
              "outputfile (%s)!" % infile
        print(err)
        return

    # remove all non phase-info-lines from bottom of list
    try:
        badline = lines.pop()
        while not badline.startswith("END_PHASE"):
            badline = lines.pop()
    except:
        err = "Error: Could not remove unwanted lines at bottom of " + \
              "NLLoc outputfile (%s)!" % infile
        print(err)
        return

    o.quality.used_phase_count = 0
    o.quality.extra = AttribDict()
    o.quality.extra.usedPhaseCountP = {'value': 0,
                                       'namespace': "http://erdbeben-in-bayern.de/xmlns/0.1"}
    o.quality.extra.usedPhaseCountS = {'value': 0,
                                       'namespace': "http://erdbeben-in-bayern.de/xmlns/0.1"}

    # go through all phase info lines
    """
    Order of fields:
    ID Ins Cmp On Pha FM Q Date HrMn Sec Coda Amp Per PriorWt > Err ErrMag
    TTpred Res Weight StaLoc(X Y Z) SDist SAzim RAz RDip RQual Tcorr
    TTerrTcorr

    Fields:
    ID (char*6)
        station name or code
    Ins (char*4)
        instrument identification for the trace for which the time pick corresponds (i.e. SP, BRB, VBB)
    Cmp (char*4)
        component identification for the trace for which the time pick corresponds (i.e. Z, N, E, H)
    On (char*1)
        description of P phase arrival onset; i, e
    Pha (char*6)
        Phase identification (i.e. P, S, PmP)
    FM (char*1)
        first motion direction of P arrival; c, C, u, U = compression; d, D = dilatation; +, -, Z, N; . or ? = not readable.
    Date (yyyymmdd) (int*6)
        year (with century), month, day
    HrMn (hhmm) (int*4)
        Hour, min
    Sec (float*7.4)
        seconds of phase arrival
    Err (char*3)
        Error/uncertainty type; GAU
    ErrMag (expFloat*9.2)
        Error/uncertainty magnitude in seconds
    Coda (expFloat*9.2)
        coda duration reading
    Amp (expFloat*9.2)
        Maxumim peak-to-peak amplitude
    Per (expFloat*9.2)
        Period of amplitude reading
    PriorWt (expFloat*9.2)
        A-priori phase weight
    > (char*1)
        Required separator between first part (observations) and second part (calculated values) of phase record.
    TTpred (float*9.4)
        Predicted travel time
    Res (float*9.4)
        Residual (observed - predicted arrival time)
    Weight (float*9.4)
        Phase weight (covariance matrix weight for LOCMETH GAU_ANALYTIC, posterior weight for LOCMETH EDT EDT_OT_WT)
    StaLoc(X Y Z) (3 * float*9.4)
        Non-GLOBAL: x, y, z location of station in transformed, rectangular coordinates
        GLOBAL: longitude, latitude, z location of station
    SDist (float*9.4)
        Maximum likelihood hypocenter to station epicentral distance in kilometers
    SAzim (float*6.2)
        Maximum likelihood hypocenter to station epicentral azimuth in degrees CW from North
    RAz (float*5.1)
        Ray take-off azimuth at maximum likelihood hypocenter in degrees CW from North
    RDip (float*5.1)
        Ray take-off dip at maximum likelihood hypocenter in degrees upwards from vertical down (0 = down, 180 = up)
    RQual (float*5.1)
        Quality of take-off angle estimation (0 = unreliable, 10 = best)
    Tcorr (float*9.4)
        Time correction (station delay) used for location
    TTerr (expFloat*9.2)
        Traveltime error used for location
    """
    used_stations = set()
    for line in lines:
        line = line.split()
        # check which type of phase
        if line[4] == "P":
            type = "P"
        elif line[4] == "S":
            type = "S"
        else:
            print("Encountered a phase that is not P and not S!! "
                  "This case is not handled yet in reading NLLOC "
                  "output...")
            continue
        # get values from line
        station = line[0]
        epidist = float(line[21])
        azimuth = float(line[23])
        ray_dip = float(line[24])
        # if we do the location on traveltime-grids without angle-grids we
        # do not get ray azimuth/incidence. but we can at least use the
        # station to hypocenter azimuth which is very close (~2 deg) to the
        # ray azimuth
        if azimuth == 0.0 and ray_dip == 0.0:
            azimuth = float(line[22])
            ray_dip = np.nan
        if line[3] == "I":
            onset = "impulsive"
        elif line[3] == "E":
            onset = "emergent"
        else:
            onset = None
        if line[5] == "U":
            polarity = "positive"
        elif line[5] == "D":
            polarity = "negative"
        else:
            polarity = None
        # predicted travel time is zero.
        # seems to happen when no travel time cube is present for a
        # provided station reading. show an error message and skip this
        # arrival.
        if float(line[15]) == 0.0:
            msg = ("Predicted travel time for station '%s' is zero. "
                   "Most likely the travel time cube is missing for "
                   "this station! Skipping arrival for this station.")
            print(msg % station)
            continue
        res = float(line[16])
        weight = float(line[17])

        # assign synthetic phase info
        pick = [p for p in ev.picks if p.waveform_id.station_code == station
                and p.phase_hint == type]
        if station.startswith('NSMT'):
            pick = [p for p in ev.picks
                    if p.waveform_id.station_code == 'NSMTC'
                    and p.waveform_id.location_code == station[-2:]
                    and p.phase_hint == type]
        if len(pick) == 0:
            msg = "This should not happen! Location output was read and a corresponding pick is missing!"
            raise NotImplementedError(msg)
        arrival = Arrival(pick_id=pick[0].resource_id.id)
        o.arrivals.append(arrival)
        # residual is defined as P-Psynth by NLLOC!
        arrival.distance = kilometer2degrees(epidist)
        arrival.phase = type
        arrival.time_residual = res
        if location in ['SURF', 'FSB']:
            arrival.time_residual = res / 1000. # CJH descale time too (why 1000)??
        arrival.azimuth = azimuth
        if not np.isnan(ray_dip):
            arrival.takeoff_angle = ray_dip
        if onset and not pick.onset:
            pick.onset = onset
        if polarity and not pick.polarity:
            pick.polarity = polarity
        # we use weights 0,1,2,3 but NLLoc outputs floats...
        arrival.time_weight = weight
        o.quality.used_phase_count += 1
        if type == "P":
            o.quality.extra.usedPhaseCountP['value'] += 1
        elif type == "S":
            o.quality.extra.usedPhaseCountS['value'] += 1
        else:
            print("Phase '%s' not recognized as P or S. " % type +
                  "Not incrementing P nor S phase count.")
        used_stations.add(station)
    o.used_station_count = len(used_stations)
    try:
        update_origin_azimuthal_gap(ev)
    except IndexError as e:
        print('Invalid resource ids breaking Arrival-->Pick lookup')
    print('Made it through location reading')
    # read NLLOC scatter file
    data = readNLLocScatter(infile.replace('hyp', 'scat'), location)
    print('Read in scatter')
    o.nonlinloc_scatter = data

def getPickForArrival(picks, arrival):
    """
    searches list of picks for a pick that matches the arrivals pick_id
    and returns it (empty Pick object otherwise).
    """
    pick = None
    for p in picks:
        if arrival.pick_id == p.resource_id:
            pick = p
            break
    return pick

def update_origin_azimuthal_gap(ev):
    origin = ev.origins[0]
    arrivals = origin.arrivals
    picks = ev.picks
    azims = {}
    for a in arrivals:
        p = getPickForArrival(picks, a)
        if p is None:
            msg = ("Could not find pick for arrival. Aborting calculation "
                   "of azimuthal gap.")
            print(msg)
            return
        netsta = ".".join([p.waveform_id.network_code, p.waveform_id.station_code])
        azim = a.azimuth
        if azim is None:
            msg = ("Arrival's azimuth is 'None'. "
                   "Calculated azimuthal gap might be wrong")
            print(msg)
        else:
            azims.setdefault(netsta, []).append(azim)
    azim_list = []
    for netsta in azims:
        tmp_list = azims.get(netsta, [])
        if not tmp_list:
            msg = ("No azimuth information for station %s. "
                   "Aborting calculation of azimuthal gap.")
            print(msg)
            return
        azim_list.append((np.median(tmp_list), netsta))
    azim_list = sorted(azim_list)
    azims = np.array([azim for azim, netsta in azim_list])
    azims.sort()
    # calculate azimuthal gap
    gaps = azims - np.roll(azims, 1)
    gaps[0] += 360.0
    gap = gaps.max()
    i_ = gaps.argmax()
    if origin.quality is None:
        origin.quality = OriginQuality()
    origin.quality.azimuthal_gap = gap
    # calculate secondary azimuthal gap
    gaps = azims - np.roll(azims, 2)
    gaps[0] += 360.0
    gaps[1] += 360.0
    gap = gaps.max()
    origin.quality.secondary_azimuthal_gap = gap

def getPick(event, network=None, station=None, phase_hint=None,
            waveform_id=None, seed_string=None):
    """
    returns first matching pick, does NOT ensure there is only one!
    if setdefault is True then if no pick is found an empty one is returned and inserted into self.picks.
    """
    picks = event.picks
    for p in picks:
        if network is not None and network != p.waveform_id.network_code:
            continue
        if station is not None and station != p.waveform_id.station_code:
            continue
        if phase_hint is not None and phase_hint != p.phase_hint:
            continue
        if waveform_id is not None and waveform_id != p.waveform_id:
            continue
        if seed_string is not None and seed_string != p.waveform_id.get_seed_string():
            continue
    return p

def readNLLocScatter(scat_filename, location):
    """
    ****
    Stolen from obspyck
    ****

    This function reads location and values of pdf scatter samples from the
    specified NLLoc *.scat binary file (type "<f4", 4 header values, then 4
    floats per sample: x, y, z, pdf value) and converts X/Y Gauss-Krueger
    coordinates (zone 4, central meridian 12 deg) to Longitude/Latitude in
    WGS84 reference ellipsoid.
    Messages on stderr are written to specified GUI textview.
    Returns an array of xy pairs.
    """
    # read data, omit the first 4 values (header information) and reshape
    print('Reading scatter')
    data = np.fromfile(scat_filename, dtype="<f4").astype("float")[4:]
    # Explicit floor divide or float--> integer error
    print('Reshaping')
    data = data.reshape((data.shape[0] // 4, 4)).swapaxes(0, 1)
    # data[0], data[1] = gk2lonlat(data[0], data[1])
    print('Converting scatter coords')
    if location == 'SURF':
        data[0], data[1] = surf_xyz2latlon(data[0], data[1])
        data[2] *= 10
        data[2] = 130 - data[2]
    elif location == 'FSB': # go straight to ch1903 for this
        print(data.shape)
        data[0] *= 10
        data[0] = 2579255. + data[0]
        data[1] *= 10
        data[1] = 1247501. + data[1]
        data[2] *= 10
        data[2] = 547 - data[2]
    # Descale depth too and convert to m (* 100 / 1000 = * 10)
    return data.T


def errorEllipsoid2CartesianErrors(azimuth1, dip1, len1, azimuth2, dip2, len2,
                                   len3):
    """
    This method converts the location error of NLLoc given as the 3D error
    ellipsoid (two azimuths, two dips and three axis lengths) to a cartesian
    representation.
    We calculate the cartesian representation of each of the ellipsoids three
    eigenvectors and use the maximum of these vectors components on every axis.
    """
    z = len1 * np.sin(np.radians(dip1))
    xy = len1 * np.cos(np.radians(dip1))
    x = xy * np.sin(np.radians(azimuth1))
    y = xy * np.cos(np.radians(azimuth1))
    v1 = np.array([x, y, z])

    z = len2 * np.sin(np.radians(dip2))
    xy = len2 * np.cos(np.radians(dip2))
    x = xy * np.sin(np.radians(azimuth2))
    y = xy * np.cos(np.radians(azimuth2))
    v2 = np.array([x, y, z])

    v3 = np.cross(v1, v2)
    v3 /= np.sqrt(np.dot(v3, v3))
    v3 *= len3

    v1 = np.abs(v1)
    v2 = np.abs(v2)
    v3 = np.abs(v3)

    error_x = max([v1[0], v2[0], v3[0]])
    error_y = max([v1[1], v2[1], v3[1]])
    error_z = max([v1[2], v2[2], v3[2]])

    return (error_x, error_y, error_z)


def dd_time2EQ(catalog, nlloc_root, in_file):
    """
    Takes a catalog with hypoDD-defined origins and populates the arrivals
    attribute for that origin using specified NLLoc Grid files through
    time2EQ

    :param catalog: Catalog containing events which we need Arrivals for
    :param nlloc_root: Root directory for file IO
    :param in_file: NLLoc/Time2EQ run file. User is responsible for defining
        the path to grid files in this control file. This file will be modified
        in-place as this function runs.
    :return:
    """
    # Temp ctrl file overwritten each iteration
    new_ctrl = '{}.new'.format(in_file)
    for ev in catalog:
        eid = ev.resource_id.id.split('/')[-1]
        o = ev.preferred_origin()
        if not o or not o.method_id:
            print('Preferred origin not DD: {}'.format(eid))
            continue
        if len(o.arrivals) > 0:
            print('DD origin has some Arrivals. '
                  + 'Removing and adding again.')
            o.arrivals = []
        print('Raytracing for: {}'.format(eid))
        obs_file = '{}/obs/{}'.format(nlloc_root, eid)
        new_obs = '{}.obs'.format(obs_file) # Only real picks in this one
        print(new_obs)
        loc_file = '{}/loc/{}'.format(nlloc_root, eid)
        out_file_hyp = glob(
            '{}.????????.??????.grid0.loc.hyp'.format(loc_file))
        # Edit the ctrl file for both Time2EQ and NLLoc statements
        if len(out_file_hyp) == 0:
            with open(in_file, 'r') as f, open(new_ctrl, 'w') as fo:
                for line in f:
                    # Time2EQ
                    if line.startswith('EQFILES'):
                        line = line.split()
                        line = '{} {} {}\n'.format(line[0], line[1], obs_file)
                    elif line.startswith("EQSRCE"):
                        line = "EQSRCE {} LATLON {} {} {} 0.0\n".format(
                            eid, o.latitude, o.longitude, o.depth / 1000.)
                    # NLLoc
                    elif line.startswith('LOCFILES'):
                        ln = line.split()
                        line = ' '.join([ln[0], new_obs, ln[2],
                                         ln[3], loc_file])
                    fo.write(line)
            call(["Time2EQ", new_ctrl])
            # Edit obs_file to have just the Time2EQ phases for which we
            # have picks!
            # Make list of sta.phase
            sta_phz = {'{}.{}'.format(pk.waveform_id.station_code,
                                      pk.phase_hint): pk
                       for pk in ev.picks}
            # Also will add the polarities in here to eliminate separate func
            with open(obs_file, 'r') as of, open(new_obs, 'w') as nof:
                for line in of:
                    ln = line.split()
                    # Write the first line
                    if ln[0] == '#':
                        nof.write(' '.join(ln) + '\n')
                        continue
                    staph = '{}.{}'.format(ln[0], ln[4])
                    # Now only write phases we picked to the obs file
                    if staph in sta_phz:
                        if sta_phz[staph].polarity == 'positive':
                            ln[5] = 'U'
                        elif sta_phz[staph].polarity == 'negative':
                            ln[5] = 'D'
                        nof.write(' '.join(ln) + '\n')
            call(["NLLoc", new_ctrl])
            out_file_hyp = glob(
                '{}.????????.??????.grid0.loc.hyp'.format(loc_file))
            if len(out_file_hyp) == 0:
                print('No observations produced. Skip.')
                continue
        pk_stas = [pk.waveform_id.station_code for pk in ev.picks]
        # Instead of using the obspy 'read_nlloc_hyp' method, like above,
        # we'll just take the toa and dip from the phases. There was some
        # weirdness with bad microseconds being read into datetime objs
        # possibly linked to origins at 1900?
        try:
            with open(out_file_hyp[0], 'r') as f:
                for i, line in enumerate(f):
                    if (i > 15 and not line.startswith('END')
                        and not line.startswith('\n')):
                        ln = line.split()
                        pha = ln[4]
                        sta = ln[0]
                        dist = kilometer2degrees(float(ln[-6]))
                        if sta not in pk_stas:
                            continue
                        toa = ln[-3]
                        to_az = ln[-4]
                        try:
                            pk = [pk for pk in ev.picks
                                  if pk.waveform_id.station_code == sta][0]
                        except IndexError:
                            continue
                        ev.preferred_origin().arrivals.append(
                            Arrival(phase=pha, pick_id=pk.resource_id.id,
                                    takeoff_angle=toa, azimuth=to_az,
                                    distance=dist))
        except:
            print('Issue opening file. Event may not have been located')
            continue
    return

def write_xyz(cat, outfile):
    import csv
    with open(outfile, 'wb') as f:
        writer = csv.writer(f, delimiter=' ')
        for ev in cat:
            if ev.preferred_origin():
                writer.writerow([ev.preferred_origin().latitude,
                                 ev.preferred_origin().longitude,
                                 ev.preferred_origin().depth / 1000])

############## GrowClust Functions ############################################

def hypoDD_to_GrowClust(in_dir, out_dir):
    """
    Helper to take input files from hypoDD and convert them for use with
    GrowClust

    :param in_dir: Path to the HypoDD input directory
    :param out_dir: Path to the GrowClust input directory
    :return:
    """
    # First, convert phase.dat to evlist.txt
    with open('{}/phase.dat'.format(in_dir), 'r') as in_f:
        with open('{}/evlist.txt'.format(out_dir), 'w') as out_f:
            for ln in in_f:
                if ln.startswith('#'):
                    out_f.write('{}\n'.format(' '.join(ln.split()[1:])))
    # Now remove occurrences of network string from dt.cc and write to
    # xcordata.txt (use sed via system call as much faster)
    sed_str = "sed 's/NZ.//g' {}/dt.cc > {}/xcordata.txt".format(in_dir,
                                                                 out_dir)
    call(sed_str, shell=True)
    return

def GrowClust_to_Catalog(hypoDD_cat, out_dir):
    """
    Take the original catalog used in generating dt's with HypoDDpy and read
    the output of GrowClust into the appropriate events as new origins.

    This is probably going to borrow heavily from hypoDDpy...
    :param hypoDD_cat: Same catalog used in hypoDDpy to generate dt's
    :param out_dir: GrowClust output directory
    :return:
    """
    # Catalog is sorted by time in hypoDDpy before event map is generated
    hypoDD_cat.events.sort(key=lambda x: x.preferred_origin().time)
    new_o_map = {}
    with open('{}/out.growclust_cat'.format(out_dir), 'r') as f:
        for ln in f:
            ln.strip()
            line = ln.split()
            # First determine if it was relocated
            # Default is line[19] == -1 for no, but also should beware of
            # unceratintites of 0.000. Deal with these later?
            eid = int(line[6]) # Event id before clustering
            if line[13] == '1' and line[19] == '-1.000':
                print('Event {} not relocated, keep original location'.format(
                    eid
                ))
                continue
            re_lat = float(line[7])
            re_lon = float(line[8])
            re_dep = float(line[9]) * 1000 # meters bsl
            x_uncert = float(line[19]) * 1000 # in m
            z_uncert = float(line[20]) * 1000 # in m
            t_uncert = float(line[21])
            o_uncert = OriginUncertainty(horizontal_uncertainty=x_uncert)
            t_uncert = QuantityError(uncertainty=t_uncert)
            d_uncert = QuantityError(uncertainty=z_uncert)
            sec = int(line[5].split('.')[0])
            microsec = int(line[5].split('.')[1]) * 1000
            method_id = ResourceIdentifier(id='GrowClust')
            re_time = UTCDateTime(year=int(line[0]), month=int(line[1]),
                                  day=int(line[2]), hour=int(line[3]),
                                  minute=int(line[4]), second=sec,
                                  microsecond=microsec)
            new_o_map[eid] = Origin(time=re_time, latitude=re_lat,
                                    longitude=re_lon, depth=re_dep,
                                    time_errors=t_uncert,
                                    depth_errors=d_uncert,
                                    origin_uncertainty=o_uncert,
                                    method_id=method_id)
    for i, ev in enumerate(hypoDD_cat):
        id = i + 1 # Python indexing
        if id in new_o_map:
            ev.origins.append(new_o_map[id])
            ev.preferred_origin_id = new_o_map[id].resource_id.id
    return hypoDD_cat