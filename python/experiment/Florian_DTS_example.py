import numpy as np

from lxml import etree
from lxml.etree import XMLSyntaxError
from obspy import UTCDateTime

def read_XTDTS(path, no_cols):
    # Read single xml file and return array for all values and time
    try:
        dts = etree.parse(path)
    except XMLSyntaxError as e:
        return None
    # Get root element
    root = dts.getroot()
    # Create one string for all values, comma sep
    measurements = np.fromstring(','.join(
        [l.text.replace('\n', '')
         for l in root[0].find('{*}logData').findall('{*}data')]),
        sep=',')
    # 6 columns in original data
    measurements = measurements.reshape(-1, no_cols)
    # Get time
    dto = UTCDateTime(root[0].find('{*}endDateTimeIndex').text).datetime
    return dto, measurements
