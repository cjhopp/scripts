import pytz

import numpy as np
import struct as st
import datetime
import pandas as pd

from obspy import Stream, Trace
from obspy.core.trace import Stats

class MirfFile:
    """
    A class that parses MIRF (v7) files.
    
    :param filename: Create the class using a MIRF file, defaults to None.
    :type filename: string, optional
        
    :examples:
    
    The following creates a MirfFile class based on an MIRF file.
    
    >>> MirfFile('f_0001.rcd')
    
    :methods:
    """
    
    def __init__(self,filename = None):
        
        if filename:
            self.open_file(filename)
        else:
            self._set_header()
            
    def __str__(self):
        return "<MIRF file: timestamp {}, record {}, {} levels>".format(
            self.time, self.record_num, self.receivers)
        
        
    def open_file(self, filename):
        """
        Opens and parses a MIRFv7 file into this class.
        
        :param filename: the file name to open
        :type filename: string
        """
         
        with open(filename,'rb') as f:
            self._raw_header = f.read(512)
            self._decode_header()
            
            self._channel_headers = f.read(self.channel_count * 64)
            self._decode_channels()
            self._decode_data(f)
            
    def get_data(self,chn=7):
        """
        Returns the channel data (in mV) with its time vector.
        
        :param chn: Channel from which to retrieve the data. Defaults to channel
                    7.
        :type chn: int, optional
        """
        
        fs = 1/self.sample_period * 1e6
        t = np.arange(self.channels[chn-1].N)/fs
        
        return (t, self.channels[chn-1].data * 1e3)
    
    def print_channels(self,show_empty=False):
        """
        Print the number of channels in the datafile.
        
        :param show_empty: Show channels that contain no data.
        :type show_empty: bool, optional
        """
        for ch in self.channels:
            
            if ch.N or show_empty:
                print(ch)
            
    
    def _set_header(self):
        """
        Write the MIRF file header, some integer representations are listed below.
        
        .. list-table:: MIRF file types
            :header-rows: 1
            
            * - Code
              - Description
            * - 0
              - Raw Data
            * - 1
              - Correlated data
                
        .. list-table:: MIRF file format codes
           :header-rows: 1
           
           * - Code
             - Description
           * - -1
             - Format varies (format obtained from channel header)
           * - 0
             - DAQ IFP (obselete)
           * - 1
             - Multilock IFP (obselete)
           * - 2
             - Geochain IFP
           * - 3
             - 32-bit Integer
           * - 4
             - 32-bit IEEE754 floating point
           * - 5
             - 24-bit Integer
           * - 6
             - 16-bit Integer
           * - 7
             - 16-bit Delta IFP (obselete)
             
        
        """
        
        # Set the file specification version
        self.version = 7
        
        #: MIRF file type
        #: 0 = Raw data
        #: 1 = Stacked data
        self.file_type = 0
        
        # Set the data format code (-1 is for varied formats, use channel header)
        self.format_code = -1
        
        # Correlated data flag (0: Raw data, 1: Correlated data)
        self.correlation_flag = 0
        
        # Set the controller type (10: DAQ, 40: GSP, 60: Sim, 70: Monitor)
        self.controller_type = 10
        
        # Set the tool system (0: Analog, 3: Geochain)
        self.tool_system = 3
        
        self.channel_count = 10
        
        self.test_mode = 0
        
        self.dataset = 1
        
        self.time = datetime.datetime.now()
        
        self.source = 1
        
        self.receivers = 1
        
        self.receiver_number_obselete = 1 # Obselete
        
        self.stack_records = 1
        
        # Measurement units (1: mm, 2: mft)
        self.units = 1
        
        # Receiver polarity (0: undefined, 1: SEG normal, 2: SEG reversed)
        self.polarity = 0
        
        self.source_ref = 1
        
        self.record_num = 1
        
        self.stack_num = 1
        
        self.fixn_num = 0
        
        self.tool_depth = 1000
        
        # Sample interval (1/fs)
        self.sample_period = 100 # microseconds
        
        self.line_number = 0
        
        self.gun_pressure = 0
        
        self.software_version = 510
        
        self.scx = 0
        self.scy = 0
        
        self.tcx = 0
        self.tcy = 0
        
        self.wref = 0
        
        self.sref = 0
        
        self.src_depth = 0
        
        self.src_to_mon = 0
        
        self.sref_error = 0
        
        self.ex_ref_delay = 0 # microseconds
        
        self.time_correction = 0
        
        self.timebreak_delay = 0
        
        self.tool_skew = 0
        
        self.gnss_scx_10 = 0
        self.gnss_scy_10 = 0
        
        self.controller_second = 0
        self.controller_us = 0
        
        # Timestamp mode (0: local time from PC, 1: UTC time from PC, 2: GNSS)
        self.timestamp_mode = 0
        
        self.sde = 0
        
        # Error control (0: No control, 1: Parity control)
        self.error_control = 0
        
        # Set microseismic mode (0: Normal, 1: Microseismic)
        self.microseismic_mode = 0
        
        # Typically 0 for normal operation, 1 for microseismic
        self.overlap_samples = 0
        
        self.last_gps_msg = ""
        
        self.reserved = ""
        
    def _decode_header(self):
        
        fields = st.unpack('54i108s188s', self._raw_header)
        
#         for i,d in enumerate(fields):
#             print("{:02d}: {}".format(i,d))
        
        self.version = fields[0]
        self.file_type = fields[1]
        self.format_code = fields[2]
        self.correlation_flag = fields[3]
        self.controller_type = fields[4]
        self.tool_system = fields[5]
        self.channel_count = fields[6]
        self.test_mode = fields[7]
        self.dataset = fields[8]
        print('foo')
        class TZ(datetime.tzinfo):
            def utcoffset(self, dt):
                return datetime.timedelta(seconds=fields[15])
        print(fields[9:15])
        self.time = datetime.datetime(fields[9], fields[10], fields[11],
                                      fields[12], fields[13], fields[14],
                                      tzinfo=TZ())#pytz.timezone('Europe/London'))
        print(self.time)
        self.source = fields[16]
        self.receivers = fields[17]
        self.receiver_number_obselete = fields[18]
        self.stack_records = fields[19]
        self.units = fields[20]
        self.polarity = fields[21]
        self.source_ref = fields[22]
        # Integer index 23 is reserved!
        self.record_num = fields[24]
        self.stack_num = fields[25]
        self.fixn_num = fields[26]
        self.tool_depth = fields[27]
        self.sample_period = fields[28] # microseconds
        self.line_number = fields[29]
        self.gun_pressure = fields[30]
        self.software_version = fields[31]
        self.scx = fields[32]
        self.scy = fields[33]
        self.tcx = fields[34]
        self.tcy = fields[35]
        self.wref = fields[36]
        self.sref = fields[37]
        self.src_depth = fields[38]
        self.src_to_mon = fields[39]
        self.sref_error = fields[40]
        self.ex_ref_delay = fields[41]
        self.time_correction = fields[42]
        self.timebreak_delay = fields[43]
        self.tool_skew = fields[44]
        self.gnss_scx_10 = fields[45]
        self.gnss_scy_10 = fields[46]
        self.controller_second = fields[47]
        self.controller_us = fields[48]
        self.timestamp_mode = fields[49]
        self.sde = fields[50]
        self.error_control = fields[51]
        self.microseismic_mode = fields[52]
        self.overlap_samples = fields[53]
        self.last_gps_msg = fields[54]
        self.reserved = fields[55]
        
    def _write_header(self):
        self._raw_header = st.pack('54i108s188s', 
               self.version,
               self.file_type,
               self.format_code,
               self.correlation_flag,
               self.controller_type,
               self.tool_system,
               self.channel_count,
               self.test_mode,
               self.dataset,
               self.time.year,
               self.time.month,
               self.time.day,
               self.time.hour,
               self.time.minute,
               self.time.second,
               self.time.timetz().utcoffset().seconds, # TODO: Fix offset
               self.source,
               self.receivers,
               self.receiver_number_obselete,
               self.stack_records,
               self.units,
               self.polarity,
               self.source_ref,
               0,
               self.record_num,
               self.stack_num,
               self.fixn_num,
               self.tool_depth,
               self.sample_period,
               self.line_number,
               self.gun_pressure,
               self.software_version,
               self.scx, self.scy,
               self.tcx, self.tcy,
               self.wref, self.sref,
               self.src_depth, self.src_to_mon,
               self.sref_error, self.ex_ref_delay,
               self.time_correction,
               self.timebreak_delay,
               self.tool_skew,
               self.gnss_scx_10, self.gnss_scy_10,
               self.controller_second,
               self.controller_us,
               self.timestamp_mode,
               self.sde, self.error_control,
               self.microseismic_mode,
               self.overlap_samples,
               self.last_gps_msg,
               self.reserved)
    
    def _write_channels(self):
        self._channel_headers = b''
        for ch in self.channels:
            self._channel_headers += st.pack('10ififfff',
                     ch.legacy_owner, ch._Channel__descriptor, ch._Channel__format_code,
                     ch.N, ch.pointer_us, ch._Channel__owner, ch.rcx, ch.rcy,
                     ch.vertical_depth, ch.depth_offset, ch.hsi, ch.reserved,
                     ch.sensor_scaling, ch.dc_offset, ch.scaling_factor,
                     ch.max_magnitude)
            
    def get_all_data(self):
        """
        Returns the data in all channels as a NumPy datablock with a time
        vector
        """
        ret = []
        for ch in self.channels:
            if ch.N:
                ret.append(ch.data)
        fs = self.sample_period / 1e6
        t = np.arange(ch.N) * fs
        time_vect = [self.time + datetime.timedelta(seconds=s) for s in t]
        return time_vect, np.vstack(ret).T

    def get_Stream(self):
        """
        Return an obspy stream
        """
        stas = ['RMNW.GC01..CHZ', 'RMNW.GC01..CHX', 'RMNW.GC01..CHY',
                'RMNW.GC02..CHZ', 'RMNW.GC02..CHX', 'RMNW.GC02..CHY',
                'RMNW.GC03..CHZ', 'RMNW.GC03..CHX', 'RMNW.GC03..CHY',
                'RMNW.GC04..CHZ', 'RMNW.GC04..CHX', 'RMNW.GC04..CHY']
        st = Stream()
        delta = self.sample_period / 1e6
        time_v, data_mat = self.get_all_data()
        for col in range(data_mat.shape[1]):
            seed = stas[col].split('.')
            tr = Trace(
                data=data_mat[:, col],
                header=Stats(dict(starttime=time_v[0], npts=data_mat.shape[0],
                                  delta=delta, network=seed[0],
                                  station=seed[1], location=seed[2],
                                  channel=seed[3])))
            st.traces.append(tr)
        return st

    def get_active_channels(self):
        
        ret = []
        
        for ch in self.channels:
            if ch.N:
                ret.append(ch)
        
        return ret
        
    
    def get_total_energies(self):
        
        dt = self.sample_period * 1e-6
        
        channels = self.get_active_channels()
        has_ref = len(channels) % 3
        
        data = []
        for ii in range(has_ref, len(channels), 3):
            data.append(np.sum(channels[ii].data**2 + channels[ii+1].data**2
                        + channels[ii+2].data**2))
        
        return np.vstack(data)
        
    class _Channel():
        """
        Represents a single channel in the file
        
        :param cid: The channel ID number
        :type cid: integer
        """
        
        def __init__(self,cid,raw_header):
            fields = st.unpack('10ififfff', raw_header)
            
            self.id = cid
            self.legacy_owner = fields[0]
            self.__descriptor = fields[1]
            self.__format_code = fields[2]
            self.N = fields[3]
            self.pointer_us = fields[4]
            self.__owner = fields[5]
            self.rcx = fields[6]
            self.rcy = fields[7]
            self.vertical_depth = fields[8]
            self.depth_offset = fields[9]
            self.hsi = fields[10]
            self.reserved = fields[11]
            self.sensor_scaling = fields[12]
            self.dc_offset = fields[13]
            self.scaling_factor = fields[14]
            self.max_magnitude = fields[15]
        
        @property
        def descriptor(self):
            "The type of data that is stored in the channel."
            values = ['Unknown data','VZ','HX','HY','DH','HYDRO','PILOT','TB',
                      'GF','RM','BP','PPS','VSIM','RSIM','NGEO','FGEO']
            return values[self.__descriptor]
        
        @property
        def format_code(self):
            "The data format of the channel stream."
            values = ['DAQ IFP', 'Multilock IFP', 'Geochain IFP',
                      '32-bit Int','32-bit Float',
                      '24-bit Int', '16-bit Int', '16-bit Delta IFP']
            return values[self.__format_code]
        
        @property
        def bytes_per_sample(self):
            "The number of bytes used per sample"
            values = [2,2,2,4,4,3,2,2]
            return values[self.__format_code]
        
        @property
        def owner(self):
            "The sonde that owns the channel (or declares a REF or AUX trace)."
            values = ['AUX','REF','NONE']
            if self.__owner < 1:
                return values[self.__owner+2]
            return self.__owner
            
        def __str__(self):
            return "<Data _Channel {}: Receiver {} {} ({} samples)>".format(
                self.id, self.owner, self.descriptor, self.N, self.format_code)
        
    def _decode_channels(self):
        self.channels = [self._Channel(i+1,self._channel_headers[x:x + 0x40]) for i,x in enumerate(range(0, len(self._channel_headers), 0x40))]
        
    def _decode_data(self,f):
        for ch in self.channels:
            if ch.N == 0: continue
            
            ch._raw_data = f.read(ch.N * ch.bytes_per_sample)
            
            if ch._Channel__format_code == 6:
                # 16-bit signed integer (easy)
                ch.data = (np.array(st.unpack('<h'*ch.N, ch._raw_data)) * ch.scaling_factor
                           - ch.dc_offset) * ch.sensor_scaling
                
            if ch._Channel__format_code == 5:
                # Horrible 24-bit format not easy for python
                ch.data = []
                for x in st.iter_unpack('>bH', ch._raw_data):
                    value = (x[0]<<16) + x[1]
                    
                    ch.data.append(value)
                ch.data = (np.array(ch.data) * ch.scaling_factor 
                           - ch.dc_offset) * ch.sensor_scaling
                # print("\nSanity check\nMaxMag: {:e}\nDecoded: {:e}".format(ch.max_magnitude,max(abs(ch.data))))
                
            if ch._Channel__format_code == 4:
                # 32-bit float (easy)
                ch.data = (np.array(st.unpack('<f'*ch.N, ch._raw_data)) * ch.scaling_factor
                           - ch.dc_offset) * ch.sensor_scaling
                           
            if ch._Channel__format_code == 3:
                # 32-bit integer (easy)
                ch.data = (np.array(st.unpack('<i'*ch.N, ch._raw_data)) * ch.scaling_factor
                           - ch.dc_offset) * ch.sensor_scaling
            
            if ch._Channel__format_code == 2:
                # ASL floating point format (efficient)
                ch.data = []
                for x in st.iter_unpack('<h', ch._raw_data):
                    power = x[0] & 0x3
                    value = (x[0] >> 2)/4**power
                    ch.data.append(value)
                
                ch.data = (np.array(ch.data) * ch.scaling_factor
                           -ch.dc_offset) * ch.sensor_scaling
                # print("\nSanity check\nMaxMag: {:e}\nDecoded: {:e}".format(ch.max_magnitude,max(abs(ch.data))))
                
def get_receiver_energy(mirf, receiver=1):
    
    misc_data = None
    z_data = None
    x_data = None
    y_data = None
    
    if not isinstance(mirf, MirfFile):
        raise Exception("Input to get_receiver_data must be an initialised MirfFile!")
    
    for ch in mirf.channels:
        if ch.owner == receiver:
            if ch._Channel__descriptor == 1:
                z_data = ch.data
                print("Found receiver {}'s Z data".format(receiver))
            if ch._Channel__descriptor == 2:
                x_data = ch.data * 1e3
            if ch._Channel__descriptor == 3:
                y_data = ch.data * 1e3
            if ch._Channel__descriptor > 3:
                misc_data = ch.data * 1e3
    
    if z_data is not None and x_data is not None  and y_data is not None:
        N = len(z_data)
        t = np.arange(N) * mirf.sample_period * 1e-6
        return pd.Series(z_data**2 + x_data**2 + y_data**2, index=t)
    if misc_data:
        return misc_data**2
    raise Exception("Receiver data not found!")
    