import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from obspy import Trace
from scipy import signal

sample_map = {'1': ['13', '18'],
              '2': ['9', '1', '14'],
              '3': ['15', '2', '10', '19'],
              '4': ['16', '3', '11'],
              '5': ['12', '4', '17', '20'],
              '6': ['5', '6', '7', '8']}

tilt_probes = ['5', '6', '7', '8']

upshift_probes = ['9', '10', '11', '12']
# All text files
data_files = glob('/home/chopp/Downloads/B_data/*.txt')
# print(data_files)

file_dict = {}
for fil in data_files:
    if 'Signal'in fil:
        continue
    no = fil.split('/')[-1].split()[-1].rstrip('.txt')
    file_dict[no] = {'extracted': fil}
sig_file_dict = {}
signal_files = glob('/home/chopp/Downloads/B_data/*Signal.txt')
for sfil in signal_files:
    if 'Signal'not in sfil:
        continue
    no = sfil.split('/')[-1].split()[-1].split('-')[0]
    # file_dict[no]['signal'] = sfil
    sig_file_dict[no] = {'signal': sfil}
print(file_dict)
print(sig_file_dict)
for sample_no, probes in sample_map.items():
    fig, ax = plt.subplots()
    for probe in probes:
        try:
            signal_data = pd.read_csv(sig_file_dict[probe]['signal'], sep='\s+', skiprows=3, names=('Time', 'Signal'))
        except KeyError:  # No data
            continue
        if probe in tilt_probes:
            tr = Trace(data=np.asarray(signal_data['Signal']))
            tr.stats.delta = 2e-9
            tr.plot()
        # tracedata = tr.data
        # print(np.where(tracedata))
        #########################################################
        extracted_data = pd.read_csv(file_dict[probe]['extracted'], sep='\s+', skiprows=2,
                                     names=('Time (s)', 'Velocity (m/s)', 'Vel err (m/s)'))
        if probe in upshift_probes:
            downshift = extracted_data['Velocity (m/s)'] - np.mean(extracted_data['Velocity (m/s)'][0:245])
            extracted_data['Corrected Velocity (m/s)'] = 0.7895 * downshift ** 0.9918
            extracted_data['Corrected Vel err (m/s)'] = 0.7895 * extracted_data['Vel err (m/s)'] ** 0.9918
        elif probe in tilt_probes:
            extracted_data['Corrected Velocity (m/s)'] = extracted_data['Velocity (m/s)']  ## no window correction applied
            extracted_data['Corrected Vel err (m/s)'] = extracted_data['Vel err (m/s)']
        else:
            extracted_data['Corrected Velocity (m/s)'] = 0.7895 * extracted_data['Velocity (m/s)'] ** 0.9918
            extracted_data['Corrected Vel err (m/s)'] = 0.7895 * extracted_data['Vel err (m/s)'] ** 0.9918
        extracted_data.to_csv('/home/chopp/Downloads/B_output_data/Sample_{}_Probe_{}.txt'.format(sample_no, probe))
        ax.plot(extracted_data['Time (s)'] * 10 ** 6, extracted_data['Corrected Velocity (m/s)'], label='P{}'.format(probe))
        ax.set_xlabel('Time ($\mu$s)')
        ax.set_ylabel('Particle Velocity (m/s)')
        ax.legend(frameon=False)
        ax.set_title('Sample {}'.format(sample_no))
        plt.tight_layout()
        plt.savefig('/home/chopp/Downloads/B_plots/Sample_{}.pdf'.format(sample_no))
