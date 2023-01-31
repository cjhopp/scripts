"""
Script to test the processing functionality

"""

import logging
from logging.handlers import RotatingFileHandler
import yaml
from dug_seis.processing.processing import processing
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(message)s')
verbose = False
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG if verbose else logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
log = 'dug-seis.log'
fh = RotatingFileHandler(log)
fh.setLevel(logging.DEBUG if verbose else logging.INFO)
fh.formatter = formatter
logger.addHandler(fh)
logger.info('DUG-Seis started')

f = open('dug-seis.yaml')
param = yaml.load(f)
param['General']['sensor_coords'] = np.reshape(param['General']['sensor_coords'],
                                               [param['General']['sensor_count'], 3])

processing(param)


