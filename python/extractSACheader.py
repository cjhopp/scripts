

# -*- coding: utf-8 -*-

"""
Created on Mon Oct 12 14:20:26 2015
@author: matsonga
"""

import os
from obspy.core import UTCDateTime
from obspy.sac import SacIO
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
templateDir = '/Volumes/GeoPhysics_07/users-data/matsonga/MRP_PROJ/data/mastersData/eventFilesLong' # masterevents/mastereventsLong'
phaseDict = {}
#slopeList = []
#x = []
#y = []
templateList = ['20120603_174213.7',
'20120607_001116.99',
'20120607_001625.42',
'20120607_001732.03',
#'20120607_053729.75',
'20120613_223617.92',
'20120613_225124.82',
'20120616_232228.44',
'20120617_125622.65',
'20120617_211506.44',
'20120618_141123.68',
'20120618_141211.79',
'20120618_141634.53',
'20120618_072456.48',
'20120618_072530.94',
'20120618_092539.81',
'20120619_105726.47',
'20120619_121536.76',
'20120619_123326.84',
'20120619_123423.28',
'20120619_123510.98',
'20120619_124550.66',
'20120619_051749.47',
'20120619_075137.67',
'20120619_091337.17',
'20120619_091806.39',
'20120619_091843.25',
'20120619_092028.18',
'20120620_060208.19',
'20120621_014144.60',
'20120621_150149.55',
'20120621_064345.06',
'20120621_092443.43',
'20120621_092523.79',
'20120621_092606.59',
'20120621_092706.50',
'20120622_103330.10',
'20120622_110447.19',
'20120622_150004.65',
'20120622_050507.42',
'20120622_050540.80',
'20120623_105752.73',
'20120623_204452.04',
'20120623_205144.06',
'20120623_093014.31',
'20120624_105819.68',
'20120624_110756.70',
'20120624_110934.39',
'20120625_145647.91',
'20120625_145704.48',
'20120625_145733.10',
'20120706_104201.36',
'20120706_104804.33',
'20120706_111838.98',
'20120706_112004.97',
'20120706_112221.74',
'20120706_154145.36',
'20120706_174935.00',
'20120706_222526.92',
'20120706_074856.48',
'20120707_163319.35',
'20120707_182712.24',
'20120707_190437.66',
'20120707_190850.80',
'20120707_191133.92',
'20120707_191948.84',
'20120707_193539.24',
'20120707_193948.72',
'20120707_024456.53',
'20120707_233920.22',
'20120707_234043.78',
'20120707_090221.56',
'20120708_191025.56',
'20120708_043744.85',
'20120716_182108.48',
'20120717_001200.59',
'20120717_062750.51',
'20120717_062828.21',
'20120717_085617.24',
'20120717_090206.13',
'20120717_090553.93',
'20120719_233504.76',
'20120719_083827.32',
'20120720_135427.53',
'20120720_140218.69',
'20120720_155202.74',
'20120720_160128.49',
'20120720_161516.86',
'20120720_161600.44',
'20120720_161805.38',
'20120720_162033.27',
'20120720_162316.01',
'20120720_205553.52',
'20120720_224616.56',
'20120720_225051.60',
'20120720_225415.06',
'20120720_071900.76',
'20120723_024610.97',
'20120723_030249.99',
'20120723_042004.52',
'20120726_140110.08',
'20120726_050847.66',
'20120726_054652.09',
'20120727_110406.80',
'20120727_093231.77',
'20120727_094956.83',
'20120728_220530.88',
'20120728_063210.12',
'20120728_075242.95',
'20120728_080729.19',
'20120728_091809.74']

for template in templateList:
    templateDateTime = UTCDateTime(template) # converts the event name to a datetime class
    templateJulDay = templateDateTime.julday # int
    templateYear = templateDateTime.year # int
    backgroundTime = 10 #Amount of time in longer event file before event origin time in seconds
    # phaseMarkerList = []
    # staList = []
    # compList = []
    phaseDict[template] = {}
#########################################################
for file in os.listdir('%s/event%s' %(templateDir,template)):
    os.chdir('%s/event%s' %(templateDir,template))
    if file.endswith('.SAC'):
        file1 = file
        station = file1[3:7]
        component = file1[13]
        quality_amarker=str(SacIO().GetHvalueFromFile(file1,'ka')) #P phase
        aMarker = SacIO().GetHvalueFromFile(file1,'a')#P phase
        quality_t0marker=str(SacIO().GetHvalueFromFile(file1,'kt0')) #S phase
        t0Marker = SacIO().GetHvalueFromFile(file1,'t0')#S phase
        if float(aMarker)!=-12345.0:
            if (not phaseDict[template].has_key(station)):
                phaseDict[template][station]=[]
                phaseDict[template][station].append(aMarker)
                aMarkerDateTime = templateDateTime - backgroundTime + aMarker
                phaseDict[template][station].append(aMarkerDateTime)
            elif float(t0Marker)!=-12345.0 and component =='E':
                if (not phaseDict[template].has_key(station)):
                    phaseDict[template][station]=[]
                    phaseDict[template][station].append(t0Marker)
                    t0MarkerDateTime = templateDateTime - backgroundTime + t0Marker
                    phaseDict[template][station].append(t0MarkerDateTime)
os.chdir(templateDir)
outputFilename = 'templatePhaseTimes.txt'
fid2 = open(outputFilename,'w')
for templateKey in phaseDict:
for key in phaseDict[templateKey]:
if len(phaseDict[templateKey][key])==2:
fid2.write('%s %s %s %s\n' %(templateKey,key,phaseDict[templateKey][key][0],phaseDict[templateKey][key][1]))
fid2.close()
