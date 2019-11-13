# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:35:52 2015

@author: matsonga
"""

import os
import matplotlib.pyplot as plt
import datetime
import numpy as np
#from obspy.core import UTCDateTime
#import matplotlib
#from datetime import strptime


startdate = datetime.date(year=2012,month=06,day=01)
numdays =61
dateList = []
for x in range(0,numdays):
    dateList.append(startdate +datetime.timedelta(days=x))
cluster1 = 'A'
cluster2 = 'B'
cluster3 = 'C'
threshMult = '7.6'

templateList = ['20120603_174213.7',
'20120607_001116.99',
'20120607_001625.42',
'20120607_001732.03',
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

multiplet1=["20120616_232228.44",
"20120617_125622.65",
"20120617_211506.44",
"20120618_072456.48",
"20120618_072530.94",
"20120618_141123.68",
"20120618_141211.79",
"20120618_141634.53",
"20120619_051749.47",
"20120619_091806.39",
"20120619_121536.76",
"20120619_123326.84",
"20120619_123423.28",
"20120619_123510.98",
"20120620_060208.19",
"20120621_014144.60",
"20120621_064345.06",
"20120621_092443.43",
"20120621_092523.79",
"20120621_092606.59",
"20120621_092706.50",
"20120621_150149.55",
"20120622_050507.42",
"20120622_050540.80",
"20120623_093014.31",
"20120624_105819.68",
"20120624_110756.70",
"20120624_110934.39",
"20120625_145647.91",
"20120625_145704.48",
"20120625_145733.10",
"20120706_074856.48",
"20120706_111838.98",
"20120706_112004.97",
"20120706_112221.74",
"20120706_154145.36",
"20120706_174935.00",
"20120706_222526.92",
"20120707_024456.53",
"20120707_182712.24",
"20120707_233920.22",
"20120707_234043.78"]
multiplet2=["20120717_001200.59",
"20120717_062750.51",
"20120717_085617.24",
"20120717_090206.13",
"20120719_233504.76",
"20120720_071900.76",
"20120720_135427.53",
"20120720_155202.74",
"20120720_160128.49",
"20120720_161516.86",
"20120720_161805.38",
"20120720_162033.27",
"20120720_162316.01",
"20120720_205553.52",
"20120720_224616.56",
"20120720_225051.60",
"20120720_225415.06",
"20120723_024610.97",
"20120723_030249.99",
"20120723_042004.52",
"20120726_050847.66",
"20120726_054652.09",
"20120728_080729.19"]
multiplet3=["20120619_075137.67",
"20120619_105726.47",
"20120622_150004.65",
"20120623_204452.04",
"20120623_205144.06",
"20120706_104201.36",
"20120706_104804.33",
"20120707_163319.35",
"20120708_043744.85"]
multiplet4=["20120707_190437.66",
"20120707_190850.80",
"20120707_191133.92",
"20120707_191948.84",
"20120707_193539.24",
"20120707_193948.72",
"20120708_191025.56"]
multiplet5=["20120716_182108.48",
"20120717_062828.21",
"20120717_090553.93",
"20120720_161600.44",
"20120727_110406.80",
"20120728_075242.95"]
multiplet6=["20120607_001116.99",
"20120607_001732.03",
"20120613_223617.92",
"20120619_092028.18",
"20120706_174935.00",
"20120719_083827.32"]
multiplet7=["20120618_092539.81",
"20120619_124550.66",
"20120622_103330.10",
"20120622_110447.19"]
multiplet8=["20120727_093231.77",
"20120728_063210.12",
"20120728_220530.88"]
multiplet9=["20120619_091337.17",
"20120619_091843.25"]
multiplet10=["20120726_140110.08",
"20120728_091809.74"]
nonmultiplet=["20120603_174213.7",
"20120607_001625.42",
"20120613_225124.82",
"20120623_105752.73",
"20120707_090221.56",
"20120720_140218.69",
"20120727_094956.83"]



#1xmean
#multiplet1 = ['20120616_232228.44',
#'20120617_125622.65',
#'20120617_211506.44',
#'20120618_072456.48',
#'20120618_072530.94',
#'20120618_141123.68',
#'20120618_141211.79',
#'20120618_141634.53',
#'20120619_051749.47',
#'20120619_091806.39',
#'20120619_105726.47',
#'20120619_121536.76',
#'20120619_123326.84',
#'20120619_123423.28',
#'20120619_123510.98',
#'20120620_060208.19',
#'20120621_014144.60',
#'20120621_064345.06',
#'20120621_092443.43',
#'20120621_092523.79',
#'20120621_092606.59',
#'20120621_092706.50',
#'20120621_150149.55',
#'20120622_050507.42',
#'20120622_050540.80',
#'20120623_093014.31',
#'20120624_105819.68',
#'20120624_110756.70',
#'20120624_110934.39',
#'20120625_145647.91',
#'20120625_145704.48',
#'20120625_145733.10',
#'20120706_074856.48',
#'20120706_111838.98',
#'20120706_112004.97',
#'20120706_112221.74',
#'20120706_154145.36',
#'20120706_222526.92',
#'20120707_024456.53',
#'20120707_182712.24',
#'20120707_233920.22',
#'20120707_234043.78',
#'20120708_043744.85']
#multiplet2 = ['20120716_182108.48',
#'20120717_001200.59',
#'20120717_062750.51',
#'20120717_085617.24',
#'20120717_090206.13',
#'20120717_090553.93',
#'20120719_233504.76',
#'20120720_071900.76',
#'20120720_135427.53',
#'20120720_155202.74',
#'20120720_160128.49',
#'20120720_161516.86',
#'20120720_161600.44',
#'20120720_161805.38',
#'20120720_162033.27',
#'20120720_162316.01',
#'20120720_205553.52',
#'20120720_224616.56',
#'20120720_225051.60',
#'20120720_225415.06',
#'20120723_024610.97',
#'20120723_030249.99',
#'20120723_042004.52',
#'20120726_050847.66',
#'20120726_054652.09',
#'20120727_093231.77',
#'20120728_075242.95',
#'20120728_080729.19']
#multiplet3 = [ '20120619_075137.67 ',
#'20120622_150004.65',
#'20120623_204452.04',
#'20120623_205144.06',
#'20120706_104201.36',
#'20120706_104804.33',
#'20120707_163319.35']
#multiplet4 =['20120707_190437.66 ',
#'20120707_190850.80',
#'20120707_191133.92',
#'20120707_191948.84',
#'20120707_193539.24',
#'20120707_193948.72',
#'20120708_191025.56']
#multiplet5 = ['20120607_001116.99 ',
#'20120607_001732.03',
#'20120613_223617.92',
#'20120619_091337.17',
#'20120619_091843.25',
#'20120619_092028.18',
#'20120706_174935.00',
#'20120719_083827.32',
#'20120728_091809.74']
#multiplet6=['20120618_092539.81',
#'20120619_124550.66',
#'20120622_103330.10',
#'20120622_110447.19']
#multiplet7=['20120717_062828.21',
#'20120726_140110.08',
#'20120727_110406.80']
#multiplet8=['20120728_063210.12',
#'20120728_220530.88']
#nonmultiplet =['20120603_174213.7',
#'20120607_001625.42',
#'20120613_225124.82',
#'20120623_105752.73',
#'20120707_090221.56',
#'20120720_140218.69',
#'20120727_094956.83'];
#colors = ['b','g','r','c','m','y','k','lime','salmon','saddlebrown','indigo'] #http://matplotlib.org/examples/color/named_colors.html
colors = ['#a6cee3',
'#1f78b4',
'#b2df8a',
'#33a02c',
'#fb9a99',
'#e31a1c',
'#fdbf6f',
'#ff7f00',
'#cab2d6',
'#6a3d9a',
'y']
#['#8dd3c7',
#'#ffffb3',
#'#bebada',
#'#fb8072',
#'#80b1d3',
#'#fdb462',
#'#b3de69',
#'#fccde5',
#'#d9d9d9',
#'#bc80bd',
#'#ccebc5']
plotTimesList = []
colorsList = []
mulLabelList = []
for template in templateList:
    if template in multiplet1:
        c = colors[0]
        mulLabel = 'Multiplet 1'
    elif template in multiplet2:
        c = colors[1]
        mulLabel = 'Multiplet 2'
    elif template in multiplet3:
        mulLabel = 'Multiplet 3'
        c = colors[2]
    elif template in multiplet4:
        mulLabel = 'Multiplet 4'
        c = colors[3]
    elif template in multiplet5:
        mulLabel = 'Multiplet 5'
        c = colors[4]
    elif template in multiplet6:
        mulLabel = 'Multiplet 6'
        c = colors[5]
    elif template in multiplet7:
        mulLabel = 'Multiplet 7'
        c = colors[6]
    elif template in multiplet8:
        mulLabel = 'Multiplet 8'
        c = colors[7]
    elif template in multiplet9:
        mulLabel = 'Multiplet 9'
        c = colors[8]
    elif template in multiplet10:
        mulLabel = 'Multiplet 10'
        c = colors[9]
    elif template in nonmultiplet:
        mulLabel = 'Non-multiplet'
        c = colors[10]
    colorsList.append(c)
    mulLabelList.append(mulLabel)
    plotTimes1 = []
    for date in dateList:
        datestr = str(date)
        print 'Processing %s' %datestr
        detectDir = '/Volumes/GeoPhysics_07/users-data/matsonga/MRP_PROJ/data/mastersData/lagfilesLongTemplateWithPicks20151125/%sxMAD' %threshMult
        timeDir1 = '%s/event%s/%s/lagTimes/' %(detectDir,template,datestr)
        if os.path.isdir(timeDir1):
                for file in os.listdir(timeDir1):
                    file1 = file
                    print file1
                    detectData1 = file1.split('-')
                    year1 = detectData1[0]
                    month1 = detectData1[1]
                    day1 = detectData1[2]
                    time1 = str(datetime.timedelta(seconds=float(detectData1[3])))
                    suffix1 = detectData1[4]
                    print time1
                    if '.' in time1:
                        datetime1=datetime.datetime.strptime('%s/%s/%s %s'%(year1,month1,day1,time1),'%Y/%m/%d %H:%M:%S.%f')
                    else:
                        datetime1=datetime.datetime.strptime('%s/%s/%s %s'%(year1,month1,day1,time1),'%Y/%m/%d %H:%M:%S')
                    plotTimes1.append(datetime1)

    plotTimesList.append(plotTimes1)



fig = plt.figure()
ax = fig.add_subplot(111)
plotValue = 0
exList = [6,74,22,61,73,1,14,103,25,109,0]
# print templateList[exList]
for i in range(1,len(plotTimesList)+1):
    plotTimes = plotTimesList[i-1]
    y = np.empty(len(plotTimes))
    y.fill(i)
    plotTimes.sort()
    ax.plot(plotTimes,y,'--o',color='gray',linewidth=0.3,markerfacecolor = colorsList[i-1],markersize = 5,markeredgewidth = 0,markeredgecolor= 'k',label= mulLabelList[i-1])
#     if (i-1) in exList: #for one plot per multiplet
#         print templateList[i-1]
# #        plotValue = plotValue + 10
#         plotValue = exList.index(i-1) +1
#         y = np.empty(len(plotTimes))
#         y.fill(plotValue)
#         ax.plot(plotTimes,y,'--o',color='gray',markerfacecolor = colorsList[i-1],markersize = 10,markeredgewidth = 0.2,markeredgecolor= 'k',label= mulLabelList[i-1])

handles,labels = ax.get_legend_handles_labels()
labels2 = set(labels)
categories = ['Multiplet 1',
'Multiplet 2',
'Multiplet 3',
'Multiplet 4',
'Multiplet 5',
'Multiplet 6',
'Multiplet 7',
'Multiplet 8',
'Multiplet 9',
'Multiplet 10',
'Non-multiplet']
handles2=[handles[6],
handles[74],
handles[22],
handles[61],
handles[73],
handles[1],
handles[14],
handles[103],
handles[23],
handles[99],
handles[0]]
# handles2=[handles[2],
# handles[8],
# handles[4],
# handles[6],
# handles[7],
# handles[1],
# handles[3],
# handles[9],
# handles[5],
# handles[10],
# handles[0]]
ax.legend(handles2,categories,loc='upper center',bbox_to_anchor=(0.5,-0.11),ncol=3)#1.1
fig.autofmt_xdate()
ax.grid()
ax.set_xlabel("Date")
ax.set_ylabel(r"Template Index")
plt.show()




#
#
#
# #os.chdir()
# plotTimes1 = []
# commonDetectors1 = []
# for folder in os.listdir('./commonDetectionsClustered7/%s' %cluster1):
#     for file in os.listdir('./commonDetectionsClustered7/%s/%s'%(cluster1,folder)):
#         file1 = file
#         fileObject1 = open('commonDetectionsClustered7/%s/%s/%s' %(cluster1,folder,file1),'r')
#         event1 = []
#         time1 = []
#         for line in fileObject1:
#             data1 = line.split(',')
#             event1.append(data1[0])
#             time1.append(data1[1])
#         commonDet1 = len(event1)
#         commonDetectors1.append(commonDet1)
#         fileData1 = file1.split('-')
#         time1 = str(datetime.timedelta(seconds=float(fileData1[3])))
#         if '.' in time1:
#             datetime1=datetime.datetime.strptime('%s/%s/%s %s'%(fileData1[0],fileData1[1],fileData1[2],time1),'%Y/%m/%d %H:%M:%S.%f')
#         else:
#             datetime1=datetime.datetime.strptime('%s/%s/%s %s'%(fileData1[0],fileData1[1],fileData1[2],time1),'%Y/%m/%d %H:%M:%S')
#         plotTimes1.append(datetime1)
#         fileObject1.close()
# fig = plt.figure()
# ax=fig.add_subplot(111)
# values1 = range(1,len(plotTimes1)+1)
# zip(*sorted(zip(plotTimes1,commonDetectors1)))
# plotTimes1.sort()
# markers_on1 = ['20120619_092028.18', #C
# #'20120613_225124.82',
# #'20120619_091337.17',
# '20120607_001116.99',
# #'20120607_001732.03',
# #'20120613_223617.92',
# #'20120619_091843.25',
# #'20120607_001625.42',
# '20120719_083827.32']
# markers_on2 = []
# for marker in markers_on1:
#     marker2 = datetime.datetime.strptime(marker,'%Y%m%d_%H%M%S.%f')
#     markers_on2.append(marker2)
#
# ax.plot(plotTimes1,values1,label = 'Threshold = 7.1xMAD' )
# #plt.axvline(*markers_on2,color='k',linestyle='--')
# #ax.plot(plotTimes1[markers_on2],values1,label = 'Threshold = 7.1xMAD' )
# #plt.scatter(plotTimes,values,s=20,c=commonDetectors,linewidths=None)
#
#
#
#
# ###########################################
# fig.autofmt_xdate()
# plotTimes2 = []
# commonDetectors2 = []
# for folder in os.listdir('./commonDetectionsClustered8/%s' %cluster1):
#     for file in os.listdir('./commonDetectionsClustered8/%s/%s'%(cluster1,folder)):
#         file2 = file
#         fileObject2 = open('commonDetectionsClustered8/%s/%s/%s' %(cluster1,folder,file2),'r')
#         event2 = []
#         time2 = []
#         for line in fileObject2:
#             data2 = line.split(',')
#             event2.append(data2[0])
#             time2.append(data2[1])
#         commonDet2 = len(event2)
#         commonDetectors2.append(commonDet2)
#         fileData2 = file2.split('-')
#         time2 = str(datetime.timedelta(seconds=float(fileData2[3])))
#         if '.' in time2:
#             datetime2=datetime.datetime.strptime('%s/%s/%s %s'%(fileData2[0],fileData2[1],fileData2[2],time2),'%Y/%m/%d %H:%M:%S.%f')
#         else:
#             datetime2=datetime.datetime.strptime('%s/%s/%s %s'%(fileData2[0],fileData2[1],fileData2[2],time2),'%Y/%m/%d %H:%M:%S')
#         plotTimes2.append(datetime2)
#         fileObject2.close()
#
# values2 = range(1,len(plotTimes2)+1)
# zip(*sorted(zip(plotTimes2,commonDetectors2)))
# plotTimes2.sort()
# ax.plot(plotTimes2,values2,label='Threshold = 8xMAD' )
#
# plotTimes3 = []
# commonDetectors3 = []
# for folder in os.listdir('./commonDetectionsClustered10/%s' %cluster1):
#     for file in os.listdir('./commonDetectionsClustered10/%s/%s'%(cluster1,folder)):
#         file3 = file
#         fileObject3 = open('commonDetectionsClustered10/%s/%s/%s' %(cluster1,folder,file3),'r')
#         event3 = []
#         time3 = []
#         for line in fileObject3:
#             data3 = line.split(',')
#             event3.append(data3[0])
#             time3.append(data3[1])
#         commonDet3 = len(event3)
#         commonDetectors3.append(commonDet3)
#         fileData3 = file3.split('-')
#         time3 = str(datetime.timedelta(seconds=float(fileData3[3])))
#         if '.' in time3:
#             datetime3=datetime.datetime.strptime('%s/%s/%s %s'%(fileData3[0],fileData3[1],fileData3[2],time3),'%Y/%m/%d %H:%M:%S.%f')
#         else:
#             datetime3=datetime.datetime.strptime('%s/%s/%s %s'%(fileData3[0],fileData3[1],fileData3[2],time3),'%Y/%m/%d %H:%M:%S')
#         plotTimes3.append(datetime3)
#         fileObject3.close()
#
# values3 = range(1,len(plotTimes3)+1)
# zip(*sorted(zip(plotTimes3,commonDetectors3)))
# plotTimes3.sort()
# ax.plot(plotTimes3,values3,label='Threshold = 10xMAD')
#
#
# file4 = '/Volumes/GeoPhysics_07/users-data/matsonga/MRP_PROJ/data/mastersData/productionData/NM08_WHP_3.csv'
# fileObject4 = open(file4,'r')
# plotTimes4 = []
# whp = []
# flow = []
# cumInj = []
# injectivity = []
# for line in fileObject4:
#     data4 = line.split(',')
#     time4 = data4[0]
#     datetime4 = datetime.datetime.strptime('%s' %time4,'%m/%d/%Y %H:%M')
#     plotTimes4.append(datetime4)
#     whp.append(float(data4[1]))
#     flow.append(float(data4[2]))
#     cumInj.append(float(data4[3]))
#     #injectivity.append(float(data4[4]))
# ax2=ax.twinx()
# ax2.plot(plotTimes4,cumInj,'m',label='Cumulative Injected Volume (tonnes)')
# ax.legend(loc=0)
# #ax2.legend(loc=0)
# ax.grid
# ax.set_xlabel("Date")
# ax.set_ylabel(r"Number of Detections")
# ax2.set_ylabel(r"Injected Fluid(tonnes)")
# #labels = ax.get_xticklabels()
# #for label in labels:
# #    label.set_rotation(30)
#
# #fig.autofmt_xdate()
# #ax.xticks(rotation=70)
# plt.show()

#dates = matplotlib.dates.date2num(plotTimes)
