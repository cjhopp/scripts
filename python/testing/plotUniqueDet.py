# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:35:52 2015

@author: matsonga
"""

import os
import matplotlib.pyplot as plt
import datetime
#from obspy.core import UTCDateTime
#import matplotlib
#from datetime import strptime


startdate = datetime.date(year=2012,month=06,day=01)
numdays =61
dateList = []
for x in range(0,numdays):
    dateList.append(startdate +datetime.timedelta(days=x))
cluster1 = 'Multiplet1'
cluster2 = 'Multiplet3'
cluster3 = 'Multiplet4'
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
if cluster1 == 'Multiplet1' or cluster1 =='A':
    c = colors[0]
    mulLabel = 'Multiplet 1'
elif cluster1 == 'Multiplet2':
    c = colors[1]
    mulLabel = 'Multiplet 2'
elif cluster1 == 'Multiplet3':
    mulLabel = 'Multiplet 3'
    c = colors[2]
elif cluster1 == 'Multiplet4':
    mulLabel = 'Multiplet 4'
    c = colors[3]
elif cluster1 == 'Multiplet5':
    mulLabel = 'Multiplet 5'
    c = colors[4]
elif cluster1 == 'Multiplet6':
    mulLabel = 'Multiplet 6'
    c = colors[5]
elif cluster1 == 'Multiplet7':
    mulLabel = 'Multiplet 7'
    c = colors[6]
elif cluster1 == 'Multiplet8':
    mulLabel = 'Multiplet 8'
    c = colors[7]
elif cluster1 == 'Multiplet9':
    mulLabel = 'Multiplet 9'
    c = colors[8]
elif cluster1 == 'Multiplet10':
    mulLabel = 'Multiplet 10'
    c = colors[9]
elif cluster1 == 'NonMultiplet':
    mulLabel = 'Non-multiplet'
    c = colors[10]
#os.chdir()
# plotTimes1 = []
# commonDetectors1 = []
# for folder in os.listdir('./commonDetectionsClustered7.1/%s' %cluster1):
#     for file in os.listdir('./commonDetectionsClustered7.1/%s/%s'%(cluster1,folder)):
#         file1 = file
#         fileObject1 = open('commonDetectionsClustered7.1/%s/%s/%s' %(cluster1,folder,file1),'r')
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
# #markers_on1 = ['20120619_092028.18', #C
# #'20120613_225124.82',
# #'20120619_091337.17',
# #'20120607_001116.99',
# #'20120607_001732.03',
# #'20120613_223617.92',
# #'20120619_091843.25',
# #'20120607_001625.42',
# #'20120719_083827.32']
# #markers_on2 = []
# #for marker in markers_on1:
# #    marker2 = datetime.datetime.strptime(marker,'%Y%m%d_%H%M%S.%f')
# #    markers_on2.append(marker2)
#
# #ax.plot(plotTimes1,values1,lw = 2.0,label = 'Threshold = 7.1xMAD' )
# #plt.axvline(*markers_on2,color='k',linestyle='--')
# #ax.plot(plotTimes1[markers_on2],values1,label = 'Threshold = 7.1xMAD' )
# #plt.scatter(plotTimes,values,s=20,c=commonDetectors,linewidths=None)

plotTimes5 = []
commonDetectors5 = []
dayVector5=[]
eqRate5 = []
threshMult5 = '7.6'
for folder in os.listdir('./commonDetectionsClustered%s/%s' %(threshMult5,cluster1)):
    eqRate5.append(len(os.listdir('./commonDetectionsClustered%s/%s/%s' %(threshMult5,cluster1,folder))))
    #print len(os.listdir('./commonDetectionsClustered%s/%s/%s' %(threshMult5,cluster1,folder)))
    dayVector5.append(datetime.datetime.strptime('%s' %folder,'%Y-%m-%d'))
    for file in os.listdir('./commonDetectionsClustered%s/%s/%s'%(threshMult5,cluster1,folder)):
        file5 = file
        fileObject5 = open('commonDetectionsClustered%s/%s/%s/%s' %(threshMult5,cluster1,folder,file5),'r')
        event5 = []
        time5 = []
        for line in fileObject5:
            data5 = line.split(',')
            event5.append(data5[0])
            time5.append(data5[1])
        commonDet5 = len(event5)
        commonDetectors5.append(commonDet5)
        fileData5 = file5.split('-')
        time5 = str(datetime.timedelta(seconds=float(fileData5[3])))
        if '.' in time5:
            datetime5=datetime.datetime.strptime('%s/%s/%s %s'%(fileData5[0],fileData5[1],fileData5[2],time5),'%Y/%m/%d %H:%M:%S.%f')
        else:
            datetime5=datetime.datetime.strptime('%s/%s/%s %s'%(fileData5[0],fileData5[1],fileData5[2],time5),'%Y/%m/%d %H:%M:%S')
        plotTimes5.append(datetime5)
        fileObject5.close()


values5 = range(1,len(plotTimes5)+1)
zip(*sorted(zip(plotTimes5,commonDetectors5)))
plotTimes5.sort()
fig = plt.figure()
ax=fig.add_subplot(111)
ax.plot(plotTimes5,values5,lw = 3.0,color =c,label='Cumulative detections') # '#a6cee3' '#1f78b4'
ax.bar(dayVector5,eqRate5,label='Detections per day',color='k')
#ax.set_zorder(2)

ax.grid()
plotTimes6 = []
threshMult6 = '8'
commonDetectors6 = []
for folder in os.listdir('./commonDetectionsClustered%s/%s' %(threshMult6,cluster2)):
    for file in os.listdir('./commonDetectionsClustered%s/%s/%s'%(threshMult6,cluster2,folder)):
        file6 = file
        fileObject6 = open('commonDetectionsClustered%s/%s/%s/%s' %(threshMult6,cluster2,folder,file6),'r')
        event6 = []
        time6 = []
        for line in fileObject6:
            data6 = line.split(',')
            event6.append(data6[0])
            time6.append(data6[1])
        commonDet6 = len(event6)
        commonDetectors6.append(commonDet6)
        fileData6 = file6.split('-')
        time6 = str(datetime.timedelta(seconds=float(fileData6[3])))
        if '.' in time6:
            datetime6=datetime.datetime.strptime('%s/%s/%s %s'%(fileData6[0],fileData6[1],fileData6[2],time6),'%Y/%m/%d %H:%M:%S.%f')
        else:
            datetime6=datetime.datetime.strptime('%s/%s/%s %s'%(fileData6[0],fileData6[1],fileData6[2],time6),'%Y/%m/%d %H:%M:%S')
        plotTimes6.append(datetime6)
        fileObject6.close()

values6 = range(1,len(plotTimes6)+1)
zip(*sorted(zip(plotTimes6,commonDetectors6)))
plotTimes6.sort()
#ax.plot(plotTimes6,values6,lw = 3.0,color = '#b2df8a',label='Multiplet 3')#'#1f78b4' Multiplet2

fig.autofmt_xdate()



plotTimes2 = []
commonDetectors2 = []
for folder in os.listdir('./commonDetectionsClustered8/%s' %cluster3):
    for file in os.listdir('./commonDetectionsClustered8/%s/%s'%(cluster3,folder)):
        file2 = file
        fileObject2 = open('commonDetectionsClustered8/%s/%s/%s' %(cluster3,folder,file2),'r')
        event2 = []
        time2 = []
        for line in fileObject2:
            data2 = line.split(',')
            event2.append(data2[0])
            time2.append(data2[1])
        commonDet2 = len(event2)
        commonDetectors2.append(commonDet2)
        fileData2 = file2.split('-')
        time2 = str(datetime.timedelta(seconds=float(fileData2[3])))
        if '.' in time2:
            datetime2=datetime.datetime.strptime('%s/%s/%s %s'%(fileData2[0],fileData2[1],fileData2[2],time2),'%Y/%m/%d %H:%M:%S.%f')
        else:
            datetime2=datetime.datetime.strptime('%s/%s/%s %s'%(fileData2[0],fileData2[1],fileData2[2],time2),'%Y/%m/%d %H:%M:%S')
        plotTimes2.append(datetime2)
        fileObject2.close()

values2 = range(1,len(plotTimes2)+1)
zip(*sorted(zip(plotTimes2,commonDetectors2)))
plotTimes2.sort()
#ax.plot(plotTimes2,values2,lw = 3.0,c= '#33a02c',label='Multiplet 4')

plotTimes3 = []
commonDetectors3 = []
for folder in os.listdir('./commonDetectionsClustered10/%s' %cluster1):
    for file in os.listdir('./commonDetectionsClustered10/%s/%s'%(cluster1,folder)):
        file3 = file
        fileObject3 = open('commonDetectionsClustered10/%s/%s/%s' %(cluster1,folder,file3),'r')
        event3 = []
        time3 = []
        for line in fileObject3:
            data3 = line.split(',')
            event3.append(data3[0])
            time3.append(data3[1])
        commonDet3 = len(event3)
        commonDetectors3.append(commonDet3)
        fileData3 = file3.split('-')
        time3 = str(datetime.timedelta(seconds=float(fileData3[3])))
        if '.' in time3:
            datetime3=datetime.datetime.strptime('%s/%s/%s %s'%(fileData3[0],fileData3[1],fileData3[2],time3),'%Y/%m/%d %H:%M:%S.%f')
        else:
            datetime3=datetime.datetime.strptime('%s/%s/%s %s'%(fileData3[0],fileData3[1],fileData3[2],time3),'%Y/%m/%d %H:%M:%S')
        plotTimes3.append(datetime3)
        fileObject3.close()

values3 = range(1,len(plotTimes3)+1)
zip(*sorted(zip(plotTimes3,commonDetectors3)))
plotTimes3.sort()
#ax.plot(plotTimes3,values3,lw = 2.0,label='Threshold = 10xMAD')

handles1,labels1 = ax.get_legend_handles_labels()


file4 = '/Volumes/GeoPhysics_07/users-data/matsonga/MRP_PROJ/data/mastersData/productionData/NM08_WHP_3.csv'
fileObject4 = open(file4,'r')
plotTimes4 = []
whp = []
flow = []
cumInj = []
injectivity = []
for line in fileObject4:
    data4 = line.split(',')
    time4 = data4[0]
    datetime4 = datetime.datetime.strptime('%s' %time4,'%m/%d/%Y %H:%M')
    datetime4 = datetime4 - datetime.timedelta(hours=12)
    plotTimes4.append(datetime4)
    whp.append(float(data4[1]))
    flow.append(float(data4[2]))
    cumInj.append(float(data4[3]))
    if data4[4] == '#DIV/0!':
        injectivity.append(0)
    else:
        injectivity.append(float(data4[4]))
plotTimes4.append(datetime.datetime.strptime('07/31/2012 23:59','%m/%d/%Y %H:%M'))
cumInj.append(float(data4[3]))
fileObject4.close()
ax2=ax.twinx()
ax2.plot(plotTimes4,cumInj,'m',label='Cumulative Injected Volume',linewidth = 3.0,alpha = 0.6)
#ax2.set_zorder(1)
#ax.legend(loc=0)
#ax2.legend(loc=0)
handles2,labels2 = ax2.get_legend_handles_labels()
handles = handles1+handles2
labels = labels1+labels2
ax.legend(handles,labels,loc=0,fontsize=14)
#ax.grid()
ax.set_xlabel("Date",fontsize = 14)
ax.set_ylabel(r"Number of Detections",fontsize = 14)
ax2.set_ylabel(r"Injected Fluid(tonnes)",color = 'm', fontsize = 14)
ax.tick_params(axis = 'both',which='major'  ,labelsize = 14)
ax2.tick_params(axis = 'both',which='major'  ,labelsize = 14)
plt.title('%s Detections' %mulLabel,fontsize=14)
#labels = ax.get_xticklabels()
#for label in labels:
#    label.set_rotation(30)

#fig.autofmt_xdate()
#ax.xticks(rotation=70)
plt.savefig('commonDetectionsFigs/forThesis/Thresh%s%s_CummDet_PerDay_CummInj.pdf' %(threshMult5,cluster1))
#plt.show()

#dates = matplotlib.dates.date2num(plotTimes)
