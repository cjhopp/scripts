import datetime
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt


database = 'NM76D'
trial = '8'
indir = '/Users/home/matsonga/seisan/REA/%s/hypoDD%s' %(database,trial)
#filename = '%s/hypoDD.reloc' %indir
filename = '%s/hypoDDJulDay.reloc' %indir
fid = open(filename, 'r')
contents = fid.readlines()
fid.close()
fig = plt.figure()
jdays = range(153,214)
for i,wjday in enumerate(jdays):
    plotLat = []
    plotLon = []
    for i,line in enumerate(contents):
        ID = contents.split()[0]
        lat = contents.split()[1]
        lon = contents.split()[2]
        depth = contents.split()[3]
        year = contents.split()[4]
        jday = int(contents.split()[5])
        if jday == wjday:
            plotLat.append(float(lat))
            plotLon.append(float(lon))

    ax=fig.add_subplot()






fig = plt.figure()
ax=fig.add_subplot(111)
ax.plot(plotTimes,values,lw = 3.0,color = '#a6cee3',label='Cumulative detections') # '#a6cee3' '#1f78b4'
ax.bar(dateList,eqRate,label='Detections per day',color='k')
fig.autofmt_xdate()
handles1,labels1 = ax.get_legend_handles_labels()
