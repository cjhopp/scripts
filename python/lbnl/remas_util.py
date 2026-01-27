#!/usr/bin/python

import os
import gzip
import pickle
import datetime
import calendar
import zlib
from io import StringIO


def loadEventData(dir, filename, deploymentname=None, datastring=None):
    """ Unpickles and returns the EventData object stored in
        the specifed directory and filename.
    """
    if datastring != None:
        #the data has been passed in as a string
        fdin = StringIO(datastring)
        gzip_wrapper = gzip.GzipFile(fileobj = fdin, mode='rb')        
        dataIn = pickle.load(gzip_wrapper)
        gzip_wrapper.close()
        fdin.close()
    else:
        #the data is in a file
        #try to locate the file
        if not os.path.exists(os.path.join(dir,filename)):
           #the event ID and file name may not match
           print("Data file not found:", os.path.join(filename), end=' ')
           thisot = calendar.timegm(datetime.strptime(filename.split('.')[0], "%y%m%d%H%M%S").utctimetuple())
           thisot -= 2.0
           foundit = False
           for i in range (4): #search 2 seconds around origin time
               thisot = thisot + 1.0
               testname = datetime.utcfromtimestamp(thisot).strftime("%y%m%d%H%M%S") + ".evt"
               if os.path.exists(os.path.join(dir,testname)):
                   filename = testname
                   foundit = True
                   print("Found it:", testname)
                   break
           if not foundit:
               print("Unable to find input file:", os.path.join(dir, filename))
               return None
        try:
            fdin=gzip.open(os.path.join(dir,filename),'rb')
            dataIn=pickle.load(fdin)
        except Exception as e:
            fdin.close()
            fdin=gzip.open(os.path.join(dir,filename),'rb')
            #print ("Try old pickle format (latin1)")
            try:
                dataIn = pickle.load(fdin,encoding="latin1")
            except (EOFError, pickle.UnpicklingError, zlib.error):
                print("File load error")
                fdin.close()
                return None
        fdin.close()
    
    #construct an EventData object and load in the event-level data
    eventtime = dataIn['timestamp']
    newevent = EventData(eventtime) #create the event
    try:
        newevent.location = dataIn['location']
    except KeyError:
        newevent.location = None
    try:
        newevent.eventid = dataIn['eventid']
    except KeyError:
        newevent.eventid = None
        
    if deploymentname != None:
        newevent.deploymentname = deploymentname
    else:
        try:
            newevent.deploymentname = dataIn['deploymentname']    
        except AttributeError:
            newevent.deploymentname = None
        if newevent.deploymentname == None:
            if deploymentname:
                newevent.deploymentname = deploymentname #use the one in the function arguments
            #print "no deployment name to use!"

    try:
        newevent.status = dataIn['status']
    except KeyError:
        newevent.status = 0
    try:
        newevent.filepath = dataIn['filepath']
    except KeyError:
        newevent.filepath = None
        
    #print "Deployment name set to", newevent.deploymentname
    if newevent.location != None:
        newevent.origin_time = newevent.location.T
        #use the origin time to set the event ID if it isn't set already
        if not hasattr(newevent,'eventid') or newevent.eventid == None:
            newevent.eventid = datetime.utcfromtimestamp(newevent.location.T).strftime("%y%m%d%H%M%S")
        newevent.timestamp = newevent.location.T
        
    #construct StationEvent objects from the imported data and add them
    for staname in dataIn.keys():
        #make sure we don't iterate on event level keys
        if staname in ('timestamp', 'location', 'origin_time', 'deploymentname', 'eventid', 'status', 'filepath'):
            continue
        #print "Found", staname, dataIn[staname]['stationid']
        #print "station data:", shIn[staname]
        dataIn[staname]['deploymentname'] = newevent.deploymentname
        dataIn[staname]['stationid'] = staname #fixes error--not sure where the error comes from
        newevent.addStationEvent(StationEvent(dataIn[staname]))
    return newevent
