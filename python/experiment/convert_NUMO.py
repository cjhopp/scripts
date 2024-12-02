import pandas as pd
import struct
import numpy as np
import pandas.tseries.offsets as offsets


binaryFile = open("a.csv", mode='rb')
col=6
num=484800
Time_diff=6
dti=[]

data=[]

for j in range(col):
    binaryFile.read(4)
    sec1=binaryFile.read(8)
    sec2=binaryFile.read(8)
    dt=struct.unpack(">d",binaryFile.read(8))
    xxx=struct.unpack(">f",binaryFile.read(4))
    for i in range(num):
        data.append(struct.unpack(">d", binaryFile.read(8)))
    xxx=(binaryFile.read(25))

 
d2=pd.DataFrame({"Zt" : data[0:num]})

d2["Zt"]=np.array(data[0:num])
d2["Yt"]=np.array(data[num:num+num])
d2["Xt"]=np.array(data[num+num:num+num+num])
d2["Zb"]=np.array(data[num+num+num:num+num+num+num])
d2["Yb"]=np.array(data[num+num+num+num:num+num+num+num+num])
d2["Xb"]=np.array(data[num+num+num+num+num:num+num+num+num+num+num])


cs1=int.from_bytes(sec1, "big")
cs2=int.from_bytes(sec2, "big")
date=pd.to_datetime(cs1*1000000000-24107*24*60*60*1000000000-Time_diff*60*60*1000000000)
date=date+offsets.Nano(int(1000000000*2**-64*cs2))

ds=int(dt[0]*1000000000)
for i in range(num):
    dti.append(date) 
    date=date+offsets.Nano(ds)
    
d2["time"]=dti
d2=d2.set_index("time")
d2.to_csv("out.csv")
