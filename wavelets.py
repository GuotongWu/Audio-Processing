#-*-coding:utf-8-*-

import matplotlib.pyplot as plt
import pywt
import math
import numpy as np

#get Data
ecg=pywt.data.ecg()  #生成心电信号
index=[]
data=[]
coffs=[]

for i in range(len(ecg)-1):
    X=float(i)
    Y=float(ecg[i])
    index.append(X)
    data.append(Y)
#create wavelet object and define parameters
w=pywt.Wavelet('db8')#选用Daubechies8小波
maxlev=pywt.dwt_max_level(len(data),w.dec_len)
print("maximum level is"+str(maxlev))
threshold=0  #Threshold for filtering

#Decompose into wavelet components,to the level selected:
coffs=pywt.wavedec(data,'db8',level=maxlev) #将信号进行小波分解

for i in range(1,len(coffs)):
    coffs[i]=pywt.threshold(coffs[i],threshold*max(coffs[i]))

datarec=pywt.waverec(coffs,'db8')#将信号进行小波重构

mintime=0
maxtime=mintime+len(data) 
print(mintime,maxtime)

plt.figure()
plt.subplot(3,1,1)
plt.plot(index[mintime:maxtime], data[mintime:maxtime])
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("Raw signal")
plt.subplot(3, 1, 2)
plt.plot(index[mintime:maxtime], datarec[mintime:maxtime])
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("De-noised signal using wavelet techniques")
plt.subplot(3, 1, 3)
plt.plot(index[mintime:maxtime],data[mintime:maxtime]-datarec[mintime:maxtime])
plt.xlabel('time (s)')
plt.ylabel('error (uV)')
plt.tight_layout()
plt.show()