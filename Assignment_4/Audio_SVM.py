# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:54:55 2022

@author: jayak
"""

##importing the required libraries
import os
import numpy as np
from sklearn.svm import SVC

minNF = np.inf

#reading data from files
def readFiles(filePath):
    global minNF
    fp = open(filePath, "r")
    nc, nf = [int(x) for x in fp.readline().strip().split(" ")]
    data = []
    minNF = min(minNF, nf)
    for i in range(nf):
        data.append([float(x) for x in fp.readline().strip().split(" ")])
    fp.close()
    return np.array(data)

def getData(phase):
    fileNums = [1, 2, 4, 5, 9]
    dataBase = {}
    for i in fileNums:
        fData = []
        fileLoc = "soundTracks/"+str(i)+"/"+phase
        for fileName in os.listdir(fileLoc):
            if fileName.endswith(".mfcc"):
                data =  readFiles(fileLoc + "/" + fileName)
                fData.append(data)
        dataBase[i] = fData
    return dataBase

def processData(data):
    inputArr = [1, 2, 4, 5, 9]
    for i in inputArr:
        tempLen = len(data[i])
        for j in range(tempLen):
            windowSZ = len(data[i][j]) - minNF + 1
            if(windowSZ != 1):
                arr = np.array([])
                k = 0
                while(k + windowSZ <= len(data[i][j])):
                    temp = np.array(data[i][j][k])
                    for z in range(k+1, k+windowSZ):
                        temp = temp + np.array(data[i][j][z])
                    temp = (windowSZ**(-1))*temp
                    arr = np.append(arr, temp)
                    k+=1
                data[i][j] = arr
            else:
                arr = np.array([])
                for elem in data[i][j]:
                    arr = np.append(arr, elem)
                data[i][j] = arr
    return data

trainData = getData("train")
devData = getData("dev")
trainData = processData(trainData)
devData = processData(devData)

dataArr = []
classArr = []
inputArr = [1, 2, 4, 5, 9]
for i in inputArr:
    for elem in trainData[i]:
        dataArr.append(elem)
        classArr.append(i)
clf = SVC(kernel='linear')
clf.fit(dataArr, classArr)
true = 0
false = 0

for i in inputArr:
    for elem in devData[i]:
        cl = clf.predict(elem.reshape(1, 2546))
        if(cl == i):
            true+=1
        else:
            false+=1
print("Accuracy :", (true*100)/(true+false), "%")