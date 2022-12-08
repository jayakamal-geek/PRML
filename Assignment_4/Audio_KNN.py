# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:57:07 2022

@author: jayak
"""

##importing the required libraries
import os
import numpy as np

#reading data from files
def readFiles(filePath):
    fp = open(filePath, "r")
    nc, nf = [int(x) for x in fp.readline().strip().split(" ")]
    data = []
    for i in range(nf):
        data.append([float(x) for x in fp.readline().strip().split(" ")])
    fp.close()
    return nf, np.array(data)

#filtering and locating required files among pool of files
def getDataProc(phase):
    fileNums = [1, 2, 4, 5, 9]
    dataBase = {}
    for i in fileNums:
        minNF = np.inf
        nfVals = []
        fData = []
        fileLoc = "soundTracks/"+str(i)+"/"+phase
        for fileName in os.listdir(fileLoc):
            if fileName.endswith(".mfcc"):
                nf, data =  readFiles(fileLoc + "/" + fileName)
                minNF = min(minNF, nf)
                nfVals.append(nf)
                fData.append(data)

        dataCount = len(nfVals)
        for j in range(dataCount):
            windowSZ = nfVals[j]-minNF+1
            if (windowSZ != 1):
                arr = []
                k = 0
                while(k + windowSZ <= nfVals[j]):
                    temp = np.array(fData[j][k])
                    for z in range(k+1, k+windowSZ):
                        temp = temp + np.array(fData[j][z])
                    temp = (windowSZ**(-1))*temp
                    arr.append(temp)
                    k+=1
                fData[j] = arr
        dataBase[i] = [minNF, fData]
    return dataBase

def getData(phase):
    fileNums = [1, 2, 4, 5, 9]
    dataBase = {}
    for i in fileNums:
        nfVals = []
        fData = []
        fileLoc = "soundTracks/"+str(i)+"/"+phase
        for fileName in os.listdir(fileLoc):
            if fileName.endswith(".mfcc"):
                nf, data =  readFiles(fileLoc + "/" + fileName)
                nfVals.append(nf)
                fData.append(data)
        dataBase[i] = [nfVals, fData]
    return dataBase

def KNN(elem, trainData, trainData_1):
    inputArr = [1, 2, 4, 5, 9]
    arr = []
    elemLen = len(elem)
    count = {}
    for i in inputArr:
        count[i] = 0
        classCF = trainData[i][0]
        if(elemLen >= classCF):
            windowSZ = elemLen - classCF + 1
            if(windowSZ != 1):
                tempArr = []
                k = 0
                while(k+windowSZ <= elemLen):
                    temp = np.array(elem[k])
                    for z in range(k+1, k+windowSZ):
                        temp = temp+np.array(elem[z])
                    temp = (windowSZ**(-1))*temp
                    tempArr.append(temp)
                    k+=1
            for h in trainData[i][1]:
                arr.append([np.linalg.norm(np.array(h) - np.array(tempArr))/classCF, i])
                
        else:
            dataLen = len(trainData_1[i][0])
            for j in range(dataLen):
                windowSZ = trainData_1[i][0][j] - elemLen + 1
                tempArr = []
                k = 0
                while(k+windowSZ <= trainData_1[i][0][j]):
                    temp = np.array(trainData_1[i][1][j][k])
                    for z in range(k+1, k+windowSZ):
                        temp = temp+np.array(trainData_1[i][1][j][z])
                    temp = (windowSZ**(-1))*temp
                    tempArr.append(temp)
                    k+=1
                arr.append([np.linalg.norm(np.array(tempArr) - np.array(elem))/elemLen, i])
                
    arr.sort();
    for i in range(10):
        count[arr[i][1]]+=1
    return max(count, key=count.get)

trainData = getDataProc("train")
trainData_1 = getData("train")
devData = getData("dev")

inputArr = [1, 2, 4, 5, 9]
count = 0
maxCount = 0
for i in inputArr:
    for elem in devData[i][1]:
        index = KNN(elem, trainData, trainData_1)
        maxCount+=1
        if(index == i):
            count+=1
print(f"Accuracy : {(count*100)/maxCount} %")