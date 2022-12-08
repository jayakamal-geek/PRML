# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:46:15 2022

@author: jayak
"""

import os
import numpy as np

minNF = np.inf
def readFiles(filePath):
    global minNF
    fp = open(filePath, "r")
    temp = fp.read().strip().split(" ")
    nf = int(temp[0])
    minNF = min(minNF, nf)
    data = []
    
    for i in range(0, 2*nf, 2):
        tempX = float(temp[i+1])
        tempY = float(temp[i+2])
        data.append([tempX, tempY])
        
    data = np.array(data)
    sd = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    
    for i in range(nf):
        data[i][0] = (data[i][0]-mean[0])/sd[0]
        data[i][1] = (data[i][1]-mean[1])/sd[1]
    
    return np.array(data)

#filtering and locating required files among pool of files
def getData(phase):
    fileNames = ['a', 'ai', 'dA', 'lA', 'tA']
    dataBase = {}
    for i in fileNames:
        fData = []
        fileLoc = "writings/"+str(i)+"/"+phase
        for fileName in os.listdir(fileLoc):
            data =  readFiles(fileLoc + "/" + fileName)
            fData.append(data)
        dataBase[i] = fData
    return dataBase

def processData(data):
    inputArr = ['a', 'ai', 'dA', 'lA', 'tA']
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

def KNN(elem, trainData):
    inputArr = ['a', 'ai', 'dA', 'lA', 'tA']
    arr = []
    elemLen = len(elem)
    count = {}
    for i in inputArr:
        count[i] = 0
        for h in trainData[i]:
            arr.append([np.linalg.norm(np.array(h) - np.array(elem)), i])
    arr.sort();
    for i in range(15):
        count[arr[i][1]]+=1
    return max(count, key=count.get)

inputArr = ['a', 'ai', 'dA', 'lA', 'tA']
trainData = getData("train")
devData = getData("dev")
trainData = processData(trainData)
devData = processData(devData)

inputArr = ['a', 'ai', 'dA', 'lA', 'tA']
count = 0
maxCount = 0
for i in inputArr:
    for elem in devData[i]:
        index = KNN(elem, trainData)
        maxCount+=1
        if(index == i):
            #print("Pass")
            count+=1
        #else:
            #print(f"Fail, Expected : {i}, Obtained : {index}")
print(f"Accuracy : {(count*100)/maxCount}%")