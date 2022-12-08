# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:52:25 2022

@author: jayak
"""

##importing the required libraries
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

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

inputArr = ['a', 'ai', 'dA', 'lA', 'tA']
trainData = getData("train")
devData = getData("dev")
trainData = processData(trainData)
devData = processData(devData)

dataArr = []
classArr = []
for i in inputArr:
    for elem in trainData[i]:
        dataArr.append(elem)
        classArr.append(i)
clf = MLPClassifier(solver="lbfgs", alpha=1e-3, hidden_layer_sizes=(128,20), random_state=1)
clf.fit(dataArr, classArr)
true = 0
false = 0

for i in inputArr:
    for elem in devData[i]:
        cl = clf.predict(elem.reshape(1, 40))
        if(cl == i):
            true+=1
        else:
            false+=1
print("Accuracy :", (true*100)/(true+false))