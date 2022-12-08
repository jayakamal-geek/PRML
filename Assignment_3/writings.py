# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 21:31:38 2022

@author: jayak
"""

##importing the required libraries
import os
import numpy as np
from matplotlib import pyplot as plt

def readFiles(filePath):
    fp = open(filePath, "r")
    temp = fp.read().strip().split(" ")
    nf = int(temp[0])
    data = []
    minX = np.inf
    maxX = -1*np.inf
    maxY = -1*np.inf
    minY = np.inf
    
    for i in range(0, 2*nf, 2):
        tempX = float(temp[i+1])
        tempY = float(temp[i+2])
        minX = min(tempX, minX)
        minY = min(tempY, minY)
        maxX = max(tempX, maxX)
        maxY = max(tempY, maxY)
        data.append([tempX, tempY])
        
    shiftX = maxX - minX
    shiftY = maxY - minY
    
    for i in range(nf):
        data[i][0] = (data[i][0]-minX)/shiftX
        data[i][1] = (data[i][1]-minY)/shiftY
    
    return nf, np.array(data)

#filtering and locating required files among pool of files
def getData(phase):
    fileNames = ['a', 'ai', 'dA', 'lA', 'tA']
    dataBase = {}
    for i in fileNames:
        nfVals = []
        fData = []
        fileLoc = "writings/"+str(i)+"/"+phase
        for fileName in os.listdir(fileLoc):
            nf, data =  readFiles(fileLoc + "/" + fileName)
            nfVals.append(nf)
            fData.append(data)
        dataBase[i] = [nfVals, fData]
    return dataBase

def dtw(a, len_a, b, len_b):
    dtw_matrix = np.full((len_a+1, len_b+1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, len_a+1):
        for j in range(1, len_b+1):
            dtw_matrix[i, j] = np.linalg.norm(a[i-1]-b[j-1]) + np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
    return dtw_matrix[len_a,len_b]

def main():
    trainData = getData("train")
    devData = getData("dev")
    inputArr = ['a', 'ai', 'dA', 'lA', 'tA']
    rightAns = 0
    wrongAns = 0
    
    for i in inputArr:
        devArr = devData[i]
        devLen = len(devArr[1])
        for j in range(devLen):
            minErr = np.inf
            index = np.inf
            for z in inputArr:
                trainArr = trainData[z]
                trainLen = len(trainArr[1])
                for k in range(trainLen):
                    dtw_res = dtw(trainArr[1][k], trainArr[0][k], devArr[1][j], devArr[0][j])
                    if(dtw_res < minErr):
                        minErr = dtw_res
                        index = z
            if(index == i):
                rightAns = rightAns + 1
            else:
                wrongAns = wrongAns + 1
    print("Accuracy : " + str(100*rightAns/(rightAns+wrongAns)) + "%")
                
main()