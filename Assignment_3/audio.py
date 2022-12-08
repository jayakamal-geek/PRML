# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 21:34:58 2022

@author: jayak
"""

##importing the required libraries
import os
import numpy as np
from matplotlib import pyplot as plt 

#reading data from files
def readFiles(filePath):
    fp = open(filePath, "r")
    nc, nf = [int(x) for x in fp.readline().strip().split(" ")]
    data = []
    for i in range(nf):
        data.append(fp.readline().strip().split(" "))
    fp.close()
    return nc, nf, np.array(data)

#filtering and locating required files among pool of files
def getData(phase):
    fileNums = [1, 2, 4, 5, 9]
    dataBase = {}
    for i in fileNums:
        ncVals = []
        nfVals = []
        fData = []
        fileLoc = "soundTracks/"+str(i)+"/"+phase
        for fileName in os.listdir(fileLoc):
            if fileName.endswith(".mfcc"):
                nc, nf, data =  readFiles(fileLoc + "/" + fileName)
                ncVals.append(nc)
                nfVals.append(nf)
                fData.append(data)
        dataBase[i] = [ncVals, nfVals, fData]
    return dataBase

#DTW function to compute euclidian distance matrix
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
    inputArr = [1, 2, 4, 5, 9]
    rightAns = 0
    wrongAns = 0
    
    for i in inputArr:
        devArr = devData[i]
        devLen = len(devArr[2])
        for j in range(devLen):
            minErr = np.inf
            index = np.inf
            for z in inputArr:
                trainArr = trainData[z]
                trainLen = len(trainArr[2])
                for k in range(trainLen):
                    dtw_res = dtw(np.asarray(trainArr[2][k], dtype=float), trainArr[1][k], np.asarray(devArr[2][j], dtype=float), devArr[1][j])
                    if(dtw_res < minErr):
                        minErr = dtw_res
                        index = z
            if(index == i):
                rightAns = rightAns + 1
            else:
                wrongAns = wrongAns + 1
    print("Accuracy : " + str(100*rightAns/(rightAns+wrongAns)) + "%")
                
main()