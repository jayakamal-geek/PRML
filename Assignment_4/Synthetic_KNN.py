# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:40:39 2022

@author: jayak
"""

import os
import numpy as np
import math
from matplotlib import pyplot as plt

def readFile(keyword):
    filePath = "36/"+keyword+".txt"
    fp = open(filePath, "r")
    data = fp.readlines()
    tempLen = len(data)
    
    for i in range(tempLen):
        data[i] = data[i].strip().split(",")
        data[i][0] = float(data[i][0])
        data[i][1] = float(data[i][1])
        data[i][2] = int(data[i][2])
    return data

def KNN(data, k):
    arr = []
    count = 0
    for elem in trainData:
        arr.append([np.linalg.norm(np.array(elem[0:2]) - np.array(data[0:2])), elem[2]])
    arr.sort()
    
    for i in range(k):
        if(arr[i][1] == 1):
            count+=1
        else:
            count-=1
    if(count>0):
        return 1
    return 2

trainData = readFile("train")
devData = readFile("dev")

count = 0
countMax = len(devData)
    
for elem in devData:
    if(elem[2] == KNN(elem, 25)):
        count+=1
print(f"Accuracy : {(count*100)/countMax}%")