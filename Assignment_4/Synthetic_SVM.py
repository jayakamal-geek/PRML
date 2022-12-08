# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:38:37 2022

@author: jayak
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC

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
    return np.array(data)

trainData = readFile("train")
devData = readFile("dev")

trainDataVals = np.array(trainData[:,:2])
clf = SVC(kernel='poly')
clf.fit(trainDataVals, trainData[:, 2])
true = 0
false = 0

for elem in devData:
    cl = clf.predict(elem[:2].reshape(1, 2))
    if(cl == elem[2]):
        true+=1
    else:
        false+=1
print("Accuracy :", (true*100)/(true+false), "%")