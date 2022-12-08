# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 01:08:53 2022

@author: jayak
"""

##importing the required libraries
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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
        data.append([tempX, tempY])
    data = np.array(data)
    sd = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    
    for i in range(nf):
        data[i][0] = (data[i][0]-mean[0])/sd[0]
        data[i][1] = (data[i][1]-mean[1])/sd[0]
    
    return data

#filtering and locating required files among pool of files
def getData(phase):
    fileNames = ['a', 'ai', 'dA', 'lA', 'tA']
    dataBase = {}
    for i in fileNames:
        fData_sep = []
        fData = []
        fileLoc = "writings/"+str(i)+"/"+phase
        for fileName in os.listdir(fileLoc):
            data =  readFiles(fileLoc + "/" + fileName)
            fData_sep.append(data)
            for elem in data:
                fData.append(elem)
        dataBase[i] = [fData, fData_sep]
    return dataBase

def getProbHMM(states,seq,pi,a,b):
    l = len(seq)
    alpha = np.empty([l,states])
    for i in range(states):
        alpha[0,i] = pi[i]*b[i,0,seq[0]]*a[i,0]
    for t in range(1,l):
        for st in range(states):
            alpha[t,st] = alpha[t-1,st]*b[st,0,seq[t]]*a[st,0] 
            if st!=0:
                alpha[t,st] += alpha[t-1,st-1]*b[st-1,1,seq[t]]*a[st-1,1] 
    prob = np.sum(alpha[l-1])
    return prob

trainData = getData("train")
devData = getData("dev")
inputArr = ['a', 'ai', 'dA', 'lA', 'tA']
ptsArr = []

for letter in inputArr:
    for elem in trainData[letter][0]:
        ptsArr.append(elem)

kmeans = KMeans(n_clusters=20, max_iter=1000, random_state=0).fit(ptsArr)

for letter in inputArr:
    fp = open('sequence_' + str(letter) + '.txt', 'w')
    for elem in trainData[letter][1]:
        temp = kmeans.predict(elem)
        for item in temp:
            fp.write(str(item)+" ")
        fp.write("\n")
    fp.close()
    cmd = "wsl ./train_hmm sequence_"+ str(letter) +".txt 1234 3 20 .01"
    os.system(cmd)
    
prob = {}
states = 3
symbols = 20
rightAns = 0
wrongAns = 0

for letter in inputArr:
    temp = []
    a = np.empty([states,2])
    b = np.empty([states,2,symbols])
    fp = open('sequence_' + str(letter) + '.txt.hmm', 'r')
    fp.readline()
    fp.readline()
    for i in range(states):
        temp.append(fp.readline().strip().split("\t"))
        temp.append(fp.readline().strip().split("\t"))
        fp.readline()
    temp = np.array(temp)
    for j in range(states):
        a[j,0] = temp[j*2,0]
        a[j,1] = temp[j*2+1,0]
        b[j,0] = temp[j*2,1:]
        b[j,1] = temp[j*2+1,1:]
    prob[letter] = [a, b]
    fp.close()
    
for i in inputArr:
    for elem in devData[i][1]:
        maxProb = 0
        index = np.inf
        temp = kmeans.predict(elem)
        for j in inputArr:
            tempVal = getProbHMM(states, temp, np.array([1, 0, 0]), prob[j][0], prob[j][1])
            if(tempVal > maxProb):
                maxProb = tempVal
                index = j
        if(i == index):
            rightAns+=1
        else:
            wrongAns += 1
    
print("Accuracy : " + str(100*rightAns/(rightAns+wrongAns)) + "%")