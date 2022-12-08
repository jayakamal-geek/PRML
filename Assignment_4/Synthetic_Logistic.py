# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:59:16 2022

@author: jayak
"""

##importing the required libraries
import os
import numpy as np
import math

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

def genPhi(x,degree):
    phi = []
    count = 0
    for x_i in x:
        poly = []
        for d1 in range(degree+1):
            for d2 in range(degree+1):
                if d1 + d2 <= degree:
                    poly.append(pow(x_i[0],d1)*pow(x_i[1],d2))
        phi.append(poly) 
        count = len(poly)

    phi = np.array(phi)
    return phi, count

def sigmoid(a):
    if a > 100 :
        return 1 - 1e-50
    elif a < -100:
        return 1e-50
    x = math.exp(-a)
    return 1/(1 + x)

def getCost(x, y, w, degree):
    cost = 0
    phi, cnt = genPhi(x,degree)
    for i in range(phi.shape[0]):
        h_i = sigmoid(np.dot(phi[i].reshape(1, phi[i].shape[0]), w))
        cost += y[i] * np.log(h_i) + (1 - y[i]) * np.log(1 - h_i)
    cost = -cost
    return cost/x.shape[0]

def getW(x, y, degree):
    phi, count = genPhi(x,degree)
    w = np.zeros((count, 1))
    w[0] = 1
    learn_rate = 0.000001
    cost_all = []
    y = y - 1

    for i in range(100):
        summ = np.zeros((phi.shape[1],1))
        for j in range(x.shape[0]):
            phi_j = phi[j].reshape(1, phi[j].shape[0])
            summ = np.add(summ, (sigmoid(np.dot(phi_j, w)) - y[j]) * phi_j.transpose())
        w = w - learn_rate * summ
        cost_all.append(getCost(x,y,w,degree))
    return w, cost_all

def testModel(x,w, degree):
    x = np.array(x)
    pred = [] 
    prob = []   
    phi, count = genPhi(x,degree)
    for i in range(phi.shape[0]):
        h_i = sigmoid(np.dot(phi[i].reshape(1, phi[i].shape[0]), w))
        prob.append(h_i)
        if h_i >= 0.5 :
            pred.append(1)
        else:
            pred.append(0)
    return pred

w, cost_all = getW(np.array(trainData[:,:2]), np.array(trainData[:,2]), 2)
pred = testModel(devData[:1000,:2], w, 2)
true = 0
false = 0
for i in range(len(devData)):
    if pred[i] == 0 and i >= 500:
        false += 1
    elif pred[i] == 1 and i < 500:
        false += 1
    else:
        true += 1
print("Accuracy:", (true*100)/(true+false), "%")