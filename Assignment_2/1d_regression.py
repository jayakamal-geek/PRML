# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 18:45:32 2022

@author: jayak
"""

#include necessary modules
import numpy as np
import math
from matplotlib import pyplot as plt

#This function fetches data from txtfiles and returns them as a list of lists
def getData(fileName):
    params = []
    target = []
    with open(fileName, 'r') as dataFile:
        dataLines = dataFile.readlines()

    for dataLine in dataLines:
        tempData = dataLine.split(' ')
        data = []
        dataLineCount = len(tempData)
        val = dataLineCount-1
        for i in range(val):
            data.append(float(tempData[i]))
        params.append(data)
        target.append(float(tempData[dataLineCount-1].strip('\n')))
    return [params, target] #print(data)

#This function fills the Matrix with power of x_i for all 0<=i<N and returns the Matrix
def fillMatrix(xIn, degree):
    x = []
    for [elem] in xIn:
        rowData = []
        power = 0 
        prod = 1
        while (power<=degree):
            rowData.append(prod)
            prod = prod*elem
            power+=1
        x.append(np.array(rowData))
    return np.array(x)

#This Function computes the RMSE when a Least squared error linear regression model of basis polynomial degree 15 is used for the data.
def lse(trainData_1d, devData_1d):
    X = fillMatrix(trainData_1d[0], 15)
    Y = fillMatrix(devData_1d[0], 15)
    W = np.linalg.pinv(X)@(np.array(trainData_1d[1]))
    calcVals = Y@W
    print("RMSE LSR : " + str(np.linalg.norm(devData_1d[1]-calcVals)/math.sqrt(X.shape[0])))

#This Function computes the RMSE when a Ridge regression model of basis polynomial degree 10 is used for the data with lamda 0.91*10^(-5).
def ridge(trainData_1d, devData_1d):
    X = fillMatrix(trainData_1d[0], 10)
    Y = fillMatrix(devData_1d[0], 10)
    tempMat = (X.T)@X
    W = np.linalg.inv(tempMat + (0.91 * (10**(-5)))*np.identity(tempMat.shape[0]))@(X.T)@(np.array(trainData_1d[1]))
    calcVals = Y@W
    print("RMSE RIDGE : " + str(np.linalg.norm(devData_1d[1]-calcVals)/math.sqrt(X.shape[0])))
 


def main():
    trainData_1d = getData('1d_team_36_train.txt') #Reading In the input data
    devData_1d = getData('1d_team_36_dev.txt') #Reading In the test data
    lse(trainData_1d, devData_1d)   #calling LSE Linear Regression function
    ridge(trainData_1d, devData_1d) #calling Ridge Regression Function
    
main()