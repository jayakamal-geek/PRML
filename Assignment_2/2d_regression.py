# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 21:03:55 2022

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

def fillMatrix(xIn, degree):
    x = []    
    for [a, b] in xIn:
        rowData = []
        subPower = 0
        while (subPower <= degree):
            power = 0
            while (power<=subPower):
                rowData.append((a**power)*(b**(subPower-power)))
                power+=1
            subPower+=1
        x.append(rowData)
    return np.array(x)



#This Function computes the RMSE when a Least squared error linear regression model of basis polynomial degree 15 is used for the data.
def lse(trainData_2d, devData_2d):
    X = fillMatrix(trainData_2d[0], 11)
    Y = fillMatrix(devData_2d[0], 11)
    W = np.linalg.inv(X.T@X)@X.T@(np.array(trainData_2d[1]))
    calcVals = Y@W
    print("RMSE LSR : " + str(np.linalg.norm(devData_2d[1]-calcVals)/math.sqrt(X.shape[0])))

#This Function computes the RMSE when a Ridge regression model of basis polynomial degree 10 is used for the data with lamda 0.91*10^(-5).
def ridge(trainData_2d, devData_2d):
    X = fillMatrix(trainData_2d[0], 11)
    Y = fillMatrix(devData_2d[0], 11)
    tempMat = (X.T)@X
    W = np.linalg.inv(tempMat + ((10**(-5))*np.identity(tempMat.shape[0])))@(X.T)@(np.array(trainData_2d[1]))
    calcVals = Y@W
    print("RMSE RIDGE : " + str(np.linalg.norm(devData_2d[1]-calcVals)/math.sqrt(X.shape[0])))
 


def main():
    trainData_2d = getData('2d_team_36_train.txt') #Reading In the input data
    devData_2d = getData('2d_team_36_dev.txt') #Reading In the test data
    lse(trainData_2d, devData_2d)   #calling LSE Linear Regression function
    ridge(trainData_2d, devData_2d) #calling Ridge Regression Function
    
main()