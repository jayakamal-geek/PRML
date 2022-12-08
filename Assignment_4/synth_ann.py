##importing the required libraries
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier

SDATAPATH = '36/'
SNM = 0.
SNV = 0.
def snormalize (data):
    xmean = np.zeros (2)
    total = 0
    for c in data:
        for x in data[c]:
            xmean += x
        total += len (data[c])
    
    xmean /= total
    xvar = np.zeros (2)
    for c in data:
        for x in data[c]:
            xvar += (x - xmean) ** 2
    
    xvar /= total
    xvar = np.sqrt (xvar)
    
    for c in data:
        for idx in range (len (data[c])):
            data[c][idx] = (data[c][idx] - xmean) / xvar
            
    return data


def snormalize_dev (data):
    # return normalize (data)
    global SNM, SNV
    xmean = SNM
    xvar = SNV
    for c in data:
        for idx in range (len (data[c])):
            data[c][idx] = (data[c][idx] - xmean) / xvar
            
    return data
def sloadclassdirs (t):
    p = ''
    with open (f"{SDATAPATH}36/{['train', 'dev'][t]}.txt", 'r') as f:
        p = [x.split(',') for x in f.read ().split ('\n')]
        
    d = {}
    for s in p:
        if len (s) < 2:
            continue
        
        if s[2] in d:
            d[s[2]].append (np.array ([float (s[0]), float (s[1])]))
        else:
            d[s[2]] = []
        
    return snormalize (d)

minNF = np.inf
trainData = sloadclassdirs (0)
devData = sloadclassdirs (1)

dataArr = []
classArr = []
for c in trainData:
    dataArr.extend (trainData[c])
    classArr.extend ([c for _ in trainData[c]])

clf = MLPClassifier(solver="lbfgs", alpha=1e-3, hidden_layer_sizes=(128,20), random_state=1)
clf.fit(dataArr, classArr)
true = 0
false = 0

for c in devData:
    for elem in devData[c]:
        cl = clf.predict(elem.reshape(1, 2))
        if(cl == c):
            true+=1
        else:
            false+=1
print("Accuracy :", (true*100)/(true+false))