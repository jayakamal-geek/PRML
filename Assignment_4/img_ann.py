##importing the required libraries
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from loader import *

minNF = np.inf
trainData = normalize (loadclassdirs (DATAPATH))
devData = normalize_dev (loadclassdirs_dev (DATAPATH))
# trainData = normalize (loadclassdirs (DATAPATH))
# devData = normalize (loadclassdirs_dev (DATAPATH))
# trainData = loadclassdirs (DATAPATH)
# devData = loadclassdirs_dev (DATAPATH)

dataArr = []
classArr = []
for c in trainData:
    dataArr.extend (trainData[c])
    classArr.extend ([c for _ in trainData[c]])

clf = MLPClassifier(solver="adam", alpha=1e-3, hidden_layer_sizes=(128,20), random_state=1)
clf.fit(dataArr, classArr)
true = 0
false = 0

for c in devData:
    for elem in devData[c]:
        cl = clf.predict(elem.reshape(1, 828))
        if(cl == c):
            true+=1
        else:
            false+=1
print("Accuracy :", (true*100)/(true+false))