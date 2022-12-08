##importing the required libraries
import os
import numpy as np
import math
from matplotlib import pyplot as plt
from loader import *
from pca import pca
from lda import lda

import warnings
warnings.filterwarnings('ignore')

def transform (data):
    fin = []
    for c in data:
        app = [[c]] * len (data[c])
        fin.extend (np.concatenate ((data[c], app), axis=1))
    
    return fin

print ("Loading data...")
trainData = pca (normalize (loadclassdirs (DATAPATH)), newdim=80)
print ("Training data loaded.")
devData = pca (normalize_dev (loadclassdirs_dev (DATAPATH)), newdim=80)
print ("Development data loaded.")

classes = list (trainData.keys ())

# x = []
# for i in range (len (trainData)):
#     for j in range (i + 1, len (trainData)):
#         x.append (abs (np.mean (trainData[classes[i]] - np.mean (trainData[classes[j]]))))

trainData = lda (trainData, newdim = 80)
devData = lda (devData, newdim = 80)
print ('Data reduced.')
    
# y = []
# for i in range (len (trainData)):
#     for j in range (i + 1, len (trainData)):
#         y.append (abs (np.mean (trainData[classes[i]] - np.mean (trainData[classes[j]]))))

# plt.plot (x)
# plt.plot (y)
# plt.show ()

trainData = transform (trainData)
devData = transform (devData)
print ('Data transformed.')

# print (f"PCA validation: ", end='')
# x = []
# for i in range (80):
#     x.append (np.var (np.array (trainData)[:, i].astype (float)))
# plt.plot (x)
# plt.show ()
# print ('')



def KNN(data, k):
    arr = []
    for elem in trainData:
        arr.append([np.linalg.norm(np.array(elem[:-1]).astype (complex).astype (float) - np.array(data[:-1]).astype (complex).astype (float)), elem[-1]])
    arr.sort()
    arr = np.array (arr)
    
    a, b = np.unique(arr[:k, -1], return_counts=True)

    return a [np.argmax (b)]


k_opt = math.ceil(math.sqrt(len(trainData)))
for i in range(1, k_opt, 2):
    count = 0
    countMax = len(devData)
    
    for elem in devData:
        if(elem[-1] == KNN(elem, i)):
            count+=1
    print(f"{i} : {count/countMax}")