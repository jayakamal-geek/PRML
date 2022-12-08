#!/usr/bin/python

import os
import numpy as np
from matplotlib import pyplot as plt


# DATA_BIT1 = 0
# DATA_BYTE1 = 'opencountry_art582.jpg_color_edh_entropy'
DATAPATH = 'Features/Features/'
# DATA = f"{DATAPATH}{['opencountry', 'mountain', 'highway', 'forest', 'coast'][DATA_BIT1]}/train/{DATA_BYTE1}"

def loadfile (path):
    with open (path, 'r') as f:
        l = [([float (y) for y in x.split (' ')]) for x in f.read ().split ('\n')[:-1]]
        lx = []
        for lp in l:
            lx.extend (lp)
        return np.array (lx)
    
def loaddir (path):
    l = []
    for d in os.listdir (path):
        l.append (loadfile (path + d))
    return np.array (l)

def loadclassdirs (path):
    l = {}
    for d in os.listdir (path):
        l[d] = loaddir (path + d + '/train/')
    return l

NM = 0.
NV = 0.
def normalize (data):
    global NM, NV
    xmean = np.zeros (len (data[list (data.keys ())[0]][0]))
    total = 0
    for c in data:
        for x in data[c]:
            xmean += x
        total += len (data[c])
    
    xmean /= total
    NM = xmean

    xvar = np.zeros (len (data[list (data.keys ())[0]][0]))
    for c in data:
        for x in data[c]:
            xvar += (x - xmean) ** 2
    
    xvar /= total
    xvar = np.sqrt (xvar)
    NV = xvar
    
    for c in data:
        for idx in range (len (data[c])):
            data[c][idx] = (data[c][idx] - xmean) / xvar
            
    return data

def normalize_dev (data):
    # return normalize (data)
    global NM, NV
    xmean = NM
    xvar = NV
    for c in data:
        for idx in range (len (data[c])):
            data[c][idx] = (data[c][idx] - xmean) / xvar
            
    return data

def loadclassdirs_dev (path):
    l = {}
    for d in os.listdir (path):
        l[d] = loaddir (path + d + '/dev/')
    return normalize (l)




data = loadclassdirs (DATAPATH)
data_dev = loadclassdirs_dev (DATAPATH)

################# DATA LOADED ########################
