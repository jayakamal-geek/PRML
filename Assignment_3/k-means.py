# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 23:09:30 2022

@author: jayak
"""

import os
import numpy as np
from matplotlib import pyplot as plt

DATA_BIT1 = 0
DATA_BYTE1 = 'opencountry_art582.jpg_color_edh_entropy'
DATAPATH = 'imgDataSet/'
SDATAPATH = ''
DATA = f"{DATAPATH}{['opencountry', 'mountain', 'highway', 'forest', 'coast'][DATA_BIT1]}/train/{DATA_BYTE1}"

K = 3
EPS = 1e-7

def loadfile (path):
    with open (path, 'r') as f:
        l = [([float (y) for y in x.split (' ')]) for x in f.read ().split ('\n')[:-1]]
        lx = []
        for lp in l:
            lx.extend (lp)
        return np.array (l[0])
    
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

def normalize (data):
    xmean = np.zeros (23)
    total = 0
    for c in data:
        for x in data[c]:
            xmean += x
        total += len (data[c])
    
    xvar = np.zeros (23)
    for c in data:
        for x in data[c]:
            xvar += (x - xmean) ** 2
    
    xmean /= total
    xvar /= total
    xvar = np.sqrt (xvar)
    
    for c in data:
        for idx in range (len (data[c])):
            data[c][idx] = (data[c][idx] - xmean) / xvar
            
    return data


def snormalize (data):
    xmean = np.zeros (2)
    total = 0
    for c in data:
        for x in data[c]:
            xmean += x
        total += len (data[c])
    
    xvar = np.zeros (2)
    for c in data:
        for x in data[c]:
            xvar += (x - xmean) ** 2
    
    xmean /= total
    xvar /= total
    xvar = np.sqrt (xvar)
    
    for c in data:
        for idx in range (len (data[c])):
            data[c][idx] = (data[c][idx] - xmean) / xvar
            
    return data


def loadclassdirs_dev (path):
    l = {}
    for d in os.listdir (path):
        l[d] = loaddir (path + d + '/dev/')
    return normalize (l)

#############################################
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


# data = loadclassdirs (DATAPATH)
# classwise_K = {'opencountry': 12,
#              'mountain': 12,
#              'highway': 12,
#              'forest': 12,
#              'coast': 12}

data = sloadclassdirs (0)

classwise_K = {'1': 12,
               '2': 12}

def init_centroids (k, data):
    l = []
    for i in range (k):
        l.append (data [np.random.choice (range (len (data)))])
    
    return l

def train_full (log = False):
    model = {}
    for c in data:
        print (f'\nProcessing {c}', end='') if log else 0
        centroids = init_centroids (classwise_K[c], data[c])
        
        for i in range (100):
            print ('.', end='') if log else 0
            pointcols = [[] for g in centroids]

            for x in data[c]:
                pointcols [np.argmin ([np.linalg.norm (x - g) for g in centroids])].append (x)

            centroids = []
            for pset in pointcols:
                centroids.append (np.mean (pset, axis=0))
                
        model[c] = centroids
    
    return model
m = train_full (True)

for c in data:
    for x in data[c][:100]:
        plt.scatter (x[0], x[1], c=['red', 'blue'][int(c) - 1], s=15)
        
    for g in m[c]:
        plt.scatter (g[0], g[1], c=['orange', 'purple'][int(c) - 1], s=75)
        
plt.show ()

def plot_decision_boundary (model, resolution = 50, with_train_data = True, path = None):
  fig = plt.figure (figsize=(15,15))
  ax = plt.axes ()

  xspc = np.linspace (-0.0005, 0.0005, resolution)
  yspc = np.linspace (-0.0005, 0.0005, resolution)
  X, Y = np.meshgrid (xspc, yspc)
  z = [[classify ([xspc[i], yspc[j]]) for i in range (len (xspc))] for j in range (len (yspc))]

  ax.contourf (X, Y, np.asarray (z), 100)

  if with_train_data:
      for c in data:
        ax.scatter (data[c][:, 0], data[c][:, 1], color=['gray', 'r', 'b', 'orange'][c], lw=1, marker='x', label=c)  

  plt.legend ()
  plt.show ()
  
  def classify (x):
    x = np.array (x)
    minc = list (m.keys ()) [0]
    mini = 0
    for c in m:
        for idx in range (len (m[c])):
            if (np.linalg.norm (x - m[c][idx]) < np.linalg.norm (x - m[minc][mini])):
                minc = c
                mini = idx
                
    return minc

for c in data:
    for x in data[c]:
        if (c != classify (x)):
            print ('f: ' + str(x))