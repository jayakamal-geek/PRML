# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 23:13:40 2022

@author: jayak
"""

import os
import numpy as np
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("error")

DATA_BIT1 = 0
DATA_BYTE1 = 'opencountry_art582.jpg_color_edh_entropy'
DATAPATH = 'imgDataSet/'
SDATAPATH = ''
DATA = f"{DATAPATH}{['opencountry', 'mountain', 'highway', 'forest', 'coast'][DATA_BIT1]}/train/{DATA_BYTE1}"

K = 3
EPS = 1e-50

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

data = sloadclassdirs (0)
classwise_K = {'1': 12,
               '2': 12}

class Gaussian:
    def __init__ (self, mean, cov, pi):
        self.dim = mean.shape[0]
        self.mean = mean
        self.cov = cov
        self.pi = pi
    
    def __call__ (self, x):
        try:
#             fac = - (self.dim / 2) * np.log (2 * np.pi) - 0.5 * (np.log (np.abs (np.linalg.det (self.cov))))
#             exp = -0.5 * ((x - self.mean).T @ np.linalg.inv (self.cov) @ (x - self.mean))
            fac = 1 / np.sqrt (((2 * np.pi) ** self.dim) * np.abs (np.linalg.det (self.cov)))
            exp = np.exp (-0.5 * ((x - self.mean).T @ np.linalg.inv (self.cov) @ (x - self.mean)))
        except RuntimeWarning:
#             print (np.linalg.det (self.cov))
            return 0.
        
        return self.pi * fac * exp
    
def find_priors (classwise_data):
    p = {}
    total = 0
    for c in classwise_data:
        total += len (classwise_data[c])
        p[c] = (len (classwise_data[c]))
    
    for c in p:
        p[c] /= total
    
    return p

# out: array (float : p(x) for each x in data)
def expectation (gaussians, data):
    total = []
    gammas = []
    for x in data:
        i = []
        t = 0.
        for g in gaussians:
            i.append (g (x))
            t += i[-1]
            
        for ix in range (len (i)):
            i[ix] = i[ix] / t
            
        gammas.append (i)
    
    for j in range (len (gammas[0])):
        t = 0.
        for i in range (len (gammas)):
            t += gammas[i][j]
        total.append (t)
    
    return total, gammas


def maximization (gaussians, totals, latents, data, find_cov):
    tt = 0.
    for t in totals:
        tt += t
    
    latents = np.array (latents)
    
    Delta = 0
    
    for idx in range (len (gaussians)):
        g = gaussians[idx]
        
        oldmean = g.mean
        oldcov = g.cov
        oldpi = g.pi
        
        g.mean = (1 / totals[idx]) * np.mean (((data.T * latents[:, idx]).T), axis=0)
        g.cov = find_cov (totals[idx], latents[:, idx], data, g.mean)
        g.pi = float ((totals[idx] / tt))
        
        Delta += abs (oldpi - g.pi)
        Delta += np.sum (np.abs (oldmean - g.mean))
        Delta += np.sum (np.abs (oldcov - g.cov))
    
    return Delta
        
        
def init_gaussians (data, k):
    gaussians = []
    
    for i in range (k):
        gaussians.append (Gaussian (data [np.random.choice (range (len (data)))], 1/k * np.identity (len (data[0])), 1/k))
        
    return gaussians

def find_cov_full (total, latents, data, mean):
    s = 0.
    for n in range (len (data)):
        s += ((data[n] - mean)[:,None] @ ((data[n] - mean)[:,None].T)) * latents[n]
    
    return (1 / total) * s

def find_cov_diag (total, latents, data, mean):
    return np.diag (np.diag (find_cov_full (total, latents, data, mean)))

def find_cov_sigma (total, latents, data, mean):
    
    avg = 0.
    fc = find_cov_full (total, latents, data, mean)
    for i in fc:
        for j in i:
            avg += j
            
    avg /= fc.shape[0] * fc.shape[1]
    
    return np.identity ((len (data[0]), len (data[0]))) * avg

def train_full (log = False):
    models = {}
    for classkey in data:
        print (f"\nProcessing {classkey}", end='') if log else 0
        data[classkey] = np.array (data[classkey])
        
        gaussians = init_gaussians (data[classkey], classwise_K[classkey])
        
        for i in range (100):
            print ('.', end='') if log else 0
            t, g = expectation (gaussians, data[classkey])
            Delta = maximization (gaussians, t, g, data[classkey], find_cov_full)
            
            if (Delta < EPS):
                break
        
        models[classkey] = gaussians
    
    return models

gx = train_full (True)

def probs (gm, x):
    terms = {}
    p = find_priors (data)
    total = 0.
    
    for classkey in gm:
        pspec = 0.
        for g in gm[classkey]:
#             print (g (x), pspec)
            pspec += g (x)
            
        terms[classkey] = p[classkey] * pspec
        total += terms[classkey]
    
    for c in terms:
        terms[c] /= total
    
    return p

def classify (gm, x):
    p = probs (gm, x)
    
    maxc = list (p.keys ())[0]
    
    for c in p:
        if p[c] > p[maxc]:
            maxc = c
    
    return maxc

find_priors (data)
devdata = sloadclassdirs (1)
for c in devdata:
    for x in devdata[c]:
        print (c, classify (gx, x))
        
for c in gx:
    for g in gx[c]:
        print (g.mean, g.pi)