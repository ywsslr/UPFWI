import torch
import numpy as np

class LogTransform(object):
    def __init__(self, k=1, c=0):
        self.k = k
        self.c = c

    def __call__(self, data):
        return log_transform(data, k=self.k, c=self.c)
    
def log_transform(data, k=1, c=0):
    return (np.log1p(np.abs(k * data) + c)) * np.sign(data)

class MinMaxNormalize(object):
    def __init__(self, datamin, datamax, scale=2):
        self.datamin = datamin
        self.datamax = datamax
        self.scale = scale
    def __call__(self, vid):
        return minmax_normalize(vid, self.datamin, self.datamax, self.scale)

def minmax_normalize(vid, vmin, vmax, scale=2):
    vid -= vmin
    vid /= (vmax - vmin)
    return (vid - 0.5) * 2 if scale == 2 else vid

def denormalize(vid,vmin,vmax,scale=2):
    if scale==2:
        vid=(vid+1)/2
    vid=(vmax-vmin)*vid+vmin
    return vid
def exp_transform(data, k=1, c=0):
    return (np.expm1(np.abs(data)) - c) * np.sign(data) / k