# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 08:44:59 2021

@author: Abberior_admin
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from datetime import datetime
#import sys
#sys.path.append(r'\\192.168.169.11\data\User\vanderVoortN\FRC\Code')
#import findPeaksLib

def findMaxima(data):
    """
    Function has been yanked from FindPeaksLib FRC for simplicity of structure
    finds all local peaks according to 'raindrop' model.
    All pixel get a raindrop. The rain flows uphill in neighbouring pixels
    Priority is given to x-direction
    input: A double gaussian-smoothed input image
    returns: a list of where the raindrops have converged
    """
    
    xlen = data.shape[0]
    ylen = data.shape[1]
    points = np.zeros([xlen*ylen, 2], dtype = np.int)
    eps_step = 1
    for x in range(xlen):
        for y in range(ylen):
            points[x*ylen + y, 0] = x
            points[x*ylen + y, 1] = y
    points_copy = points.copy()
    padded_data = np.pad(data, 1, mode = 'constant')

    while(eps_step != 0):
        for index, point in enumerate(points):
            x,y = point
            xset = padded_data[x: x + 3, y + 1]
            if xset[0] > xset[1] and xset[0] > xset[2]: #move point left
                points_copy[index, 0] -= 1
                continue
            elif xset[2] > xset[1] and xset[2] > xset[0]: # move point right
                points_copy[index, 0] += 1
                continue
            yset = padded_data[ x + 1, y: y + 3]
            if yset[0] > yset[1] and yset[0] > yset [2]: #move up
                points_copy[index, 1] -= 1
            if yset[2] > yset[1] and yset[2] > yset [1]: #move down
                points_copy[index, 1] += 1
        eps_step = np.linalg.norm(points-points_copy)
        points = points_copy.copy()
        
    #peaks is array, each row contains xcoord, ycoord and intensity
    #from finding algorithm, peaks contains many duplicates.
    peak_intensities = np.zeros(xlen * ylen)
    for i in range(xlen * ylen):
        peak_intensities[i] = data[points[i,0], points[i,1]]
    peaks = np.array([points[:, 0], points[:, 1]]).transpose()
    #sort according to intensity
    peaks = peaks[peak_intensities.argsort(-1)][-1::-1]
    return peaks

def smooth_image(im, sigma = 1):
    """this function has been yanked from FindPeaksLib in FRC too"""
    return gaussian_filter(im.astype(np.double), sigma)

def filterjunk(image, peaks, minarea = 20, bglevel = 2,
                return_diagnostics = False):
    #get the flowarea per peak and apply threshold
    peaks0, counts = np.unique(peaks, return_counts = True, axis = 0)
    #peaks0 = peaks0[counts>minflowlevel]
    #get the values per peak and apply threshold
    vals = image[tuple(peaks0.T)] #indexing wants a tuple for future safety
    good = np.logical_and(vals > bglevel, counts > minarea)
    peaks0 = peaks0[good]
    if return_diagnostics:
        res = [peaks0, counts[good], vals[good]]
    else:
        res = peaks0
    return res
def filterRmin(peaks, Rmin):
    #consider changing this function such that 
    #order = vals.argsort()[::-1]
    #ordered = peaks[order]
    goodpeaks = []
    for i, peak in enumerate(peaks):
        # need to exclude testing with self
        to_test = np.concatenate((peaks[:i], peaks[i+1:]))
        distances = np.linalg.norm(to_test - peak, axis = 1)
        if (distances > Rmin).all():
            goodpeaks.append(peak)
    print('filtered %i peaks' % (len(peaks) - len(goodpeaks)))
    return np.array(goodpeaks)
def plotpeaks(image, peaks, savedir = None, isshow = False):
    plt.figure(figsize = (10,10))
    plt.imshow(image)
    plt.colorbar()
    plt.scatter(peaks[:,1], peaks[:,0], s = 5, c = 'r')
    if savedir:
        dateTimeObj = datetime.now()
        timestamp = dateTimeObj.strftime("%Y-%b-%d_%H-%M-%S")
        plt.savefig(os.path.join(savedir, timestamp + '_overview.png'),\
                    bbox_inches = 'tight', dpi = 600)
    if isshow: plt.show()
def findPeaks(data, smooth_sigma = 1):
    """ finds three initial estimates for spot locations.
    First the image is smoothed.
    Second all local maxima are found
    Third the highest 3 maxima seperated by mindiff are selected
    returns: 3 x,y entries containing the peak coordinated
    """
    xlen, ylen = data.shape
    data = smooth_image(data, sigma = smooth_sigma)
    peaks = findMaxima(data)
    return peaks, data
import time
def filterPeaks(image, peaks, bglevel = 5, minarea = 50, Rmin = 10):
    """this function currently print the plot of the peaks, but it should print
    to the display of the GUI.
    Also it should save the selected peaks in a sensible way: both the image and
    the coordinates"""
    reportdir = r'D:\current data\peak_overviews' 
    goodpeaks, counts, vals = filterjunk(image, peaks, \
                                          minarea = minarea,\
                                          bglevel = bglevel, 
                                          return_diagnostics = True)
    ggoodpeaks = filterRmin(goodpeaks, Rmin)

    return ggoodpeaks, counts, vals