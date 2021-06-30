# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 17:05:12 2021

@author: Abberior_admin
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r'\\192.168.169.11\data\User\vanderVoortN\FRC\Code')
import findPeaksLib
#%%
wdir = r'C:\Users\Abberior_admin\Desktop\Abberior-Tools\materials'
TSbeads = [np.genfromtxt( os.path.join(wdir, 'TSbead_Ch%i.txt' % i )) for i in [1,3]]
TSbeads = TSbeads[0] + TSbeads[1]
EGFPthick = [np.genfromtxt( os.path.join(wdir, 'EGFPInCells_Ch%i.txt' % i )) for i in [0,2]]
EGFPthick = EGFPthick[0] + EGFPthick[1]
EGFPsparse = [np.genfromtxt( os.path.join(wdir, 'EGFPInCells_sparse_Ch%i.txt' % i )) for i in [0,2]]
EGFPsparse = EGFPsparse[0] + EGFPsparse[1]
#%%
def filterPeaks(image, peaks, minflowlevel = 20, bglevel = 2,
                return_diagnostics = False):
    #get the flowarea per peak and apply threshold
    peaks0, counts = np.unique(peaks, return_counts = True, axis = 0)
    #peaks0 = peaks0[counts>minflowlevel]
    #get the values per peak and apply threshold
    vals = image[tuple(peaks0.T)] #indexing wants a tuple for future safety
    good = np.logical_and(vals > bglevel, counts > minflowlevel)
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
        print(i)
        to_test = np.concatenate((peaks[:i], peaks[i+1:]))
        print(to_test.shape)
        distances = np.linalg.norm(to_test - peak, axis = 1)
        if (distances > Rmin).all():
            goodpeaks.append(peak)
    print('filtered %i peaks' % (len(peaks) - len(goodpeaks)))
    return np.array(goodpeaks)
def plotpeaks(image, peaks):
    plt.figure(figsize = (10,10))
    plt.imshow(image)
    plt.colorbar()
    plt.scatter(peaks[:,1], peaks[:,0], s = 5, c = 'r')
def findPeaks(data, smooth_sigma = 1):
    """ finds three initial estimates for spot locations.
    First the image is smoothed.
    Second all local maxima are found
    Third the highest 3 maxima seperated by mindiff are selected
    returns: 3 x,y entries containing the peak coordinated
    """
    xlen, ylen = data.shape
    data = findPeaksLib.smooth_image(data, sigma = smooth_sigma)
    peaks = findPeaksLib.findMaxima(data)
    return peaks, data
#%%
reportdir = r'\\192.168.169.11\data\User\vanderVoortN\04_reports'
Rmin = 10,
minflowlevel = 50
bglevel = 5
#peaks, image = findPeaks(EGFPthick) # slow step
goodpeaks, counts, vals = filterPeaks(image, peaks, \
                                      minflowlevel = minflowlevel,\
                                      bglevel = bglevel, 
                                      return_diagnostics = True)
ggoodpeaks = filterRmin(goodpeaks, Rmin)
plotpeaks(image, ggoodpeaks)
plt.savefig(os.path.join(reportdir, 'minarea50minbg5Rmin10.png'), bbox_inches = 'tight', dpi = 600)


#%%

plt.figure(figsize = (10,10))
plt.imshow(data)
plt.title('smoothed EGFP on cell surface', fontsize = 40)
plt.xlabel('20 micro meter', fontsize = 20)
plt.ylabel('20 micro meter', fontsize = 20)
plt.colorbar()
plt.savefig(os.path.join(reportdir, 'smoothed data.png'), bbox_inches = 'tight', dpi = 600)
#%%
plt.hist(counts, bins = 100)
plt.xlabel('peak surface area (pixels)', fontsize = 20)
plt.ylabel('occurence')
ax = plt.gca()
ax.set_yscale('log')
plt.xlim(0,250)
plt.savefig(os.path.join(reportdir, 'hist_surfaceArea_area50.png'), bbox_inches = 'tight', dpi = 600)
#%%
plt.hist(vals, bins = 100)
plt.xlabel('peak maximum (photons)', fontsize = 20)
plt.ylabel('occurence')
ax = plt.gca()
ax.set_yscale('log')
plt.xlim(0,16)
plt.savefig(os.path.join(reportdir, 'hist_maximum_bg5.png'), bbox_inches = 'tight', dpi = 600)
