# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 18:13:18 2021

@author: Abberior_admin
"""
import os
import plotTrace
import numpy as np
#%%
#this time stamp is generating the same time stamp for all files in a folder, whereas it shouldn't
wd = r'D:\current data\2021-Jul-02-17-30-10Overview_0.00_numberSPOTS_7'
# for file in os.listdir(wd):
#     ffile = os.path.join(wd, file)
#     print(os.path.getmtime(ffile))
def getLastModified(wd):
    """gets the last modified file in folder"""
    files = os.listdir(wd)
    modtimes = [os.path.getmtime(os.path.join(wd, file)) for file in files]
    lastfile = os.path.join(wd, files[np.argmax(modtimes)])
    print(lastfile)
    assert lastfile[-4:] == '.ptu', 'cannot plot trace, not a ptu file'
    return lastfile.encode()

lastfile = plotTrace.getLastModified(wd)
channels = plotTrace.getTraces(lastfile, [0,2])
binneddata = plotTrace.plotTrace(channels, (0,10), lastfile, outname = 'inferred')