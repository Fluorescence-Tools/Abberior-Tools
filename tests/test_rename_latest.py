# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:59:49 2022

@author: Abberior_admin
"""
import os
import glob
wdir = r"D:\current data"
filelist = glob.glob(wdir + r'\*.ptu')
print(filelist)
lastfile = max(filelist, key = os.path.getctime)
print(lastfile)