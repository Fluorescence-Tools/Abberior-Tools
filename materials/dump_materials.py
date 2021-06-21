# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:45:15 2021

@author: Abberior_admin
"""
#aid snippet to store some materials
#assumes that you've run the GUI first
#%%
import os
for i in range(4):
    path = os.path.join(r'C:\Users\Abberior_admin\Desktop\Abberior-Tools\materials', 'TSbead_Ch%i.txt'%i)
    np.savetxt(path, abberiorGUI.xy_data[i])
#%%
import specpy
import numpy as np
import matplotlib.pyplot as plt
im = specpy.Imspector()
stack = im.active_stack()
xy_stack = np.sum(stack.data(), axis = (0, 1))
path = os.path.join(r'C:\Users\Abberior_admin\Desktop\Abberior-Tools\materials', 'EGFPInCells_sparse_ch2.txt')
np.savetxt(path, xy_stack)
plt.imshow(xy_stack)