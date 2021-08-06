# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 13:39:25 2021

@author: Abberior_admin
"""
import specpy
from pprint import pprint
im = specpy.Imspector()
#%%
msr = im.active_measurement()
config = msr.active_configuration()
#%%
#get coarse offset
offset = msr.parameters('ExpControl/scan/range/offsets/coarse/y/g_off')
#change course offset
offset = offset + 20e-6
# set coarse offset
config.set_parameters('ExpControl/scan/range/offsets/coarse/y/g_off', offset) 
#seems to work ok.