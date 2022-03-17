# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:44:35 2022

@author: Abberior_admin
learnings: using a function to return msr does not work, inherent in specpy 
package, have to call functions within the same scope.
"""
import specpy
from pprint import pprint
im = specpy.Imspector()
def try_get_active_measurement():
    """gets the active measurement, when none exists, creates and returns one"""
    im = specpy.Imspector()
    try:
        msr = im.active_measurement()
    except RuntimeError:
        msr = im.create_measurement()
    return msr
#%%
msr = im.active_measurement()
config = msr.active_configuration()
#%%
msr=im.create_measurement()
#%%
msr = try_get_active_measurement()
#%%
im.run(msr)