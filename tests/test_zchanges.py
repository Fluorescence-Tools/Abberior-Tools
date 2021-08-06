# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 17:26:27 2021

@author: Abberior_admin
"""
import specpy
from pprint import pprint
im = specpy.Imspector()
#%%
#msr = im.create_measurement()
msr = im.active_measurement()
msr.activate(msr.configuration(msr.configuration_names()[0]))
config = msr.active_configuration()
for config_name in msr.configuration_names()[1:]:
    msr.remove(msr.configuration(config_name))
#%%
config = msr.active_configuration()
config.set_parameters('ExpControl/scan/range/offsets/coarse/y/g_off', 1e-5) 
#%%
config = msr.clone(msr.active_configuration())        
config.set_parameters('ExpControl/lasers/power_calibrated/0/value/calibrated', 5)
config.set_parameters('ExpControl/gating/linesteps/laser_enabled',[True, False, False, True, False, False, False, False])
config.set_parameters('ExpControl/scan/range/mode',784)
stream = 'HydraHarp'
config.set_parameters('ExpControl/gating/tcspc/channels/0/mode', 0) 
config.set_parameters('ExpControl/gating/tcspc/channels/0/stream', stream)
config.set_parameters('ExpControl/gating/tcspc/channels/1/mode', 0)
config.set_parameters('ExpControl/gating/tcspc/channels/1/stream', stream)
config.set_parameters('ExpControl/gating/tcspc/channels/2/mode', 0)
config.set_parameters('ExpControl/gating/tcspc/channels/2/stream', stream)
config.set_parameters('ExpControl/gating/tcspc/channels/3/mode', 0)
config.set_parameters('ExpControl/gating/tcspc/channels/3/stream', stream)
config.set_parameters('HydraHarp/data/streaming/enable', True)
config.set_parameters('HydraHarp/is_active', True) #not sure what this does
config.set_parameters('ExpControl/scan/range/x/off', 1e-6 )
config.set_parameters('ExpControl/scan/range/y/off', -10e-6 )
im.run(msr)