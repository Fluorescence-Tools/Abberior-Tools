# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:54:10 2021

@author: Abberior_admin
"""
import specpy
from pprint import pprint
im = specpy.Imspector()
#%%
meas = im.active_measurement()
pprint(meas.parameters("ExpControl"))
#%%
#meas = im.create_measurement()
#d = im.active_measurement()
#msr= im.measurement(d.name())
#config.clone(d.active_configuration())
msr = im.active_measurement()
config = msr.clone(msr.active_configuration())
#config.activate(config.configuration(0))          
config.set_parameters('ExpControl/lasers/power_calibrated/0/value/calibrated', 5)
config.set_parameters('ExpControl/gating/linesteps/laser_enabled',[True, False, False, True, False, False, False, False])
config.set_parameters('ExpControl/scan/range/mode',1363)
config.set_parameters('ExpControl/scan/range/t/len', 2)
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

#%%
config = msr.active_configuration()
config.set_parameters('ExpControl/scan/range/z/off', 1e-5)
im.run(msr)