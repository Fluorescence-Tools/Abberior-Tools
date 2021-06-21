# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 14:10:34 2021

@author: Abberior_admin
"""
import tkinter as tk
import matplotlib.cm as cm
from PIL import Image, ImageTk
import specpy
import sys
sys.path.append(r'C:\Users\Abberior_admin\Desktop\Abberior-Tools')
from FUNCTION import setDefaultMeasurementSettings
import matplotlib.pyplot as plt
import numpy as np
#%%

im = specpy.Imspector()
meas=im.create_measurement()
setDefaultMeasurementSettings(meas)
meas.set_parameters('ExpControl/scan/range/mode', 528)
im.run(meas)
#this sums over the t and z dimensions and puts all channels in a list
#%%
xy_data = [np.sum(meas.stack(i).data(), axis = (0, 1)) for i in range(4)]
c1 = (xy_data[0] + xy_data[2])

#%%
master = tk.Tk()
frame_top = tk.Frame(master, width = 400, height = 400)
frame_top.pack()
photo = cm.hot(c1.astype(np.float))#needs floats
photo = np.uint8(photo * 255)
photo = Image.fromarray(photo)
photo = photo.resize((400, 400), Image.ANTIALIAS)
photoTk = ImageTk.PhotoImage(photo, master = master)
label = tk.Label(frame_top,image=photoTk)
label.pack()
master.mainloop()
#%%
from pprint import pprint
meas = im.active_measurement()
pprint(meas.parameters())
xy_data = [np.sum(meas.stack(i).data(), axis = (0, 1)) for i in range(4)]
