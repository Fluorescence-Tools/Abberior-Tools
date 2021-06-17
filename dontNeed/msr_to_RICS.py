

from specpy import *
import specpy
import matplotlib.pyplot as pp
import numpy as np
import PIL.Image as Image
import time
import numpy as np
import matplotlib.pyplot as pp
import lmfit
from scipy.optimize import minimize
from lmfit import Model
import pandas as pd
import numpy.fft
from PIL import Image
from skimage import io
import specpy
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

def RICS(x, N, D, w0, wz, tp, px, offset):
    Gdif = 1/N * (1 + 4*D*tp*x/(w0**2))**(-1) * (1 + 4*D*tp*x/(wz**2))**(-0.5)
    Gmotion = np.exp(-(px**2*x**2)/(w0**2 + 4*D*tp*x))
    
    return  Gdif * Gmotion + offset

def correlate(a, b):
    """Return circular correlation of two arrays using DFT."""
    size = a.size

    # forward DFT
    a = np.fft.rfft(a)
    b = np.fft.rfft(b)
    # multiply by complex conjugate
    c = a.conj() * b
    # reverse DFT
    c = np.fft.irfft(c)
    
    # positive delays only
    c = c[:size // 2]
        
    # normalize with the averages of a and b
    #   c is already normalized by size
    #   the 0th value of the DFT contains the sum of the signal
    c /= a[0].real * b[0].real / size
    c -= 1.0
    
    return c

# Connect to Imspector
im = specpy.Imspector()
M_name_1= im.measurement_names()
last =len(M_name_1)
M_name= im.measurement_names()[last-1]
M_obj= im.measurement(M_name)
M_act= im.activate(M_obj)


                            
meas = im.active_measurement()
stk_names = meas.stack_names()
stk_g1 = meas.stack(stk_names[0])
stk_g2 = meas.stack(stk_names[2])
stk_r1 = meas.stack(stk_names[1])
stk_r2 = meas.stack(stk_names[3])
pix_data_green = stk_g1.data() + stk_g2.data()
pix_data_red = stk_r1.data() + stk_r2.data()

fig =pp.figure()
pp.subplot(221)
pp.imshow(np.mean(pix_data_green, axis=0)[0], cmap = "Greens")
pp.subplot(222)
pp.imshow(np.mean(pix_data_red, axis=0)[0], cmap = "Reds")
stk_names = meas.stack_names()
print(stk_names[0])
stk = meas.stack(stk_names[0])

imarray = pix_data_red[0]


G_mean = []
dwelltime = meas.parameters('ExpControl/scan/dwelltime')
pixelsize = meas.parameters('ExpControl/scan/range/x/psz')*1000000
for i,frame in enumerate(imarray):
    for line in frame:
        G = correlate(line, line)
        G_mean.append(G)

G_ = np.mean(G_mean, axis = 0)[0:]
time = np.arange(0,len(G_))*dwelltime*1000
fig = pp.figure()
#pp.plot(time, G_)



     #               N,           D,         w0,       wz,       tp [s],    px [µm],     offset 
Init_value = [       1,         300,       0.244,     1.93,     dwelltime,     pixelsize,        0]
Bound_min  = [       0,           0,          0,        0,            0,        0.0,      -1]
Bound_max  = [      200,         600,          1,      20,       200,       200,       1]
Value_fit  = [    True,       True,     True,     True,       False,      False,    True]
Value_exp =  [    None,        None,       None,     None,         None,       None,     None] 
 

x = np.arange(0,len(G_[1:]))
y = G_[1:]
model = Model(RICS)
params = model.make_params()
P = model.param_names
for i in range(len(P)):
    params[P[i]].value= Init_value[i]
    params[P[i]].min= Bound_min[i]
    params[P[i]].max= Bound_max[i]
    params[P[i]].vary= Value_fit[i]
    params[P[i]].expr= Value_exp[i]

result = model.fit(y, params, x=x, weight = 1)
pp.plot(x, y, 'blue', label='data');
pp.plot(x, result.best_fit, 'r-', label='best fit');
pp.xlabel('pixel')
pp.ylabel('G_Corr. ampl.')
pp.xlim(0,np.max(x))
pp.ylim(np.min(y),np.max(y))
print(result.fit_report());
