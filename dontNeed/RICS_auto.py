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
import time  

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

def powerseries(Exc_pv, STED_pv, Exc_laser, STED_laser, samplename):
    
    TEST =[]
    linesteps_number = [1,0,0,0,0,0,0,0]
    linesteps   = [True, False, False, False, False, False, False, False]
    laser_activ = [False, False, False, False, True, False, False, False] 
      
    for ii in range(len(Exc_pv)):
        laser = Exc_laser[ii]
        laser_STED = STED_laser[ii]
        if   laser == 485:laservalue = 0
        elif laser == 518:laservalue = 1
        elif laser == 595:laservalue = 2
        elif laser == 561:laservalue = 3
        elif laser == 640:laservalue = 4
        elif laser == 775:laservalue = 5
            
        if   laser_STED == 485:laser_STEDvalue = 0
        elif laser_STED == 518:laser_STEDvalue = 1
        elif laser_STED == 595:laser_STEDvalue = 2
        elif laser_STED == 561:laser_STEDvalue = 3
        elif laser_STED == 640:laser_STEDvalue = 4
        elif laser_STED == 775:laser_STEDvalue = 5
        
        im = specpy.Imspector()
        c=im.create_measurement()
        c.set_parameters('Pinhole/wheel_position', 2751)
        c.set_parameters('Pinhole/pinhole_size', 1e-04)
        c.set_parameters('ExpControl/scan/range/mode',784)#1328)
        c.set_parameters('ExpControl/scan/range/elliptical', False)
        c.set_parameters('OlympusIX/scanrange/z/z-stabilizer/enabled', False)
        c.set_parameters('ExpControl/scan/range/x/psz',10e-9)
        c.set_parameters('ExpControl/scan/range/y/psz',10e-9)
        c.set_parameters('ExpControl/scan/range/z/psz',1e-7)
        c.set_parameters('ExpControl/scan/range/x/len',0.00000256)
        c.set_parameters('ExpControl/scan/range/y/len',0.00000256)
        c.set_parameters('ExpControl/scan/range/z/len',0)
        c.set_parameters('ExpControl/scan/dwelltime', 0.000004)
        c.set_parameters('ExpControl/scan/range/t/res',100)
        c.set_parameters('ExpControl/gating/linesteps/laser_enabled',[True, True, True, True, True, True, False, False])
        c.set_parameters('{}{}{}'.format('ExpControl/lasers/power_calibrated/', laservalue,'/value/calibrated'), Exc_pv[ii])
        c.set_parameters('{}{}{}'.format('ExpControl/lasers/power_calibrated/', laser_STEDvalue,'/value/calibrated'),STED_pv[ii])
        c.set_parameters('ExpControl/gating/linesteps/step_values',linesteps_number)
        c.set_parameters('ExpControl/gating/linesteps/steps_active', linesteps)
        c.set_parameters('ExpControl/gating/linesteps/laser_on',
                         [[False,False, False,False, True, True, False, False],
                         [False,False, False,False,False, False, False, False],
                         [False,False, False,False,False, False, False, False],
                         [False,False, False,False,False, False, False, False],
                         [False,False, False,False,False, False, False, False],
                         [False,False, False,False,False, False, False, False],
                         [False,False, False,False,False, False, False, False],
                         [False,False, False,False,False, False, False, False]])
        
        
    
        
        M_name= im.measurement_names()[ii]
        M_obj= im.measurement(M_name)
        M_act= im.activate(M_obj)
        
        im.run(M_obj)
            
        #time.sleep(5) # timewait set to 5 sec
        meas = im.active_measurement()

        stk_names = meas.stack_names()
        stk_g1 = meas.stack(stk_names[0])
        stk_g2 = meas.stack(stk_names[2])
        stk_r1 = meas.stack(stk_names[1])
        stk_r2 = meas.stack(stk_names[3])
        pix_data_green = stk_g1.data() + stk_g2.data()
        pix_data_red = stk_r1.data() + stk_r2.data()
        TEST.append(pix_data_red)
        #fig = pp.figure()
        #pp.imshow(np.mean(pix_data_red, axis=0)[0], cmap = "Greens")
        savepath = r'D:/current data'
        Excname = '_meas_Exc_'
        STEDname = '_STED_'
        path = '{}{}{}{}{}{}{}{}{}{}{}{}{}{}'.format(savepath,'/', samplename, Excname, laser,'_',Exc_pv[ii] , STEDname, laser_STED, '_',STED_pv[ii], '_',ii,'.msr')
        M_obj.save_as(path)
        
    return TEST

#Exc_laser= [640,640,640,640,640,640,640,640,640,640]
#Exc_pv=    [20,20,20,20,20,20,20,20,20,20]
#STED_laser=[775,775,775,775,775,775,775,775,775,775]
#STED_pv=   [0,2,5,10,20,30,40,50,60,70]

Exc_laser=[640,640,640]
Exc_pv= [10,10,10]
STED_laser=[775,775,775]
STED_pv=[0,0,0]

#Exc_laser=[640]
#Exc_pv= [10]
#STED_laser=[775]
#STED_pv=[0]

sample = 'Atto647N_PAAM_025perc_'
R = powerseries(Exc_pv= Exc_pv, STED_pv=STED_pv, Exc_laser=Exc_laser, STED_laser= STED_laser, samplename = sample)

W0 =[]
Ddiff = []
Wz = []
N = []
for frame in R:
    
    imarray = frame[0]
    
    
    G_mean = []

    for i,frame in enumerate(imarray):
        for line in frame:
            G = correlate(line, line)
            G_mean.append(G)
    
    G_ = np.mean(G_mean, axis = 0)[0:]
   
  
    
    
         #               N,           D,         w0,       wz,       tp [s],    px [µm],     offset 
    Init_value = [       1,         300,       0.27,      1.87,     0.000004,     0.01,        0]
    Bound_min  = [       0,           0,          0,        0,            0,        0.0,      -1]
    Bound_max  = [      200,         400,          10,      1000,            2,        1.2,       1]
    Value_fit  = [    True,       True,       False,     False,        False,      False,    True]
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
    fig = pp.figure()
    pp.plot(x, y, label='data');
    pp.plot(x, result.best_fit, 'r-', label='best fit');
    pp.xlabel('pixel')
    pp.ylabel('G_Corr. ampl.')
    pp.xlim(0,np.max(x))
    pp.ylim(np.min(y),np.max(y))
    W0.append(result.params['w0'].value)
    Ddiff.append(result.params['D'].value)
    Wz.append(result.params['wz'].value)
    N.append(result.params['N'].value)
    print(result.fit_report());

fig=pp.figure(figsize =(15,10))
pp.subplot(551)
pp.plot(STED_pv,W0)
pp.xlabel('STED power [%]')
pp.ylabel('W0 [um]')
pp.subplot(552)
pp.plot(STED_pv,Wz)
pp.xlabel('STED power [%]')
pp.ylabel('Wz [um]')
pp.subplot(553)
pp.plot(STED_pv,Ddiff)
pp.xlabel('STED power [%]')
pp.ylabel('Ddiff [um2/s]')
pp.subplot(554)
pp.plot(STED_pv,W0/np.max(W0))
pp.xlabel('STED power [%]')
pp.ylabel('Weff/Wconf')
pp.subplot(555)
pp.plot(STED_pv,N)
pp.xlabel('STED power [%]')
pp.ylabel('N')
print('Wconf:', np.max(W0), '  WSTED_min:', np.min(W0))









