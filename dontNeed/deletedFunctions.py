# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 18:21:02 2021

@author: Nicolaas van der Voort
Functions originally written by Jan Budde
AFAIK functions were never used, not sure if they were tested    
"""
import specpy
import numpy as np
import matplotlib.pyplot as pp
import os
from scipy import ndimage
from skimage.feature import peak_local_max
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg#, NavigationToolbar2TkAgg
from skimage.morphology import watershed
import imutils
import cv2
from lmfit import Model
import shutil
import matplotlib.cm as cm
import tkinter as tk
import time
from datetime import datetime
import threading  


#imports used by Findpeak module
from sklearn.neighbors import NearestNeighbors
import math
import scipy
from astropy.stats import RipleysKEstimator
from scipy.ndimage.filters import gaussian_filter

def pinholeseries(pinholevector, laser):

    number_frames = 21
    LP485 = 80
    pixelsize = 100
    Roisize = 5
    #Dwelltime = 5*1e-06  
    
    #dwellvector  = [  1,  5, 200]#, 10, 50, 100, 200, 1000]
    #nframesvector= [ 50, 50,  50]#, 51, 51, 100, 51, 51]
    
    
    dwellvector  = [  1,  1.5,  2,  2.5,  3,  4,  5, 10,  20,  40  ]#60,  80,  100, 200, 500, 1000[::-1]#,  1000][::-1]#, 10, 50, 100, 200, 1000]
    nframesvector= [ 50,   50, 50,   50, 50, 50, 50, 50,  50,  50 ]#50,  50,   50,  50,  20,   20]#[::-1]#   2][::-1]#, 51, 51, 100, 51, 51]
    
    #dwellvector  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200, 500, 1000 ]#, 10, 50, 100, 200, 1000]
    #nframesvector= [50,50,50,50,50,50, 50, 50, 50, 50 ,3,3,3,3,1,1]#, 51, 51, 100, 51, 51]
    
    #dwellvector  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 200, 500 ]#, 10, 50, 100, 200, 1000]
    #nframesvector= [101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 51, 21 , 21]#, 51, 51, 100, 51, 51]
    
    x_pixelsize = float(pixelsize)*1e-09        # in meter
    y_pixelsize = float(pixelsize)*1e-09        # in meter
    z_pixelsize = float(pixelsize)*1e-09        # in meter
    roi_size =    float(Roisize)*1e-06            # in meter 
    #Dwelltime= float(dwelltime)*1e-06         # in seconds  
    
    
    z_position =  5e-08 #float(pixelsize)*1e-09         # in meter
    modelinesteps= True      #Type True for linesteps
    number_linesteps = 1     #Type the number of linesteps
    xyt_mode = 784           #Type 785 for xyt mode
    
    Streaming_HydraHarp = True
    detector1= True
    detector2= True
    detector3= True
    detector4= True
    
    Activate485 = True                    
    Activate518 = False  
    Activate561 = False  
    Activate640 = False 
    Activate595 = False   
    Activate775 = False   
    
    Activate485_02 = True                   
    Activate518_02 = False
    Activate561_02 = False   
    Activate640_02 = False   
    Activate595_02 = False   
    Activate775_02 = False
    
    Ch1_mode= 0                # type 0 for counter and 1 for flim
    Ch1_stream= 'HydraHarp'         # type hydraharp or fpga for internal
    Ch2_mode= 0          
    Ch2_stream= 'HydraHarp'
    Ch3_mode= 0         
    Ch3_stream= 'HydraHarp'
    Ch4_mode= 0         
    Ch4_stream= 'HydraHarp'
    
    linesteps = [False, False, False, False, False, False, False, False]
    for i in range(number_linesteps):
        linesteps[i] = True
    
    Cntsrate =[]
    Tpx = []
    if a==0:
        import specpy
        import time
        im = specpy.Imspector()
        for ii, dwell in enumerate(dwellvector):
            c=im.create_measurement()   
            d = im.active_measurement()   
            M_obj= im.measurement(d.name())
            x_off = d.parameters('ExpControl/scan/range/offsets/coarse/x/g_off') # current stage position in x [m]
            x_add = x_off + 2*roi_size*(ii+1)  ### initial position 
            #c.set_parameters('ExpControl/scan/range/offsets/coarse/x/g_off', x_add)
            c.set_parameters('ExpControl/scan/range/x/psz',x_pixelsize)
            c.set_parameters('ExpControl/scan/range/y/psz',y_pixelsize)
            c.set_parameters('ExpControl/scan/range/z/psz',z_pixelsize)
            c.set_parameters('ExpControl/scan/range/x/off', 0 )#+ ROI_offset)
            c.set_parameters('ExpControl/scan/range/y/off', 0)#+ ROI_offset)
            c.set_parameters('ExpControl/scan/range/z/off', 1e-15)#z_position*z_pixelsize)
            c.set_parameters('ExpControl/scan/range/x/len',roi_size)
            c.set_parameters('ExpControl/scan/range/y/len',roi_size)
            c.set_parameters('ExpControl/scan/range/z/len',roi_size)
            c.set_parameters('ExpControl/scan/dwelltime', dwell*1e-06 )
            c.set_parameters('HydraHarp/data/streaming/enable', Streaming_HydraHarp)
            c.set_parameters('HydraHarp/is_active', Streaming_HydraHarp)
            c.set_parameters('ExpControl/gating/tcspc/channels/0/mode', Ch1_mode)
            c.set_parameters('ExpControl/gating/tcspc/channels/0/stream', Ch1_stream)
            c.set_parameters('ExpControl/gating/tcspc/channels/1/mode', Ch2_mode)
            c.set_parameters('ExpControl/gating/tcspc/channels/1/stream', Ch2_stream)
            c.set_parameters('ExpControl/gating/tcspc/channels/2/mode', Ch3_mode)
            c.set_parameters('ExpControl/gating/tcspc/channels/2/stream', Ch3_stream)
            c.set_parameters('ExpControl/gating/tcspc/channels/3/mode', Ch4_mode)
            c.set_parameters('ExpControl/gating/tcspc/channels/3/stream', Ch4_stream)
            c.set_parameters('ExpControl/gating/linesteps/on', modelinesteps)
            c.set_parameters('ExpControl/gating/linesteps/steps_active', linesteps)
            c.set_parameters('ExpControl/gating/linesteps/step_values',[1,0,0,0,0,0,0,0])
            c.set_parameters('ExpControl/scan/range/mode',xyt_mode)
            c.set_parameters('ExpControl/lasers/power_calibrated/0/value/calibrated', float(LP485))
            c.set_parameters('ExpControl/gating/pulses/pulse_chan/delay',[0.0, 0.0, 0.0, 0.0])
            c.set_parameters('ExpControl/gating/linesteps/laser_enabled',[True, False, False, False, False, False, False, False])
            c.set_parameters('ExpControl/gating/linesteps/laser_on',[[Activate485, Activate518, Activate595, Activate561, Activate640, Activate775, False, False],
             [Activate485_02, Activate518_02, Activate595_02, Activate561_02, Activate640_02, Activate775_02, False, False],
             [Activate485, Activate518, Activate595, Activate561, Activate640, Activate775, False, False],
             [Activate485, Activate518, Activate595, Activate561, Activate640, Activate775, False, False],
             [Activate485, Activate518, Activate595, Activate561, Activate640, Activate775, False, False],
             [Activate485, Activate518, Activate595, Activate561, Activate640, Activate775, False, False],
             [Activate485, Activate518, Activate595, Activate561, Activate640, Activate775, False, False],
             [Activate485, Activate518, Activate595, Activate561, Activate640, Activate775, False, False]])
            c.set_parameters('ExpControl/scan/range/t/res',nframesvector[ii])
            c.set_parameters('ExpControl/gating/linesteps/chans_enabled',[detector1,detector2,detector3,detector4])
            c.set_parameters('ExpControl/scan/detsel/detsel',['APD1', 'APD2', 'APD3', 'APD4'])
            c.set_parameters('ExpControl/gating/linesteps/chans_on', [[True, True, True, True],
             [True, True, True, True],
             [False, False, False, False],
             [False, False, False, False],
             [False, False, False, False],
             [False, False, False, False],
             [False, False, False, False],
             [False, False, False, False]])
         
                  #### # c.set_parameters('{}{}{}'.format('ExpControl/lasers/pinholeseries'), ii)
                   
            time.sleep(1)
            im.run(M_obj)
            
            meas = im.active_measurement()
            stk_names = meas.stack_names()
            stk_g1 = meas.stack(stk_names[0])
            stk_g2 = meas.stack(stk_names[2])
            stk_r1 = meas.stack(stk_names[1])
            stk_r2 = meas.stack(stk_names[3])
            pix_data_green = stk_g1.data() + stk_g2.data()
            SUMME = np.nansum(pix_data_green)/(pix_data_green.shape[1]*pix_data_green.shape[2]*pix_data_green.shape[3]*dwell*1e-06)
            tpx_ = 0.000000237*dwell/x_pixelsize  #return time per pixel in µs
            Cntsrate.append(SUMME)
            print(pix_data_green.shape, np.nansum(pix_data_green), SUMME)
            Tpx.append(tpx_)
        
             # timewait set to 5 sec
            time.sleep(1)
    else:
        print('pinholeseries not available! No connection')
    
    files = os.listdir('D:/current data/')
    files_ptu = [i for i in files if i.endswith('.ptu')]
                
    n_files_ptu = len(files_ptu)
                    
    for ii in range(n_files_ptu):
        os.rename('{}{}'.format('D:/current data/',files_ptu[ii]), '{}{}{}{}'.format('D:/current data/', 'dwelltime_', dwellvector[ii], '.ptu'))
                
    fig =pp.figure()  
    pp.plot( Tpx, Cntsrate, 'o')
    pp.ylabel('Countsrate Hz')
    pp.xlabel('tpx [us]')
    pp.xscale('log')

def powerseries(laser_value_01, laser_STEDvalue_01,laser_01,laser_STED_01):
    import time
    import specpy
    
    powervector = int(laser_value_01.get())
    powervector_STED = [int(s) for s in laser_STEDvalue_01.get().split(',')]
    laser = int(laser_01.get()) 
    laser_STED = int(laser_STED_01.get())
        
    if a==0:
        im = specpy.Imspector() 


        if laser == 485:laservalue = 0
        elif laser == 518:laservalue = 1
        elif laser == 595:laservalue = 2
        elif laser == 561:laservalue = 3
        elif laser == 640:laservalue = 4
        elif laser == 775:laservalue = 5
        
        if laser_STED == 485:laser_STEDvalue = 0
        elif laser_STED == 518:laser_STEDvalue = 1
        elif laser_STED == 595:laser_STEDvalue = 2
        elif laser_STED == 561:laser_STEDvalue = 3
        elif laser_STED == 640:laser_STEDvalue = 4
        elif laser_STED == 775:laser_STEDvalue = 5
      
              
        for ii in powervector_STED:
            c=im.create_measurement()

            #print(ii)
            c.set_parameters('{}{}{}'.format('ExpControl/lasers/power_calibrated/', laservalue,'/value/calibrated'), powervector)
            c.set_parameters('{}{}{}'.format('ExpControl/lasers/power_calibrated/', laser_STEDvalue,'/value/calibrated'), ii)
            M_name= im.measurement_names()[ii + 1]
            M_obj= im.measurement(M_name)
            im.run(M_obj)
            
            time.sleep(2) # timewait set to 5 sec
    else:
        print('Powerseries not available! No connection')
        
# =============================================================================
# def Findpeak(self):
#     #function uses slightly different names, I don't want to invest right now
#     #in changing them all
#     path = self.dataout
#     roi_size = self.ROIsize
#     T = self.T
#     circle = self.circle.get()
#     foldername = self.foldername
#     pixelsize_global = self.pxsize_overview_value
#     # global x_new
#     # global y_new
#     # global R
#     # global scale_val
#     # global aa
#     # global number_peaks_new 
#     # global time_wait_Multirun
#     
#     # global x_transfer
#     # global y_transfer
#     # global CO
# 
#     threshold = self.scale_01.get()      # define the lower threshold value
#     print("threshold value is %.1f" % threshold)
#     R_min = self.scale_02.get()          # additional distance threshold
#     R_max = self.scale_03.get()
#     #exc_border_peaks = 1 # type 0 for activate for border peaks 
#     
#     # if data is Red, probably want to change
#     data = self.xy_data[1] + self.xy_data[2] 
#     #I guess this was a way to retrieve the data
# # =============================================================================
# #     if a==0:
# #         path = 'D:/current data/'
# #         data = Image.open('{}{}'.format(path,'Overview_image.tiff'))
# #         
# #     else:
# #         if circle ==0:
# #             path = path
# #             data = Image.open('{}{}'.format(path,'Overview_image.tiff'))
# #         elif circle == 1:
# #             path = path
# #             data = Image.open('{}{}'.format(path,'Overview_image.tiff'))
# # =============================================================================
# 
#             
#     #what does this do?
#     #if self.circle ==0:   
#     data_sz = data.size
#     zz = np.zeros(data_sz)
#     aa = self.ROIsize
#     testfolder = os.path.join(path, self.foldername)
#     #why is this needed?
#     if os.path.exists(testfolder) == False:
#         os.makedirs(testfolder)
#         os.makedirs(os.path.join(path, 'test_images.tiff'))
#         
# ############################################
# #Calculate parameters    
# ############################################
# 
#     imarray = np.array(data)
#     imarray_thres = np.where(imarray > threshold,  imarray, 0)
# 
#     array_size_x = imarray_thres.shape[1]
#     
#     v1_x = int(array_size_x/2)
#     v1_y = int(array_size_x/2)
#     
#     ##############################################
#     # Find peaks in images:   
#     coordinates = peak_local_max(imarray_thres, min_distance= R_min, indices=True) #exclude_border=exc_border_peaks)
#     y_coord = coordinates[:, 0]
#     x_coord = coordinates[:, 1]
#      
#     if len(coordinates)<= 1:
#         CO=1
#         print('NearestNeighbors failed: coordinates is <= 1')
#         x_transfer = []
#         y_transfer = []
#         np.savetxt('{}{}{}'.format(path,'x_transfer','.dat'), x_transfer)       
#         np.savetxt('{}{}{}'.format(path,'y_transfer','.dat'), y_transfer) 
#         
#     else:
#         CO= 0
#         print('NearestNeighbors okay')
#         nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(coordinates)
#         distances, indices = nbrs.kneighbors(coordinates) 
#             
#         number_peaks = int(coordinates.size/ 2)
#         t_x= np.zeros(number_peaks)
#         t_y= np.zeros(number_peaks)
#         R_new = np.zeros(number_peaks)
#         
#         for i in range(number_peaks):
#             for ii in range(number_peaks):
#                 R = math.sqrt((coordinates[i][0]-coordinates[ii][0])**2+(coordinates[i][1]-coordinates[ii][1])**2)
#                 
#                 if R < R_max and R > 0:#R_min:
#                     #print ('peaks',i,ii,'x1_coord',coordinates[i][0],'y1_coord',coordinates[i][1],'x2_coord',coordinates[ii][0],'y2_coord',coordinates[ii][1],'Distance R',R,'below', R_min  )
#                     #print (coordinates[i][0],coordinates[i][1])
#                     t_y[i] = coordinates[i][0]
#                     t_x[i] = coordinates[i][1]
#                     R_new[i] = R
#   
#         R_new_thres = R_new[np.where(R_new>0)]
#     
#         ###################################
#         # Libary import
#             
#         # Do you want to simulate a Poisson distribution of spots [own_data = 0] or use own data [own_data = 1]????
#         own_data = 1
#              
#         ###############  Window parameters  ##########################################
#         pixelsize= 1 #[nm]
#         
#         xMin=0
#         xMax=data.size[0]; # pixelnumber in x
#         yMin=0
#         yMax=data.size[1]; # pixelnumber in y
#          
#         #Poisson process parameters simulation
#         lambda0=5; #intensity (ie mean density) of the Poisson process
#          
#         ###############################################################################
#         if own_data ==0:
#             print('Poisson distribution of spots')
#             #Simulate Poisson point process
#             xDelta=xMax-xMin;yDelta=yMax-yMin; #rectangle dimensions
#             areaTotal=xDelta*yDelta; #total area
#             numbPoints = scipy.stats.poisson( lambda0*areaTotal ).rvs()#Poisson number of points
#             x_Poisson = xDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+xMin#x coordinates of Poisson points
#             y_Poisson = yDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+yMin#y coordinates of Poisson points
#             xx = x_Poisson
#             yy = y_Poisson
#             
#         else:
#             print('Using real data for statistics')
#             xDelta=xMax-xMin;yDelta=yMax-yMin; #rectangle dimensions
#             areaTotal=xDelta*yDelta; #total area
#                 
#        
#         z = coordinates
#         #np.savetxt('C:/Users/buddeja/Desktop/test/z_GUI.txt', z, delimiter='\t')   # X is an array
#         #print(z.shape)
#         Kest = RipleysKEstimator(area=areaTotal, x_max=xMax, y_max=yMax, x_min=xMin, y_min=yMin)
#         r = np.linspace(0, xMax, 1000)
#         ###########################################################################
#     
#     
#         f1 = pp.figure(figsize=(3,3), dpi=100, edgecolor='k',facecolor = 'r')
#         LABELSIZE = 7
#         a = f1.add_subplot(211)
#         a.hist(distances[:,1], bins=20,color='blue', label='1st neighbor',alpha=0.2)
#         a.hist(distances[:,2], bins=20,color='red', label='2nd neighbor',alpha=0.2)
#         pp.xlabel('Nearst neighbor distance [a.u]', fontsize = LABELSIZE, fontweight='bold')
#         pp.ylabel('counts', fontsize = LABELSIZE, fontweight='bold')
#         a.xaxis.set_tick_params(labelsize=LABELSIZE)
#         a.yaxis.set_tick_params(labelsize=LABELSIZE)
#         pp.legend(fontsize = LABELSIZE)
#     
#         
#         y = r*0
#         b = f1.add_subplot(212)
#         #b.plot(r*pixelsize, Kest(data=z, radii=r, mode='translation'), color='black',label=r'$K_{trans}$')
#         b.plot(r*pixelsize, Kest.Hfunction(data=z, radii=r, mode='translation'), color='black',label=r'$K_{trans}$')
#         b.plot(r*pixelsize, y, color='red', ls='--', label=r'$K_{zerosline}$')
#         #b.plot(r*pixelsize, Kest.poisson(r), color='green', ls=':', label=r'$K_{pois}$')
#         pp.xlabel('radius [a.u]', fontsize = LABELSIZE, fontweight='bold')
#         pp.ylabel('H-function', fontsize = LABELSIZE, fontweight='bold')
#         pp.text(5, 70, r'Clustering')
#         pp.text(5, -90, r'Dispersion')
#         #pp.text(500, 70, '{}{}'.format('\u03C1=',x_pixelsize))
#         b.set_xlim([0,999])
#         b.set_ylim([-100,100])
#         b.xaxis.set_tick_params(labelsize=LABELSIZE)
#         b.yaxis.set_tick_params(labelsize=LABELSIZE)
#         pp.legend(fontsize = LABELSIZE)
#     
#         pp.subplots_adjust(bottom=0.12, right=0.98, left=0.2,  top=0.95, wspace= 0.5, hspace= 0.43)
#         
# 
#      
#     
#         x_new = t_x[np.where(t_x > 0 )]        
#         y_new = t_y[np.where(t_y > 0 )]
#         
#         s = np.zeros(number_peaks)
#         x = np.zeros(number_peaks)
#         y = np.zeros(number_peaks)
#         
#         for i in range(R_new_thres.size):
#             if R_new_thres[i] not in s:
#                 s[i]= R_new_thres[i]
#                 x[i]= x_new[i]
#                 y[i]= y_new[i]
#             else:
#                 s[i]= 0
#                 x[i]= 0
#                 y[i]= 0
#                  
#         s = s[np.where(s > 0 )] 
#          
#               
#     
#         x_new = x[np.where(x > 0 )]
#         y_new = y[np.where(y > 0 )]
#         
#         x_transfer = x_new - v1_x 
#         y_transfer = y_new - v1_y
#         
#         number_peaks_new = x_new.size
#     
#         ##########################################
#         # printing and saving the results
#         
#         np.savetxt('{}{}{}'.format(path,'x_coord','.dat'), x_coord)      
#         np.savetxt('{}{}{}'.format(path,'y_coord','.dat'), y_coord)   
#         np.savetxt('{}{}{}'.format(path,'x_new','.dat'), x_new)         
#         np.savetxt('{}{}{}'.format(path,'y_new','.dat'), y_new)
#         np.savetxt('{}{}{}'.format(path,'x_transfer','.dat'), x_transfer)       
#         np.savetxt('{}{}{}'.format(path,'y_transfer','.dat'), y_transfer) 
#     
# 
#     
#         for i in range(x_new.size):
#             zz[int(y_new[i]),int(x_new[i])] = 1
#     
#         sigma_1 = 1
#         sigma = [sigma_1, sigma_1]   
#         y = gaussian_filter(zz, sigma, mode='constant')
#         #yy= stats.threshold(y, threshmax=0, newval=255)
#         yy = np.where(y > 0,  y, 255)
#         #pp.imshow(yy, 'hot')
#         print(np.max(yy))
#         xxx = Image.fromarray(yy)#imarray_thres)
#         resized = xxx.resize((300, 300), Image.ANTIALIAS)
#         
# 
#         img = Image.open('{}{}'.format(path,'Overview_image.tiff'))
#         im = np.array(img)
#         im = cm.hot(im)
#         im = np.uint8(im * 255)
#         t = Image.fromarray(im)
#         A = t.convert("RGBA")
#         
#         im = yy
#         im = cm.Greens(im)
#         im = np.uint8(im * 255)
#         t = Image.fromarray(im)
#         B = t.convert("RGBA")
#         C = Image.blend(A, B, alpha=.5)
#         C = C.resize((300, 300), Image.ANTIALIAS)
#         
#         photo = ImageTk.PhotoImage(C)
# 
# 
# #########################
#      
#         canvas = FigureCanvasTkAgg(f1, master = self.frame_topright)
#         canvas.get_tk_widget().grid(row=0, column=0, rowspan=1)
#         label = tk.Label(self.frame_topleft,image=photo)
#         label.image = photo
#         label.grid(row=0)
#         
# #        AREA_spots = 0.09
# #        rho= number_peaks_new/((roi_size*1000000)*(roi_size*1000000))
# #        Lbda= rho*AREA_spots
# #        
# #        TEST = np.zeros((4,4))
# #        
# #        
# #        for i in range(4):
# #            P= (Lbda**i)/math.factorial(i)*math.exp(- Lbda )
# #            TEST[0,i]=i
# #            TEST[1,i]=Lbda
# #            TEST[2,i]=P
# #            TEST[3,i]=i
# #        
# #        print('Prop. of Singlet-events:',np.around(100-(TEST[2,1]/np.sum(TEST[2,1:])*100), decimals=2),'%')  
#             
#             
#         
#         
# #        T.delete('1.0', END)  
# #        T.insert(END,'{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}'.format('peaks_found:  ',number_peaks_new-1,'\n',
# #                                                             '\n',
# #                                                             'Surface-density [µm^(-2)]:',rho,'\n',
# #                                                             'Area of spots [µm^(2)]:',AREA_spots,'\n',
# #                                                             'Total Area [µm^(2)]:',((roi_size*1000000)*(roi_size*1000000)),'\n',
# #                                                             'Prop. of Singlet-events: ',np.around((TEST[2,1]/np.sum(TEST[2,1:])*100), decimals=2),'%','\n',
# #                                                             'Prop. of Multi-events:   ',np.around(100-(TEST[2,1]/np.sum(TEST[2,1:])*100), decimals=2),'%'))
# 
#         ######################################
#         AREA_spots = 0.09
#         rho= number_peaks_new/((roi_size*1000000)*(roi_size*1000000))
#         Lbda= rho*AREA_spots
#         TEST = np.zeros((4,4))
#        
#         for i in range(4):
#             P= (Lbda**i)/math.factorial(i)*math.exp(- Lbda )
#             TEST[0,i]=i
#             TEST[1,i]=Lbda
#             TEST[2,i]=P
#             TEST[3,i]=i
#            
#         T.delete('1.0', tk.END)  
#         T.insert(tk.END,'{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}'.format('peaks_found:  ',number_peaks_new,'\n',
#                                                                  '\n',
#                                                                  'Surface-density [µm^(-2)]:',rho,'\n',
#                                                                  'Area of spots [µm^(2)]:',AREA_spots,'\n',
#                                                                  'Total Area [µm^(2)]:',((roi_size*1000000)*(roi_size*1000000)),'\n',
#                                                                  'Prop. of Singlet-events: ',np.around((TEST[2,1]/np.sum(TEST[2,1:])*100), decimals=2),'%','\n',
#                                                                  'Prop. of Multi-events:   ',np.around(100-(TEST[2,1]/np.sum(TEST[2,1:])*100), decimals=2),'%'))
#         #time_wait_Multirun = number_peaks_new
#             
# # =============================================================================
# #     #this uses some pyramid mean shift filtering whatever it is
# #     if circle ==1:
# #         data_sz = data.size
# #         zz = np.zeros(data_sz)
# #         #aa = roi_size
# #     
# #         if os.path.exists('{}{}'.format(path,foldername)) == False:
# #             os.makedirs('{}{}'.format(path,foldername))
# #             os.makedirs('{}{}'.format(path, 'test_images.tiff'))
# #             
# #         imarray = np.array(data)
# #         imarray_thres = imarray
# #     
# #         array_size_x = imarray_thres.shape[1]
# #         
# #         v1_x = int(array_size_x/2)
# #         v1_y = int(array_size_x/2)
# #         
# #         x_coord, y_coord, image, thresh, gray, shifted, radius_corr, result, image_orig    = find_circle(data, R_min, R_max, threshold, pixelsize_global)
# #              
# #    
# #         x_new = np.array(x_coord)
# #         y_new = np.array(y_coord)
# #             
# #         x_transfer = x_new - v1_x 
# #         y_transfer = y_new - v1_y
# #             
# #         number_peaks_new = x_new.size
# #         
# #             ##########################################
# #             # printing and saving the results
# #             
# #         np.savetxt('{}{}{}'.format(path,'x_coord','.dat'), x_coord)      
# #         np.savetxt('{}{}{}'.format(path,'y_coord','.dat'), y_coord)   
# #         np.savetxt('{}{}{}'.format(path,'x_new','.dat'), x_new)         
# #         np.savetxt('{}{}{}'.format(path,'y_new','.dat'), y_new)
# #         np.savetxt('{}{}{}'.format(path,'x_transfer','.dat'), x_transfer)       
# #         np.savetxt('{}{}{}'.format(path,'y_transfer','.dat'), y_transfer) 
# #         
# #     
# #         
# #         for i in range(x_new.size):
# #             zz[int(y_new[i]),int(x_new[i])] = 1
# #         
# #         sigma_1 = 1
# #         sigma = [sigma_1, sigma_1]   
# #         y = gaussian_filter(zz, sigma, mode='constant')
# #         yy = np.where(y > 0,  y, 255)
# #         print(np.max(yy))
# #         xxx = Image.fromarray(yy)#imarray_thres)
# #         resized = xxx.resize((300, 300), Image.ANTIALIAS)
# #             
# #         img = Image.open('{}{}'.format(path,'Overview_image.tiff'))
# #         img = np.array(image_orig)
# #         img = np.stack((img, img, img), axis = 2)   
# #         img = np.sum(image, axis =2)
# #         im = np.array(img)
# #         im = cm.hot(im)
# #         im = np.uint8(im * 255)
# #         t = Image.fromarray(im)
# #         A = t.convert("RGBA")
# #             
# #         im = yy
# #         im = cm.Greens(im)
# #         im = np.uint8(im * 255)
# #         t = Image.fromarray(im)
# #         B = t.convert("RGBA")
# #         C = Image.blend(A, B, alpha=.5)
# #         C = Image.fromarray(image)
# #         C = C.resize((300, 300), Image.ANTIALIAS)
# #             
# #         photo = ImageTk.PhotoImage(C)
# #         
# #         def Gauss(x, a, x0, sigma):
# #             return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
# # 
# #         model = Model(Gauss)
# #         params = model.make_params()
# #         P = model.param_names
# #         for i in range(len(P)):
# #             params[P[i]].value= 1
# #         
# #         pixelsize_global = np.around(pixelsize_global*10**9,1)
# #          
# #         counts, bins =np.histogram(radius_corr, bins =20)
# #         
# #         x =[]
# #         for i in range(len(bins)-1):
# #             x.append(bins[i]+(bins[i+1]-bins[i])/2)
# #         result  = model.fit(counts, params, x=x)
# #         f1 = pp.figure(figsize=(3,3), dpi=100, edgecolor='k',facecolor = 'r')
# #         LABELSIZE = 7
# #         a = f1.add_subplot(211)
# #         pp.bar(np.array(x)*pixelsize_global, counts, color ='black', label= 'raw', width = 0.8*pixelsize_global)
# #         pp.plot(np.array(x)*pixelsize_global, result.best_fit, 'r-', label='{}{}'.format('Radius [nm]: ',np.around(pixelsize_global*result.values['x0'],0)))
# #         
# #         pp.xlim(R_min*pixelsize_global,R_max*pixelsize_global)
# #         pp.xlabel('Radius [nm]', fontsize = LABELSIZE, fontweight='bold')
# #         pp.ylabel('counts', fontsize = LABELSIZE, fontweight='bold')
# #         a.xaxis.set_tick_params(labelsize=LABELSIZE)
# #         a.yaxis.set_tick_params(labelsize=LABELSIZE)
# #         pp.legend(fontsize = LABELSIZE)
# #         print(pixelsize_global)
# #         
# #         canvas = FigureCanvasTkAgg(f1, master = self.frame_topright)  
# #         canvas.get_tk_widget().grid(row=0, column=0, rowspan=1)          
# #         label = tk.Label(self.frame_topleft,image=photo)
# #         label.image = photo
# #         label.grid(row=0)
# # =============================================================================
# 
# 
# #########################
#            
#         CO = 0
#         #time_wait_Multirun = number_peaks_new
#     #I think these things don't do anything, I will just keep them here for now
#     #self.photoTk = photo # this was giving an error - I hope it is not needed
#     self.number_peaks_new = number_peaks_new
#     self.CO = CO
#     return
# =============================================================================

def SSS(scale_03_value):
    Findpeak()
    
# =============================================================================
# # def set_value(pxsize_01,ROIsize_01, dwell_01, frames_01, var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14, var15):
# #     pixelsize = pxsize_01.get()
# #     Roisize= ROIsize_01.get()
# #     dwelltime = dwell_01.get()
# #     frame_number = frames_01.get()
#     
# #     act485 = var1.get()
# #     act518 = var2.get()
# #     act561 = var3.get()
# #     act640 = var4.get()
# #     act595 = var5.get()
# #     act775 = var6.get()
#     
# #     act485_02 = var7.get()
# #     act518_02 = var8.get()
# #     act561_02 = var9.get()
# #     act640_02 = var10.get()
# #     act595_02 = var11.get()
# #     act775_02 = var12.get()
# #     act_Autofocus = var13.get()
# #     act_QFS = var14.get()
# #     circ = var15.get()
# 
# # #    pixelsize_overview = pxsize_overview_value.get()
#     
#     
# #     return pixelsize, Roisize, dwelltime, frame_number, act485, act518, act561, act640, act595, act775, act485_02, act518_02, act561_02, act640_02, act595_02, act775_02, act_Autofocus, act_QFS, circ
# =============================================================================
    
    
# =============================================================================
# def find_circle(data, radius_thresh_min, radius_thresh_max, distance_thresh, pixelsize_global):
#   
#     # load the image and perform pyramid mean shift filtering
#     # to aid the thresholding step
#     
#     #image_ini = cv2.imread(path)
#     #image = image_ini#[100:300, 200:600]
#     image_ini = data
#     image = np.array(image_ini)
#     image = np.stack((image, image, image), axis = 2)
#     image_orig = image_ini
#     print(np.shape(image_ini))
#     print(np.shape(image))
#     image = image.astype('uint8')
#     shifted = cv2.pyrMeanShiftFiltering(image, 0,0)
#     gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)
#     thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# 
#     #pp.imshow(thresh)
# 
#     D = ndimage.distance_transform_edt(thresh)
#     localMax = peak_local_max(D, indices=True, min_distance=distance_thresh,labels=thresh)
#     localMax_02 = peak_local_max(D, indices=False, min_distance=distance_thresh,labels=thresh)
#     y_coord = localMax[:, 0]
#     x_coord = localMax[:, 1]
#     markers = ndimage.label(localMax_02, structure=np.ones((3, 3)))[0]
#     labels = watershed(-D, markers, mask=thresh)
# 
# 
#     #loop over the unique labels returned by the Watershed
#     #algorithm
# 
#     label = max(np.unique(labels))
#     result = np.zeros([label, 3])
#     for label in np.unique(labels):
#         # if the label is zero, we are examining the 'background'
#         # so simply ignore it
#         if label == 0:
#                 continue
#         # otherwise, allocate memory for the label region and draw
#         # it on the mask
# 
#         else:
#                 mask = np.zeros(gray.shape, dtype="uint8")
#                 mask[labels == label] = 255
#         # detect contours in the mask and grab the largest one
#                 cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 cnts = imutils.grab_contours(cnts)
#                 c = max(cnts, key=cv2.contourArea)
# 
#         # draw a circle enclosing the object
#                 ((x, y), radius) = cv2.minEnclosingCircle(c)
#             
#                 if radius < radius_thresh_min:
#                     continue
#                 elif radius > radius_thresh_max:
#                     continue
#                 else:
#                     cv2.circle(image, (int(x), int(y)), int(radius), (255,0,0), 1, lineType=200)
#                     #cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#                     result[label-1,0] = radius
#                     result[label-1,1] = x
#                     result[label-1,2] = y
# 
# 
# 
#     radius_corr = []    
#     xcoord_corr = []
#     ycoord_corr = []
# 
#     for ii in range(len(result)):
# 
#             if result[:,0][ii]<radius_thresh_min:
#                 continue
#                 
#             elif result[:,0][ii]>radius_thresh_max:
#                 continue
#                 
#             else:
#                 radius_corr.append(result[ii,0])
#                 xcoord_corr.append(result[ii,1])
#                 ycoord_corr.append(result[ii,2])
# 
# 
#     print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))            
#     print("[INFO] {} unique segments deleted".format(len(np.unique(labels))-len(radius_corr)-1))
#     print("[INFO] {} unique segments identified".format(len(radius_corr)))
# 
# 
# 
#     x_coordinate =  xcoord_corr 
#     y_coordinate =  ycoord_corr
# 
#     return x_coordinate, y_coordinate, image, thresh, gray, shifted, radius_corr, result, image_orig
# =============================================================================