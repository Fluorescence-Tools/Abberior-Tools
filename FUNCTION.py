# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:41:20 2020

@author: Abberior_admin
"""
#from tkinter import*
from PIL import Image,ImageTk
from tkinter import ttk
#from specpy import *
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
import tkinter

def SSS(scale_03_value):
    Findpeak()
    
def set_value(pxsize_01,ROIsize_01, dwell_01, frames_01, var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14, var15):
    pixelsize = pxsize_01.get()
    Roisize= ROIsize_01.get()
    dwelltime = dwell_01.get()
    frame_number = frames_01.get()
    
    act485 = var1.get()
    act518 = var2.get()
    act561 = var3.get()
    act640 = var4.get()
    act595 = var5.get()
    act775 = var6.get()
    
    act485_02 = var7.get()
    act518_02 = var8.get()
    act561_02 = var9.get()
    act640_02 = var10.get()
    act595_02 = var11.get()
    act775_02 = var12.get()
    act_Autofocus = var13.get()
    act_QFS = var14.get()
    circ = var15.get()

#    pixelsize_overview = pxsize_overview_value.get()
    
    
    return pixelsize, Roisize, dwelltime, frame_number, act485, act518, act561, act640, act595, act775, act485_02, act518_02, act561_02, act640_02, act595_02, act775_02, act_Autofocus, act_QFS, circ
    
    
def find_circle(data, radius_thresh_min, radius_thresh_max, distance_thresh, pixelsize_global):
  
    # load the image and perform pyramid mean shift filtering
    # to aid the thresholding step
    
    #image_ini = cv2.imread(path)
    #image = image_ini#[100:300, 200:600]
    image_ini = data
    image = np.array(image_ini)
    image = np.stack((image, image, image), axis = 2)
    image_orig = image_ini
    print(np.shape(image_ini))
    print(np.shape(image))
    image = image.astype('uint8')
    shifted = cv2.pyrMeanShiftFiltering(image, 0,0)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #pp.imshow(thresh)

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=True, min_distance=distance_thresh,labels=thresh)
    localMax_02 = peak_local_max(D, indices=False, min_distance=distance_thresh,labels=thresh)
    y_coord = localMax[:, 0]
    x_coord = localMax[:, 1]
    markers = ndimage.label(localMax_02, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)


    #loop over the unique labels returned by the Watershed
    #algorithm

    label = max(np.unique(labels))
    result = np.zeros([label, 3])
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
                continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask

        else:
                mask = np.zeros(gray.shape, dtype="uint8")
                mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                c = max(cnts, key=cv2.contourArea)

        # draw a circle enclosing the object
                ((x, y), radius) = cv2.minEnclosingCircle(c)
            
                if radius < radius_thresh_min:
                    continue
                elif radius > radius_thresh_max:
                    continue
                else:
                    cv2.circle(image, (int(x), int(y)), int(radius), (255,0,0), 1, lineType=200)
                    #cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    result[label-1,0] = radius
                    result[label-1,1] = x
                    result[label-1,2] = y



    radius_corr = []    
    xcoord_corr = []
    ycoord_corr = []

    for ii in range(len(result)):

            if result[:,0][ii]<radius_thresh_min:
                continue
                
            elif result[:,0][ii]>radius_thresh_max:
                continue
                
            else:
                radius_corr.append(result[ii,0])
                xcoord_corr.append(result[ii,1])
                ycoord_corr.append(result[ii,2])


    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))            
    print("[INFO] {} unique segments deleted".format(len(np.unique(labels))-len(radius_corr)-1))
    print("[INFO] {} unique segments identified".format(len(radius_corr)))



    x_coordinate =  xcoord_corr 
    y_coordinate =  ycoord_corr

    return x_coordinate, y_coordinate, image, thresh, gray, shifted, radius_corr, result, image_orig
    
def Connect(T):
    global a
    a=0
    try:
        im=specpy.Imspector()
    except (RuntimeError):
        a=1
        print(a)
        
    if a==0:
        print('connect')
        T.insert(END, 'Connected')
    else:
        print('disconnect')
        T.insert(END, 'Disconnected')
    return a


def Overview(Multi, Pos, path, foldername, scale_01, frame_top, T,  laser_overview_value, laser_overview_entry, frames_overview_value, ROIsize_overview_value, dwell_overview_value, pxsize_overview_value, circle):   
    
    #residual / unneeded imports
    #import time
    #global pix_data #whatever this is, it should not be global
    
    #some directory is made
    if os.path.exists('{}{}'.format(path,foldername)) == False:
        os.makedirs('{}{}'.format(path,foldername))    
    else:
        shutil.rmtree('{}{}{}'.format(path,foldername,'/'))#(D:/current data/testfolder')
        os.makedirs('{}{}'.format(path,foldername))
    
    
    #get values from GUI entries
    roi_size = float(ROIsize_overview_value.get())*1e-06          # in meter
    Dwelltime= np.around(float(dwell_overview_value.get()))*1e-06         # in seconds  
    number_frames = int(frames_overview_value.get())      #Type number of frames t in xyt mode
    z_position = 0        # in meter
    x_pixelsize = float(pxsize_overview_value.get())*1e-09       # in meter
    y_pixelsize = float(pxsize_overview_value.get())*1e-09       # in meter
    z_pixelsize = float(pxsize_overview_value.get())*1e-09       # in meter
    px_num = roi_size/x_pixelsize


    Streaming_HydraHarp= True # Type True or False for Streaming via HydraHarp
    modelinesteps= True      #Type True for linesteps
    xyt_mode = 784#1296           #Type 785 for xyt mode  784
    
    
     #here the laser values are read in, but why the strange format
    laser_overview = [int(s) for s in laser_overview_entry.get().split(',')]
    laser_overview_VALUE = [int(s) for s in laser_overview_value.get().split(',')]
#    laser_overview_len = len([laser_overview]) 
    laser_steps = len(laser_overview)
    number_linesteps = laser_steps
    
    #this piece of code seems to set which lasers are active, but can it be shorter? Should it be left as-is?
    LASER_ACT =[]
    for i in range(len(laser_overview)):
        if laser_overview[i]  == 485:laservalue = 0
        elif laser_overview[i]  == 518:laservalue = 1
        elif laser_overview[i]  == 595:laservalue = 2
        elif laser_overview[i]  == 561:laservalue = 3
        elif laser_overview[i]  == 640:laservalue = 4
        elif laser_overview[i]  == 775:laservalue = 5
        #this seems like a bug
        laser_activ = [False]*8 
        laser_activ[laservalue] = True
        LASER_ACT.append(laser_activ)
    if not len(LASER_ACT) == 8:
        calc = 8-len(LASER_ACT)
        for i in range(calc):
            LASER_ACT.append([False]*8)
#    print(LASER_ACT)
            
    
    #this sets which linesteps are active
    linesteps = [False]*8
    linesteps_number = [0]*8
    for i in range(number_linesteps):
        linesteps[i] = True
        linesteps_number[i] = 1
        
    
    ############

    
    if a==0:  #what is a?
        path = 'D:/current data/'
        im = specpy.Imspector()    
        c=im.create_measurement()
                
        if Multi == 1: # why is MULTI needed for an overview?
            print('MULTIRUN')
            c.set_parameters('ExpControl/scan/range/offsets/coarse/y/g_off', Pos) # current stage position in x [m]
        else:
            print('non MULTIRUN')
    
#        if Activate_QFS == True:
#            M_obj= im.measurement(c.name())
        #a bunch of these steps are default and can be put into a helper function
        c.set_parameters('ExpControl/scan/range/mode',xyt_mode)
        c.set_parameters('OlympusIX/scanrange/z/z-stabilizer/enabled', True)
        c.set_parameters('ExpControl/scan/range/x/psz',x_pixelsize)
        c.set_parameters('ExpControl/scan/range/y/psz',y_pixelsize)
        c.set_parameters('ExpControl/scan/range/z/psz',z_pixelsize)
        c.set_parameters('ExpControl/scan/range/z/off', 1e-15)#z_position*z_pixelsize)
        c.set_parameters('ExpControl/scan/range/x/len',roi_size)
        c.set_parameters('ExpControl/scan/range/y/len',roi_size)
        c.set_parameters('ExpControl/scan/range/z/len',0)
        c.set_parameters('Pinhole/pinhole_size', 10e-5)
        c.set_parameters('ExpControl/scan/dwelltime', Dwelltime)
        c.set_parameters('ExpControl/scan/range/t/res',number_frames)
        c.set_parameters('HydraHarp/data/streaming/enable', False)
        c.set_parameters('HydraHarp/is_active', True)
        c.set_parameters('ExpControl/gating/linesteps/steps_active', linesteps)
        c.set_parameters('ExpControl/gating/tcspc/channels/0/mode', 0)
        c.set_parameters('ExpControl/gating/tcspc/channels/0/stream', 'HydraHarp')
        c.set_parameters('ExpControl/gating/tcspc/channels/1/mode', 0)
        c.set_parameters('ExpControl/gating/tcspc/channels/1/stream', 'HydraHarp')
        c.set_parameters('ExpControl/gating/tcspc/channels/2/mode', 0)
        c.set_parameters('ExpControl/gating/tcspc/channels/2/stream', 'HydraHarp')
        c.set_parameters('ExpControl/gating/tcspc/channels/3/mode', 0)
        c.set_parameters('ExpControl/gating/tcspc/channels/3/stream', 'HydraHarp')
        c.set_parameters('ExpControl/scan/detsel/detsel',['APD1', 'APD2', 'APD3', 'APD4'])
        c.set_parameters('ExpControl/gating/linesteps/chans_enabled',[True, True, True, True])
        c.set_parameters('ExpControl/gating/pulses/pulse_chan/delay',[0.0, 0.0, 0.0, 0.0])
        c.set_parameters('ExpControl/gating/linesteps/laser_enabled',[True, True, True, True, True, True, False, False])
        c.set_parameters('ExpControl/gating/linesteps/laser_on',LASER_ACT)

        #why only if laser steps = 1?
        if laser_steps == 1:
            c.set_parameters('ExpControl/gating/linesteps/step_values',linesteps_number)
            c.set_parameters('{}{}{}'.format('ExpControl/lasers/power_calibrated/', laservalue,'/value/calibrated'), laser_overview_VALUE[0])
            c.set_parameters('ExpControl/gating/linesteps/chans_on', [[True, True, True, True],[True, True, True, True],[True, True, True, True],[False, False, False, False],[False, False, False, False],[False, False, False, False],[False, False, False, False],[False, False, False, False]])
            
        else: 
            for i in range(len(laser_overview)):
                if laser_overview[i]  == 485:laservalue = 0
                elif laser_overview[i]  == 518:laservalue = 1
                elif laser_overview[i]  == 595:laservalue = 2
                elif laser_overview[i]  == 561:laservalue = 3
                elif laser_overview[i]  == 640:laservalue = 4
                elif laser_overview[i]  == 775:laservalue = 5
                c.set_parameters('{}{}{}'.format('ExpControl/lasers/power_calibrated/', laservalue,'/value/calibrated'), laser_overview_VALUE[i])
            c.set_parameters('ExpControl/gating/linesteps/step_values',linesteps_number)
            c.set_parameters('ExpControl/gating/linesteps/chans_on', [[True, False, True, False],[False, True, False, True],[True, True, True, True],[False, False, False, False],[False, False, False, False],[False, False, False, False],[False, False, False, False],[False, False, False, False]])
        
        #here the measurement is actually run
        meas= im.active_measurement()
        im.run(meas)
        
        #time.sleep(1)
        #here the data is read out of the image
        stk_names = meas.stack_names()
        stk0 = meas.stack(0)
        stk1 = meas.stack(1)
        stk2 = meas.stack(2)
        stk3 = meas.stack(3)
        pix_data0 = stk0.data()
        pix_data1 = stk1.data()
        pix_data2 = stk2.data()
        pix_data3 = stk3.data()

        pix_data = pix_data0
        r1 = 0
        r2 = 0
        print(pix_data0.shape)
        #here some adding is done, but this multiplication factor x10 is dubious
        if pix_data0.shape == (1,number_frames,np.round(px_num,0),np.round(px_num,0)):
            r1 = (np.sum(pix_data0, axis=1)[0] + np.sum(pix_data2, axis=1)[0])*10
            r2 = (np.sum(pix_data1, axis=1)[0] + np.sum(pix_data3, axis=1)[0])*10
            io_green = Image.fromarray(r1)
            io_red =   Image.fromarray(r2)
            pp.imshow(io_red)

            #this seems like it can go.
            if laser_steps == 1:
                data = io_red
            else:
                #data =  r2
                data = io_red

        #this is capturing some bug where the data has funky dimensions
        elif pix_data.shape == (1, px_num, px_num, 1):
            io = Image.fromarray(pix_data[0,:,:,0]*10)
            pp.imshow(io)
            data = io
        else:
            io = Image.fromarray((np.mean(pix_data, axis=3)[0])*10) # create greyscale image object  
            pp.imshow(io)
            data = io
    #this else statement seems unneeded
    else:
        #these if statements too
        if circle == 0:
            #path = 'C:/Users/buddeja/Desktop/GUI/'
            r1 =1
            r2 = 2
            data = Image.open('{}{}'.format(path,'offline_image.tiff'))
        elif circle == 1:
            #path = 'C:/Users/buddeja/Desktop/GUI/'
            r1 =1
            r2 = 2
            data = Image.open('{}{}'.format(path,'offline_image.tiff'))
            


    io_1 = np.array(data)
    io = Image.fromarray(io_1 ) 
    scale_01.config(to =np.max(io))    
           
    io.save('{}{}'.format(path,'Overview_image.tiff'), format = 'TIFF')#'D:/current data/','Overview_image.tiff'
    print(path) 
    #renaming of Imspector as IMAGE BAD IDEA
    photo = np.array(data)
    photo = cm.hot(photo)
    image = np.uint8(photo * 255)
    photo = Image.fromarray(photo)
    photo = photo.resize((300, 300), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(photo)
    label = Label(frame_top,image=photo)
    label.image = photo
    label.grid(row=0)
    T.delete('1.0', END) 
    T.insert(END, "Overview created") 
    return data, roi_size, r1, r2, x_pixelsize
  
    
def Findpeak(path, foldername, scale_01, scale_02, scale_03, roi_size, frame_top, T, frame_top3, a, circle, pixelsize_global):
    from PIL import Image,ImageTk
    from scipy.stats import stats
    from skimage.feature import peak_local_max
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg#, NavigationToolbar2TkAgg
    from scipy.ndimage.filters import gaussian_filter
    import matplotlib.cm as cm
    import numpy as np
    import os
    import math
    import scipy.stats
    from scipy.spatial import distance
    from astropy.stats import RipleysKEstimator
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as pp
    
    global x_new
    global y_new
    global R
    global scale_val
    global aa
    global number_peaks_new 
    global time_wait_Multirun
    
    global x_transfer
    global y_transfer
    global CO

    threshold = scale_01.get()      # define the lower threshold value
    print("threshold value is %.1f" % threshold)
    R_min = scale_02.get()          # additional distance threshold
    R_max = scale_03.get()
    exc_border_peaks = 1 # type 0 for activate for border peaks 
    if a==0:
        path = 'D:/current data/'
        data = Image.open('{}{}'.format(path,'Overview_image.tiff'))
        
    else:
        if circle ==0:
            path = path
            data = Image.open('{}{}'.format(path,'Overview_image.tiff'))
        elif circle == 1:
            path = path
            data = Image.open('{}{}'.format(path,'Overview_image.tiff'))

            
        
    if circle ==0:   
        data_sz = data.size
        zz = np.zeros(data_sz)
        aa = roi_size
        
    
    ############################################
        #Calculate parameters    
    ############################################
    
        if os.path.exists('{}{}'.format(path,foldername)) == False:
            os.makedirs('{}{}'.format(path,foldername))
            os.makedirs('{}{}'.format(path, 'test_images.tiff'))
            
    
        imarray = np.array(data)
        imarray_thres = np.where(imarray > threshold,  imarray, 0)
    
        array_size_x = imarray_thres.shape[1]
        
        v1_x = int(array_size_x/2)
        v1_y = int(array_size_x/2)
        
        ##############################################
        # Find peaks in images:   
        coordinates = peak_local_max(imarray_thres, min_distance= R_min, indices=True) #exclude_border=exc_border_peaks)
        y_coord = coordinates[:, 0]
        x_coord = coordinates[:, 1]
         
        if len(coordinates)<= 1:
            CO=1
            print('NearestNeighbors failed: coordinates is <= 1')
            x_transfer = []
            y_transfer = []
            np.savetxt('{}{}{}'.format(path,'x_transfer','.dat'), x_transfer)       
            np.savetxt('{}{}{}'.format(path,'y_transfer','.dat'), y_transfer) 
            
        else:
            CO= 0
            print('NearestNeighbors okay')
            nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(coordinates)
            distances, indices = nbrs.kneighbors(coordinates) 
                
            number_peaks = int(coordinates.size/ 2)
            t_x= np.zeros(number_peaks)
            t_y= np.zeros(number_peaks)
            R_new = np.zeros(number_peaks)
            
            for i in range(number_peaks):
                for ii in range(number_peaks):
                    R = math.sqrt((coordinates[i][0]-coordinates[ii][0])**2+(coordinates[i][1]-coordinates[ii][1])**2)
                    
                    if R < R_max and R > 0:#R_min:
                        #print ('peaks',i,ii,'x1_coord',coordinates[i][0],'y1_coord',coordinates[i][1],'x2_coord',coordinates[ii][0],'y2_coord',coordinates[ii][1],'Distance R',R,'below', R_min  )
                        #print (coordinates[i][0],coordinates[i][1])
                        t_y[i] = coordinates[i][0]
                        t_x[i] = coordinates[i][1]
                        R_new[i] = R
      
            R_new_thres = R_new[np.where(R_new>0)]
        
            ###################################
            # Libary import
                
            # Do you want to simulate a Poisson distribution of spots [own_data = 0] or use own data [own_data = 1]????
            own_data = 1
                 
            ###############  Window parameters  ##########################################
            pixelsize= 1 #[nm]
            
            xMin=0
            xMax=data.size[0]; # pixelnumber in x
            yMin=0
            yMax=data.size[1]; # pixelnumber in y
             
            #Poisson process parameters simulation
            lambda0=5; #intensity (ie mean density) of the Poisson process
             
            ###############################################################################
            if own_data ==0:
                print('Poisson distribution of spots')
                #Simulate Poisson point process
                xDelta=xMax-xMin;yDelta=yMax-yMin; #rectangle dimensions
                areaTotal=xDelta*yDelta; #total area
                numbPoints = scipy.stats.poisson( lambda0*areaTotal ).rvs()#Poisson number of points
                x_Poisson = xDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+xMin#x coordinates of Poisson points
                y_Poisson = yDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+yMin#y coordinates of Poisson points
                xx = x_Poisson
                yy = y_Poisson
                
            else:
                print('Using real data for statistics')
                xDelta=xMax-xMin;yDelta=yMax-yMin; #rectangle dimensions
                areaTotal=xDelta*yDelta; #total area
                    
           
            z = coordinates
            #np.savetxt('C:/Users/buddeja/Desktop/test/z_GUI.txt', z, delimiter='\t')   # X is an array
            #print(z.shape)
            Kest = RipleysKEstimator(area=areaTotal, x_max=xMax, y_max=yMax, x_min=xMin, y_min=yMin)
            r = np.linspace(0, xMax, 1000)
            ###########################################################################
        
        
            f1 = pp.figure(figsize=(3,3), dpi=100, edgecolor='k',facecolor = 'r')
            LABELSIZE = 7
            a = f1.add_subplot(211)
            a.hist(distances[:,1], bins=20,color='blue', label='1st neighbor',alpha=0.2)
            a.hist(distances[:,2], bins=20,color='red', label='2nd neighbor',alpha=0.2)
            pp.xlabel('Nearst neighbor distance [a.u]', fontsize = LABELSIZE, fontweight='bold')
            pp.ylabel('counts', fontsize = LABELSIZE, fontweight='bold')
            a.xaxis.set_tick_params(labelsize=LABELSIZE)
            a.yaxis.set_tick_params(labelsize=LABELSIZE)
            pp.legend(fontsize = LABELSIZE)
        
            
            y = r*0
            b = f1.add_subplot(212)
            #b.plot(r*pixelsize, Kest(data=z, radii=r, mode='translation'), color='black',label=r'$K_{trans}$')
            b.plot(r*pixelsize, Kest.Hfunction(data=z, radii=r, mode='translation'), color='black',label=r'$K_{trans}$')
            b.plot(r*pixelsize, y, color='red', ls='--', label=r'$K_{zerosline}$')
            #b.plot(r*pixelsize, Kest.poisson(r), color='green', ls=':', label=r'$K_{pois}$')
            pp.xlabel('radius [a.u]', fontsize = LABELSIZE, fontweight='bold')
            pp.ylabel('H-function', fontsize = LABELSIZE, fontweight='bold')
            pp.text(5, 70, r'Clustering')
            pp.text(5, -90, r'Dispersion')
            #pp.text(500, 70, '{}{}'.format('\u03C1=',x_pixelsize))
            b.set_xlim([0,999])
            b.set_ylim([-100,100])
            b.xaxis.set_tick_params(labelsize=LABELSIZE)
            b.yaxis.set_tick_params(labelsize=LABELSIZE)
            pp.legend(fontsize = LABELSIZE)
        
            pp.subplots_adjust(bottom=0.12, right=0.98, left=0.2,  top=0.95, wspace= 0.5, hspace= 0.43)
            
    
         
        
            x_new = t_x[np.where(t_x > 0 )]        
            y_new = t_y[np.where(t_y > 0 )]
            
            s = np.zeros(number_peaks)
            x = np.zeros(number_peaks)
            y = np.zeros(number_peaks)
            
            for i in range(R_new_thres.size):
                if R_new_thres[i] not in s:
                    s[i]= R_new_thres[i]
                    x[i]= x_new[i]
                    y[i]= y_new[i]
                else:
                    s[i]= 0
                    x[i]= 0
                    y[i]= 0
                     
            s = s[np.where(s > 0 )] 
             
                  
        
            x_new = x[np.where(x > 0 )]
            y_new = y[np.where(y > 0 )]
            
            x_transfer = x_new - v1_x 
            y_transfer = y_new - v1_y
            
            number_peaks_new = x_new.size
        
            ##########################################
            # printing and saving the results
            
            np.savetxt('{}{}{}'.format(path,'x_coord','.dat'), x_coord)      
            np.savetxt('{}{}{}'.format(path,'y_coord','.dat'), y_coord)   
            np.savetxt('{}{}{}'.format(path,'x_new','.dat'), x_new)         
            np.savetxt('{}{}{}'.format(path,'y_new','.dat'), y_new)
            np.savetxt('{}{}{}'.format(path,'x_transfer','.dat'), x_transfer)       
            np.savetxt('{}{}{}'.format(path,'y_transfer','.dat'), y_transfer) 
        
    
        
            for i in range(x_new.size):
                zz[int(y_new[i]),int(x_new[i])] = 1
        
            sigma_1 = 1
            sigma = [sigma_1, sigma_1]   
            y = gaussian_filter(zz, sigma, mode='constant')
            #yy= stats.threshold(y, threshmax=0, newval=255)
            yy = np.where(y > 0,  y, 255)
            #pp.imshow(yy, 'hot')
            print(np.max(yy))
            xxx = Image.fromarray(yy)#imarray_thres)
            resized = xxx.resize((300, 300), Image.ANTIALIAS)
            
    
            img = Image.open('{}{}'.format(path,'Overview_image.tiff'))
            im = np.array(img)
            im = cm.hot(im)
            im = np.uint8(im * 255)
            t = Image.fromarray(im)
            A = t.convert("RGBA")
            
            im = yy
            im = cm.Greens(im)
            im = np.uint8(im * 255)
            t = Image.fromarray(im)
            B = t.convert("RGBA")
            C = Image.blend(A, B, alpha=.5)
            C = C.resize((300, 300), Image.ANTIALIAS)
            
            photo = ImageTk.PhotoImage(C)


#########################
         
            canvas = FigureCanvasTkAgg(f1, master = frame_top3)
            canvas.get_tk_widget().grid(row=0, column=0, rowspan=1)
            label = Label(frame_top,image=photo)
            label.image = photo
            label.grid(row=0)
            
    #        AREA_spots = 0.09
    #        rho= number_peaks_new/((roi_size*1000000)*(roi_size*1000000))
    #        Lbda= rho*AREA_spots
    #        
    #        TEST = np.zeros((4,4))
    #        
    #        
    #        for i in range(4):
    #            P= (Lbda**i)/math.factorial(i)*math.exp(- Lbda )
    #            TEST[0,i]=i
    #            TEST[1,i]=Lbda
    #            TEST[2,i]=P
    #            TEST[3,i]=i
    #        
    #        print('Prop. of Singlet-events:',np.around(100-(TEST[2,1]/np.sum(TEST[2,1:])*100), decimals=2),'%')  
                
                
            
            
    #        T.delete('1.0', END)  
    #        T.insert(END,'{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}'.format('peaks_found:  ',number_peaks_new-1,'\n',
    #                                                             '\n',
    #                                                             'Surface-density [µm^(-2)]:',rho,'\n',
    #                                                             'Area of spots [µm^(2)]:',AREA_spots,'\n',
    #                                                             'Total Area [µm^(2)]:',((roi_size*1000000)*(roi_size*1000000)),'\n',
    #                                                             'Prop. of Singlet-events: ',np.around((TEST[2,1]/np.sum(TEST[2,1:])*100), decimals=2),'%','\n',
    #                                                             'Prop. of Multi-events:   ',np.around(100-(TEST[2,1]/np.sum(TEST[2,1:])*100), decimals=2),'%'))
    
            ######################################
            AREA_spots = 0.09
            rho= number_peaks_new/((roi_size*1000000)*(roi_size*1000000))
            Lbda= rho*AREA_spots
            TEST = np.zeros((4,4))
           
            for i in range(4):
                P= (Lbda**i)/math.factorial(i)*math.exp(- Lbda )
                TEST[0,i]=i
                TEST[1,i]=Lbda
                TEST[2,i]=P
                TEST[3,i]=i
               
            T.delete('1.0', END)  
            T.insert(END,'{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}'.format('peaks_found:  ',number_peaks_new,'\n',
                                                                     '\n',
                                                                     'Surface-density [µm^(-2)]:',rho,'\n',
                                                                     'Area of spots [µm^(2)]:',AREA_spots,'\n',
                                                                     'Total Area [µm^(2)]:',((roi_size*1000000)*(roi_size*1000000)),'\n',
                                                                     'Prop. of Singlet-events: ',np.around((TEST[2,1]/np.sum(TEST[2,1:])*100), decimals=2),'%','\n',
                                                                     'Prop. of Multi-events:   ',np.around(100-(TEST[2,1]/np.sum(TEST[2,1:])*100), decimals=2),'%'))
            time_wait_Multirun = number_peaks_new
            
    if circle ==1:
        data_sz = data.size
        zz = np.zeros(data_sz)
        aa = roi_size
    
        if os.path.exists('{}{}'.format(path,foldername)) == False:
            os.makedirs('{}{}'.format(path,foldername))
            os.makedirs('{}{}'.format(path, 'test_images.tiff'))
            
        imarray = np.array(data)
        imarray_thres = imarray
    
        array_size_x = imarray_thres.shape[1]
        
        v1_x = int(array_size_x/2)
        v1_y = int(array_size_x/2)
        
        x_coord, y_coord, image, thresh, gray, shifted, radius_corr, result, image_orig    = find_circle(data, R_min, R_max, threshold, pixelsize_global)
             
   
        x_new = np.array(x_coord)
        y_new = np.array(y_coord)
            
        x_transfer = x_new - v1_x 
        y_transfer = y_new - v1_y
            
        number_peaks_new = x_new.size
        
            ##########################################
            # printing and saving the results
            
        np.savetxt('{}{}{}'.format(path,'x_coord','.dat'), x_coord)      
        np.savetxt('{}{}{}'.format(path,'y_coord','.dat'), y_coord)   
        np.savetxt('{}{}{}'.format(path,'x_new','.dat'), x_new)         
        np.savetxt('{}{}{}'.format(path,'y_new','.dat'), y_new)
        np.savetxt('{}{}{}'.format(path,'x_transfer','.dat'), x_transfer)       
        np.savetxt('{}{}{}'.format(path,'y_transfer','.dat'), y_transfer) 
        
    
        
        for i in range(x_new.size):
            zz[int(y_new[i]),int(x_new[i])] = 1
        
        sigma_1 = 1
        sigma = [sigma_1, sigma_1]   
        y = gaussian_filter(zz, sigma, mode='constant')
        yy = np.where(y > 0,  y, 255)
        print(np.max(yy))
        xxx = Image.fromarray(yy)#imarray_thres)
        resized = xxx.resize((300, 300), Image.ANTIALIAS)
            
        img = Image.open('{}{}'.format(path,'Overview_image.tiff'))
        img = np.array(image_orig)
        img = np.stack((img, img, img), axis = 2)   
        img = np.sum(image, axis =2)
        im = np.array(img)
        im = cm.hot(im)
        im = np.uint8(im * 255)
        t = Image.fromarray(im)
        A = t.convert("RGBA")
            
        im = yy
        im = cm.Greens(im)
        im = np.uint8(im * 255)
        t = Image.fromarray(im)
        B = t.convert("RGBA")
        C = Image.blend(A, B, alpha=.5)
        C = Image.fromarray(image)
        C = C.resize((300, 300), Image.ANTIALIAS)
            
        photo = ImageTk.PhotoImage(C)
        
        def Gauss(x, a, x0, sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

        model = Model(Gauss)
        params = model.make_params()
        P = model.param_names
        for i in range(len(P)):
            params[P[i]].value= 1
        
        pixelsize_global = np.around(pixelsize_global*10**9,1)
         
        counts, bins =np.histogram(radius_corr, bins =20)
        
        x =[]
        for i in range(len(bins)-1):
            x.append(bins[i]+(bins[i+1]-bins[i])/2)
        result  = model.fit(counts, params, x=x)
        f1 = pp.figure(figsize=(3,3), dpi=100, edgecolor='k',facecolor = 'r')
        LABELSIZE = 7
        a = f1.add_subplot(211)
        pp.bar(np.array(x)*pixelsize_global, counts, color ='black', label= 'raw', width = 0.8*pixelsize_global)
        pp.plot(np.array(x)*pixelsize_global, result.best_fit, 'r-', label='{}{}'.format('Radius [nm]: ',np.around(pixelsize_global*result.values['x0'],0)))
        
        pp.xlim(R_min*pixelsize_global,R_max*pixelsize_global)
        pp.xlabel('Radius [nm]', fontsize = LABELSIZE, fontweight='bold')
        pp.ylabel('counts', fontsize = LABELSIZE, fontweight='bold')
        a.xaxis.set_tick_params(labelsize=LABELSIZE)
        a.yaxis.set_tick_params(labelsize=LABELSIZE)
        pp.legend(fontsize = LABELSIZE)
        print(pixelsize_global)
        
        canvas = FigureCanvasTkAgg(f1, master = frame_top3)  
        canvas.get_tk_widget().grid(row=0, column=0, rowspan=1)          
        label = Label(frame_top,image=photo)
        label.image = photo
        label.grid(row=0)


#########################
           
        CO = 0
        time_wait_Multirun = number_peaks_new
             
        
    return photo, number_peaks_new, CO
    

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

    
 

def Run_meas(pixelsize, Roisize, dwelltime, frame_number, act485, act518, act561, act640, act595, act775, act485_02, act518_02, act561_02, act640_02, act595_02, act775_02, act_Autofocus, act_QFS,L485_value_01,L518_value_01,L561_value_01, L640_value_01, L595_value_01, L775_value_01, T, mm, Pos, pixelsize_global):
    
    Multi = mm
    Pos = Pos
    z_position =  5e-08 #float(pixelsize)*1e-09         # in meter
    x_pixelsize = float(pixelsize)*1e-09        # in meter
    y_pixelsize = float(pixelsize)*1e-09        # in meter
    z_pixelsize = float(pixelsize)*1e-09        # in meter
    roi_size =    float(Roisize)*1e-06            # in meter 
    Dwelltime= float(dwelltime)*1e-06         # in seconds  
           
    LP485 = L485_value_01.get()
    LP518 = L518_value_01.get()
    LP561 = L561_value_01.get()
    LP640 = L640_value_01.get()
    LP595 = L595_value_01.get()
    LP775 = L775_value_01.get()
                
    Streaming_HydraHarp= True # Type True or False for Streaming via HydraHarp 
    modelinesteps= True      #Type True for linesteps
    number_linesteps = 2     #Type the number of linesteps
    xyt_mode = 784           #Type 785 for xyt mode
    
    number_frames = float(frame_number)     #Type number of frames t in xyt mode
    #time_wait = math.ceil((roi_size/x_pixelsize) * (roi_size/y_pixelsize)* number_frames* Dwelltime) + 1
    #measurement time consists of line scan rate + flyback time + buffer
    time_wait = 1
                    
    Activate485 = bool(act485)                    
    Activate518 = bool(act518)   
    Activate561 = bool(act561)   
    Activate640 = bool(act640)   
    Activate595 = bool(act595)   
    Activate775 = bool(act775)   
    
    Activate485_02 = bool(act485_02)                    
    Activate518_02 = bool(act518_02)   
    Activate561_02 = bool(act561_02)   
    Activate640_02 = bool(act640_02)   
    Activate595_02 = bool(act595_02)   
    Activate775_02 = bool(act775_02) 
    #Activate_Autofocus= bool(act_Autofocus)
        
        
            
        ############################
        #Hydraharp-setting
            
    if Streaming_HydraHarp == True:
        stream = 'HydraHarp'
    else:
        stream = 'fpga'
                       
    Ch1_mode= 0                # type 0 for counter and 1 for flim
    Ch1_stream= stream         # type hydraharp or fpga for internal
    Ch2_mode= 0          
    Ch2_stream= stream
    Ch3_mode= 0         
    Ch3_stream= stream
    Ch4_mode= 0         
    Ch4_stream= stream
            
        ############################
        #detector activation
        
    detector1= True
    detector2= True
    detector3= True
    detector4= True
            
       ####################################################################################
            
    T.delete('1.0', END) 
    T.insert(END, 'L485',L485_value_01.get())
            
        
    linesteps = [False, False, False, False, False, False, False, False]
    for i in range(number_linesteps):
        linesteps[i] = True
            
        #M_obj= im.measurement(im.measurement_names()[0])
    if a==0:
        import math
        import time
        
        im = specpy.Imspector() 
        d = im.active_measurement()
        M_obj= im.measurement(d.name())
                 
    #    if Activate_Autofocus == True: #Activate_Autofocus
    #        M_obj.set_parameters('OlympusIX/scanrange/z/z-stabilizer/enabled', False)
    #        time.sleep(2) 
    #        M_obj.set_parameters('OlympusIX/scanrange/z/z-stabilizer/enabled', True)
            
                        
        if CO ==1 :
            print('failed')
            save_path = '{}{}{}{}{}'.format('D:/current data/','Overview_',Pos,'_numberSPOTS_', 0)
            os.makedirs(save_path)  
            (im.measurement(im.measurement_names()[1])).save_as('{}{}{}{}{}'.format(save_path,'/','Overview_', Pos,'.msr'))
            im.close(im.measurement(im.measurement_names()[1]))
        else:
            x_global_offset = M_obj.parameters('ExpControl/scan/range/x/off')
            y_global_offset = M_obj.parameters('ExpControl/scan/range/y/off')
            x_roi_new = np.loadtxt('D:/current data/x_transfer.dat') - x_global_offset
            y_roi_new = np.loadtxt('D:/current data/y_transfer.dat') - y_global_offset
            for i in range(1,x_roi_new.size):
                x_position = x_roi_new[i]
                y_position = y_roi_new[i]
                   
                d = im.active_measurement()
                M_obj.clone(d.active_configuration())
                M_obj.activate(M_obj.configuration(i))          
                c = M_obj.configuration(i)
                
                c.set_parameters('ExpControl/scan/range/x/psz',x_pixelsize)
                c.set_parameters('ExpControl/scan/range/y/psz',y_pixelsize)
                c.set_parameters('ExpControl/scan/range/z/psz',z_pixelsize)
                c.set_parameters('ExpControl/scan/range/x/off', x_position*pixelsize_global )#+ ROI_offset)
                c.set_parameters('ExpControl/scan/range/y/off', y_position*pixelsize_global )#+ ROI_offset)
                c.set_parameters('ExpControl/scan/range/z/off', 1e-15)#z_position*z_pixelsize)
                c.set_parameters('ExpControl/scan/range/x/len',roi_size)
                c.set_parameters('ExpControl/scan/range/y/len',roi_size)
                c.set_parameters('ExpControl/scan/range/z/len',roi_size)
                c.set_parameters('ExpControl/scan/dwelltime', Dwelltime)
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
                c.set_parameters('ExpControl/gating/linesteps/step_values',[1,1,0,0,0,0,0,0])
                c.set_parameters('ExpControl/scan/range/mode',xyt_mode)
                c.set_parameters('ExpControl/lasers/power_calibrated/0/value/calibrated', float(LP485))
                c.set_parameters('ExpControl/lasers/power_calibrated/2/value/calibrated', float(LP518))
                c.set_parameters('ExpControl/lasers/power_calibrated/3/value/calibrated', float(LP561))
                c.set_parameters('ExpControl/lasers/power_calibrated/4/value/calibrated', float(LP640))
                c.set_parameters('ExpControl/lasers/power_calibrated/5/value/calibrated', float(LP775))
                c.set_parameters('ExpControl/gating/pulses/pulse_chan/delay',[0.0, 0.0, 0.0, 0.0])
                c.set_parameters('ExpControl/gating/linesteps/laser_enabled',[True, True, True, True, True, True, False, False])
                c.set_parameters('ExpControl/gating/linesteps/laser_on',[[Activate485, Activate518, Activate595, Activate561, Activate640, Activate775, False, False],
                 [Activate485_02, Activate518_02, Activate595_02, Activate561_02, Activate640_02, Activate775_02, False, False],
                 [Activate485, Activate518, Activate595, Activate561, Activate640, Activate775, False, False],
                 [Activate485, Activate518, Activate595, Activate561, Activate640, Activate775, False, False],
                 [Activate485, Activate518, Activate595, Activate561, Activate640, Activate775, False, False],
                 [Activate485, Activate518, Activate595, Activate561, Activate640, Activate775, False, False],
                 [Activate485, Activate518, Activate595, Activate561, Activate640, Activate775, False, False],
                 [Activate485, Activate518, Activate595, Activate561, Activate640, Activate775, False, False]])
                c.set_parameters('ExpControl/scan/range/t/res',number_frames)
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
                   
        
                im.run(M_obj)
                time.sleep(time_wait) 
                    
        #        
            if Multi ==0:
                save_path = '{}{}{}'.format('D:/current data/','Overview','.msr')
                (im.measurement(im.measurement_names()[0])).save_as(save_path)
                im.close(im.measurement(im.measurement_names()[0])) 
                files = os.listdir('D:/current data/')
                files_txt = [i for i in files if i.endswith('.ptu')]
                n_files_txt = len(files_txt)
                    
                for ii in range(n_files_txt):
                    new_path = '{}{}{}{}'.format('D:/current data/testfolder/ptu_files/','measurement_',ii,'.ptu')
                    os.rename('{}{}'.format('D:/current data/',files_txt[ii]), '{}{}{}{}{}'.format('D:/current data/','Overview_','spot_', ii, '.ptu'))
                    
            else:
                save_path = '{}{}{}{}{}'.format('D:/current data/','Overview_',Pos,'_numberSPOTS_', number_peaks_new-1)
                os.makedirs(save_path)  
                (im.measurement(im.measurement_names()[1])).save_as('{}{}{}{}{}'.format(save_path,'/','Overview_', Pos,'.msr'))
                 
                files = os.listdir('D:/current data/')
                files_ptu = [i for i in files if i.endswith('.ptu')]
                files_dat = [i for i in files if i.endswith('.dat')]
                
                n_files_ptu = len(files_ptu)
                n_files_dat = len(files_dat)
                    
                for ii in range(n_files_ptu):
                    os.rename('{}{}'.format('D:/current data/',files_ptu[ii]), '{}{}{}{}{}{}{}'.format(save_path,'/','Overview_Pos_y', Pos, '_spot_', ii, '.ptu'))
                
                for ii in range(n_files_dat):
                    os.rename('{}{}'.format('D:/current data/',files_dat[ii]), '{}{}{}{}{}{}{}'.format(save_path,'/','Overview_Pos_y', Pos, '_spot_', files_dat[ii],'.dat'))
            
    else:
        print('Multirun not available! No connection')
        save_path = 'D:/current data/'
        print(save_path)
            
    return save_path
       
def SAVING(path,a):
    from docx import Document
    from docx.shared import Pt
    from docx.shared import Length
    from docx.shared import RGBColor
    from docx.enum.text import WD_LINE_SPACING
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches
    #import matplotlib as pp
    
    if a==0:
        #### functions #####
        def GT(m_name):
            import matplotlib as pp
            
            if m_name == 0:
                stk_names = meas.configuration(m_name).stack_names()
                stk = meas.stack(stk_names[0])
                pix_data = stk.data()
                im = np.mean(pix_data, axis=3)[0]
                fig = pp.figure(figsize=(6,6))
                pp.axis('off')
                pp.imshow(im, cmap ='hot' )
                pp.text(10, 30, 'Overview',color='white', fontsize=15, fontweight='bold')
                pp.savefig('D:/current data/testa.tiff',bbox_inches='tight')
                pp.close(fig)
                document.add_picture('D:/current data/testa.tiff', width=Inches(5.5))
            else:
                stk_names = meas.configuration(m_name).stack_names()
                stk_g1 = meas.stack(stk_names[0])
                stk_g2 = meas.stack(stk_names[2])
                stk_r1 = meas.stack(stk_names[1])
                stk_r2 = meas.stack(stk_names[3])
                
                pix_data_green = stk_g1.data() + stk_g2.data()
                pix_data_red = stk_r1.data() + stk_r2.data()
                im_g = np.mean(pix_data_green, axis=1)[0]
                im_r = np.mean(pix_data_red, axis=1)[0]
                
                    
                fig = pp.figure(figsize=(6,6))
                pp.subplot(221)
                pp.imshow(im_g, cmap ='hot' )
                pp.axis('off')
                pp.text(3, 10, '{}{}'.format(meas_names[m_name],': green'),color='white', fontsize=15, fontweight='bold')
                pp.subplot(222)
                pp.imshow(im_r, cmap ='hot' )
                pp.axis('off')
                pp.text(3, 10, '{}{}'.format(meas_names[m_name],': red'),color='white', fontsize=15, fontweight='bold')
                pp.savefig('D:/current data/testa.tiff',bbox_inches='tight')
                pp.close(fig)
                document.add_picture('D:/current data/testa.tiff', width=Inches(5.5))
                
                
        
        
        
        def test(dicta, levels):
                
                
                if type(dicta) == type(True) or type(dicta) == int or type(dicta) == list():
                    print('level0')
                else:
                    for i in dicta.keys():
                        style = document.styles['Normal']
                        paragraph_format = style.paragraph_format
                        
                        document.add_page_break()
                        p1_header = document.add_heading(str(i), level=1)
                        p1_header.paragraph_format.left_indent = Inches(0.0)
                        
                        if type(dicta[i]) == int or type(dicta[i]) == type(True) or type(dicta[i]) == str :#or type(dicta[i]) == list:
                            p1 = document.add_paragraph(str(dicta[i]))
                            p1.paragraph_format.left_indent = Inches(0.5)
                            
                        elif type(dicta[i]) == dict:
                            for ii in dicta[i].keys():
                                p2_header = document.add_heading(str(ii),level = 2)
                                p2_header.paragraph_format.left_indent = Inches(0.5)
                                if type(dicta[i][ii]) == int or type(dicta[i][ii]) == type(True) or type(dicta[i][ii]) == str or type(dicta[i][ii]) == list or type(dicta[i][ii]) == float:
                                    p2 = document.add_paragraph(str(dicta[i][ii]))
                                    p2.paragraph_format.left_indent = Inches(0.5)
                                    
                                else:
                                    for iii in dicta[i][ii].keys():
                                        p3_header = document.add_heading(str(iii),level = 3)
                                        p3_header.paragraph_format.left_indent = Inches(1)
                                        if type(dicta[i][ii][iii]) == int or type(dicta[i][ii][iii]) == type(True) or type(dicta[i][ii][iii]) == str or type(dicta[i][ii][iii]) == list or type(dicta[i][ii][iii]) == float:
                                               p3 = document.add_paragraph(str(dicta[i][ii][iii]))
                                               p3.paragraph_format.left_indent = Inches(1)
                                               
                                        else:
                                            for iiii in dicta[i][ii][iii].keys():
                                                p4_header = document.add_heading(str(iiii),level = 4)
                                                p4_header.paragraph_format.left_indent = Inches(2)
                                                p4 =document.add_paragraph(str(dicta[i][ii][iii][iiii]))
                                                p4.paragraph_format.left_indent = Inches(2)
                                                
        
        document = Document()
        
        im = specpy.Imspector()
        meas = im.measurement(im.measurement_names()[1])
        a = meas.parameter('')    
        document.add_heading('{}{}'.format('Measurement-Report:                                                ',a['Measurement']['MeasTime']), 0) 
        document.add_heading('Important Parameter', level=2)
        table = document.add_table(rows=6, cols=2)
        cell = table.cell(0,0)
        cell.text = 'Parameter'
        cell = table.cell(0,1)  
        cell.text = 'Value'
        
        meas_names = meas.configuration_names()
        
        document.add_page_break()
        for ii in range(len(meas_names)):
            GT(ii)    
        table.style = 'Light Grid Accent 1'
        test(a,0)                   
        document.save('{}{}'.format(path,'/Meas_protocol.docx'))
    
    else:
        print('saving works')
        document = Document()
        document.add_heading('this is just a test document')
        document.save('{}{}'.format(path,'/Meas_protocol.docx'))



       
def layout(path, root):
    
    root.title("Imspector Control Interface")
    root1 = tkinter.Frame(root, width = 350, height = 350, bg = 'grey')
    root1.grid()
   
    foldername= 'testfolder'                         #testfolder'#Type here the foldername
 
    image_ID = Image.open('Alphatubulin.tif')
    resized = image_ID.resize((300, 300), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(resized)
    colour = 'grey'
    txtcolour = 'white'
    space = 10
    framespacer= tkinter.Frame(root1, width = space, height = 2, bg = colour)
    
    
    frame_s1 = framespacer
    frame_s1.grid(row=0, column=0, sticky = 'n')
    frame_s2 = framespacer
    frame_s2.grid(row=0, column=3, sticky = 'n')
    frame_s3 = framespacer
    frame_s3.grid(row=4, column=0, sticky = 'n')
    frame_s4 = framespacer
    frame_s4.grid(row=4, column=3, sticky = 'n')
    frame_s5 = framespacer
    frame_s5.grid(row=2, column=1, sticky = 'n')
    frame_s6 = framespacer
    frame_s6.grid(row=2, column=2, sticky = 'n')
    frame_top = tkinter.Frame(root1, width = 300, height = 300, bg = colour)
    frame_top.grid(row=1, column=1)
    label = tkinter.Label(frame_top,image=photo)
    label.image = photo
    label.grid(row=0)
    
    frame_top2 = tkinter.Frame(root1, width = 300, height = 300, bg = colour)
    frame_top2.grid(row=3, column=1)
    frame_top2.grid_propagate(0)
    
    frame_top3 = tkinter.Frame(root1, width = 300, height = 300, bg = colour)
    frame_top3.grid(row= 1, column=2, sticky = 'n')
    
    frame_top4 = tkinter.Frame(root1, width = 300, height = 300, bg = colour)
    frame_top4.grid(row=3, column=2, sticky = 'n')

    f = pp.figure(figsize=(2,2), dpi=150, edgecolor='k')
    canvas = FigureCanvasTkAgg(f, master = frame_top3)
    canvas.get_tk_widget().grid(row=0, column=0)
    
    labtext_1 = tkinter.Label(frame_top4,width = 300, height = 200,bg = colour)
    labtext_1.grid(row=1, column = 0, sticky = 's')
    
    T = tkinter.Text(frame_top4, height=10, width=37)
    T.grid()

    style = ttk.Style()
    settings = {"TNotebook.Tab": 
                   {"configure": {"padding": [1, 2], "background": colour , "font" : ('Sans','9','bold')}, 
                    "map": {"background": [("selected", "red"),  ("active", "#fc9292")], 
                            "foreground": [("selected", "pink"), ("active", "white")], 
                            "expand": [("selected", [1, 1, 1, 1])]} },
                "TFrame": 
                   {"configure": {"padding": [0, 2], "background": colour } },
                "TNotebook": 
                   {"configure": {"padding": [1, 2], "background": colour } }}
                
                       
                       
    style.theme_create('JHB', parent="alt", settings=settings)
    style.theme_create('JHB2', parent="alt", settings=settings)
    style.theme_use('JHB')

    nb = ttk.Notebook(frame_top2, width = 7000)
    nb.pressed_index = None
    nb.grid(row=0, column=0, sticky='NESW')
    
    page1 = ttk.Frame(nb)
    nb.add(page1, text='Overview')
    
    page2 = ttk.Frame(nb)
    nb.add(page2, text='ROI_select')
    
    page3 = ttk.Frame(nb)   
    nb.add(page3, text='Powerseries')
    
    page4 = ttk.Frame(nb)
    nb.add(page4, text='Pinholeseries')
    
    
    
    
    frame_spacer_01 = tkinter.Frame(page2, width = 50, height = 1, background =colour)
    frame_spacer_01.grid(row=0, column=0, sticky = 'w')
    
    frame_5 = tkinter.Frame(frame_spacer_01, width = 150, height = 110, background = colour)
    frame_5.grid(row=0, column=0, sticky = 'wn')
    frame_5.grid_propagate (False)
    
    frame_6 = tkinter.Frame(frame_spacer_01, width = 150, height = 180, background = colour)
    frame_6.grid(row=1, column=0, sticky = 'wn')
    frame_6.grid_propagate (False)
    
    frame_7 = tkinter.Frame(frame_spacer_01, width = 150, height = 180, background = colour)
    frame_7.grid(row=1, column=1, sticky = 'wn')
    frame_7.grid_propagate (False)
    
    frame_8 = tkinter.Frame(frame_spacer_01, width = 150, height = 110, background = colour)
    frame_8.grid(row=0, column=1, sticky = 'wn')
    frame_8.grid_propagate (False)
    
    laser= tkinter.Label(page3, text=' Laser:', height = 1, foreground= txtcolour, background=colour)
    laser.grid(row = 0, column = 0)
    laser_01 = tkinter.Entry(page3,width = 8)
    laser_01.insert(tkinter.END, '561')
    laser_01.grid(row = 0,column = 1)
    
    
#### Overview tab ######   

    space= tkinter.Label(page1, text='', height = 1 , foreground= txtcolour, background=colour)
    space.grid(row = 0, column = 0 , sticky = tkinter.W+tkinter.N)
    
    dwell_overview= tkinter.Label(page1, text=' Dwelltime [us]:', height = 1 , foreground= txtcolour, background=colour)
    dwell_overview.grid(row = 1, column = 0 , sticky = tkinter.W + tkinter.N)
    dwell_overview_value = tkinter.Entry(page1,width = 3, foreground= 'white', bg = 'grey')
    dwell_overview_value.insert(tkinter.END, '10')
    dwell_overview_value.grid(row = 1, column = 1, sticky = 'w')
    
    pxsize_overview= tkinter.Label(page1, text=' Pixelsize [nm]:', height = 1,foreground= txtcolour, background=colour)
    pxsize_overview.grid(row = 2, column = 0, sticky = 'wn')
    pxsize_overview_value = tkinter.Entry(page1, width = 3, foreground= 'white', bg = 'grey')
    pxsize_overview_value.insert(tkinter.END, '200')
    pxsize_overview_value.grid(row = 2, column = 1, sticky = 'w')
    
    ROIsize_overview= tkinter.Label(page1, text=' ROIsize [um]:', height = 1,foreground= txtcolour, background=colour)
    ROIsize_overview.grid(row = 3, column = 0, sticky = 'wn')
    ROIsize_overview_value = tkinter.Entry(page1,width = 3, foreground= 'white', bg = 'grey')
    ROIsize_overview_value.insert(tkinter.END, '70')
    ROIsize_overview_value.grid(row = 3, column = 1, sticky = 'w')
    
    frames_overview= tkinter.Label(page1, text=' # Frames:           ', height = 1, foreground= txtcolour, background=colour)
    frames_overview.grid(row = 4, column = 0, sticky = 'wn')
    frames_overview_value = tkinter.Entry(page1,width = 3, foreground= 'white', bg = 'grey')
    frames_overview_value.insert(tkinter.END, '3')
    frames_overview_value.grid(row = 4, column = 1, sticky = 'w')

    laser_overview= tkinter.Label(page1, text=' Laser| power[%]:', height = 1, foreground= txtcolour, background=colour)
    laser_overview.grid(row = 5, column = 0, sticky = 'wn')
    laser_overview_entry = tkinter.Entry(page1,width = 8, foreground= 'white', bg = 'grey')
    laser_overview_entry.insert(tkinter.END, '640')
    laser_overview_entry.grid(row = 5,column = 1, sticky = 'w') 
    
    laser_overview_value = tkinter.Entry(page1, width = 8, foreground= 'white', bg = 'grey')
    laser_overview_value.insert(tkinter.END, '5')
    laser_overview_value.grid(row = 5, column = 2, sticky = 'w')
    
#########
### ROI_select 

    
    laservalue= tkinter.Label(page3, text=' value:', height = 1, foreground= txtcolour, background=colour)
    laservalue.grid(row = 0, column = 2, sticky = 'w')
    laser_value_01 = tkinter.Entry(page3,width = 8)
    laser_value_01.insert(tkinter.END, '50')
    laser_value_01.grid(row = 0, column = 3, sticky = 'w')
    
    laser_STED= tkinter.Label(page3, text=' STED-Laser:', height = 1, foreground= txtcolour, background=colour)
    laser_STED.grid(row = 1, column = 0, sticky = 'w')
    laser_STED_01 = tkinter.Entry(page3,width = 8)
    laser_STED_01.insert(tkinter.END, '775')
    laser_STED_01.grid(row = 1, column = 1, sticky = 'w')
    
    laser_STEDvalue= tkinter.Label(page3, text=' value:', height = 1, foreground= txtcolour, background=colour)
    laser_STEDvalue.grid(row = 1, column = 2)
    laser_STEDvalue_01 = tkinter.Entry(page3,width = 20)
    laser_STEDvalue_01.insert(tkinter.END, '50, 50, 20,32')
    laser_STEDvalue_01.grid(row = 1, column = 3)
    
    dwell= tkinter.Label(frame_5, text=' Dwelltime [us]:', height = 1 , foreground= txtcolour, background=colour)
    dwell.grid(row = 0, column = 0 , sticky = 'wn')
    dwell_01 = tkinter.Entry(frame_5,width = 3 )
    dwell_01.insert(tkinter.END, '5')
    dwell_01.grid(row = 0, column = 1, sticky = 'w')
    
    pxsize= tkinter.Label(frame_5, text=' Pixelsize [nm]:', height = 1,foreground= txtcolour, background=colour)
    pxsize.grid(row = 1, column = 0, sticky = 'wn')
    pxsize_01 = tkinter.ENTRY(frame_5, width = 3)
    pxsize_01.insert(tkinter.END, '10')
    pxsize_01.grid(row = 1, column = 1, sticky = 'w')
    
    ROIsize= tkinter.Label(frame_5, text=' ROIsize [um]:', height = 1,foreground= txtcolour, background=colour)
    ROIsize.grid(row = 2, column = 0, sticky = 'wn')
    ROIsize_01 = tkinter.ENTRY(frame_5,width = 3)
    ROIsize_01.insert(tkinter.END, '1')
    ROIsize_01.grid(row = 2, column = 1, sticky = 'w')
    
    frames= tkinter.Label(frame_5, text=' # Frames:           ', height = 1, foreground= txtcolour, background=colour)
    frames.grid(row = 3, column = 0, sticky = 'wn')
    frames_01 = tkinter.ENTRY(frame_5,width = 3)
    frames_01.insert(tkinter.END, '61')
    frames_01.grid(row = 3, column = 1, sticky = 'w')
    
    
    MultiRUN= tkinter.Label(frame_8, text=' Focus search:', height = 1, foreground= txtcolour, background=colour)
    MultiRUN.grid(row = 0, column = 2, sticky = 'w')
    MultiRUN_01 = tkinter.ENTRY(frame_8,width = 8)
    MultiRUN_01.insert(tkinter.END, '3')
    MultiRUN_01.grid(row = 0, column = 3, sticky = 'w')
    
    MultiRUN_s= tkinter.Label(frame_8, text='', height = 1, foreground= txtcolour, background=colour)
    MultiRUN_s.grid(row = 0, column = 0, sticky = 'w')
    
    Autofocus= tkinter.Label(frame_8, text=' Autofocus:', height = 1 ,foreground= txtcolour, background=colour)
    Autofocus.grid(row = 2, column = 2, sticky = tkinter.W)
    var13 = tkinter.IntVar()
    Autofocus_01 = tkinter.Checkbutton(frame_8,  variable = var13, background =colour)
    Autofocus_01.grid(row=2, column = 3, sticky = tkinter.W)
    
    QFS= tkinter.Label(page1, text=' Autofocus:', height = 1 ,foreground= txtcolour, background=colour)
    QFS.grid(row = 2, column = 2, sticky = tkinter.W)
    var14 = tkinter.IntVar()
    QFS_01 = tkinter.Checkbutton(page1,  variable = var14, background =colour)
    QFS_01.grid(row=2, column = 3, sticky = tkinter.W)
    
    Circle= tkinter.Label(frame_8, text=' Circle:', height = 1 ,foreground= txtcolour, background=colour)
    Circle.grid(row = 3, column = 2, sticky = tkinter.W)
    var15 = tkinter.IntVar(value=0)
    Circle_01 = tkinter.Checkbutton(frame_8,  variable = var15, background =colour)
    Circle_01.grid(row=3, column = 3, sticky = tkinter.W)

    
    
    L485= tkinter.Label(frame_6, text=' L485:', height = 1 ,foreground= txtcolour, background=colour)
    L485.grid(row = 0, column = 0, sticky = tkinter.W)
    var1 = tkinter.IntVar()
    L485_01 = tkinter.Checkbutton(frame_6,  variable = var1, background =colour)
    L485_01.grid(row=0, column = 1, sticky = tkinter.W)
    
    L485_02= tkinter.Label(frame_6, text=' L485:', height = 1 ,foreground= txtcolour, background=colour)
    L485_02.grid(row = 0, column = 0, sticky = tkinter.W)
    var7 = tkinter.IntVar()
    L485_02_ = tkinter.Checkbutton(frame_6,  variable = var7, background =colour)
    L485_02_.grid(row=0, column = 2, sticky=tkinter.W)
    
    L518= tkinter.Label(frame_6, text=' L518:', height = 1 ,foreground= txtcolour, background=colour)
    L518.grid(row = 1, column = 0, sticky = 'wn')
    var2 = tkinter.IntVar()
    L518_01 = tkinter.Checkbutton(frame_6,  variable = var2, background =colour)
    L518_01.grid(row=1, column = 1, sticky=tkinter.W)
    
    L518_02= tkinter.Label(frame_6, text=' L518:', height = 1,foreground= txtcolour, background=colour)
    L518_02.grid(row = 1, column = 0, sticky = 'wn')
    var8 = tkinter.IntVar()
    L518_02_ = tkinter.Checkbutton(frame_6,  variable = var8, background =colour)
    L518_02_.grid(row=1, column = 2, sticky=tkinter.W)
    
    L561= tkinter.Label(frame_6, text=' L561:', height = 1,foreground= txtcolour, background=colour)
    L561.grid(row = 2, column = 0, sticky = 'wn')
    var3 = tkinter.IntVar()
    L561_01 = tkinter.Checkbutton(frame_6,  variable = var3, background =colour)
    L561_01.grid(row=2, column = 1, sticky=tkinter.W)
    
    L561_02= tkinter.Label(frame_6, text=' L561:', height = 1,foreground= txtcolour, background=colour)
    L561_02.grid(row = 2, column = 0, sticky = 'wn')
    var9 = tkinter.IntVar()
    L561_02_ = tkinter.Checkbutton(frame_6,  variable = var9, background =colour)
    L561_02_.grid(row=2, column = 2, sticky=tkinter.W)
    
    L640= tkinter.Label(frame_6, text=' L640:', height = 1,foreground= txtcolour, background=colour)
    L640.grid(row = 3, column = 0, sticky = 'wn')
    var4 = tkinter.IntVar()
    L640_01 = tkinter.Checkbutton(frame_6, variable = var4, background =colour)
    L640_01.grid(row=3, column = 1, sticky=tkinter.W)
    
    L640_02= tkinter.Label(frame_6, text=' L640:', height = 1,foreground= txtcolour, background=colour)
    L640_02.grid(row = 3, column = 0, sticky = 'wn')
    var10 = tkinter.IntVar()
    L640_02_ = tkinter.Checkbutton(frame_6,  variable = var10, background =colour)
    L640_02_.grid(row=3, column = 2, sticky = 'wn')
    
    L595= tkinter.Label(frame_6, text=' L595:', height = 1,foreground= txtcolour, background=colour)
    L595.grid(row = 4, column = 0, sticky = 'wn')
    var5 = tkinter.IntVar()
    L595_01 = tkinter.Checkbutton(frame_6,  variable = var5, background =colour)
    L595_01.grid(row=4, column = 1, sticky = 'wn')
    
    L595_02= tkinter.Label(frame_6, text=' L595:', height = 1,foreground= txtcolour, background=colour)
    L595_02.grid(row = 4, column = 0, sticky = 'wn')
    var11 = tkinter.IntVar()
    L595_02_ = tkinter.Checkbutton(frame_6,  variable = var11, background =colour)
    L595_02_.grid(row=4, column = 2, sticky = 'wn')
    
    L775= tkinter.Label(frame_6, text=' L775:', height = 1,foreground= txtcolour, background=colour)
    L775.grid(row = 5, column = 0, sticky = 'wn')
    var6 = tkinter.IntVar()
    L775_01 = tkinter.Checkbutton(frame_6, variable = var6, background =colour)
    L775_01.grid(row=5, column = 1, sticky = 'wn')
    
    L775_02= tkinter.Label(frame_6, text=' L775:', height = 1,foreground= txtcolour, background=colour)
    L775_02.grid(row = 5, column = 0, sticky = 'wn')
    var12 = tkinter.IntVar()
    L775_02_ = tkinter.Checkbutton(frame_6,  variable = var12, background =colour)
    L775_02_.grid(row=5, column = 2, sticky = 'wn')
    
    L485_value= tkinter.Label(frame_6, height = 1,foreground= txtcolour, background=colour)
    L485_value.grid(row = 2, column = 0, sticky = 'wn')
    L485_value_01 = tkinter.ENTRY(frame_6,width = 3)
    L485_value_01.insert(tkinter.END, '0')
    L485_value_01.grid(row = 0, column = 5, sticky = 'en')
    
    L518_value= tkinter.Label(frame_6, height = 1,foreground= txtcolour, background=colour)
    L518_value.grid(row = 1, column = 0, sticky = 'wn')
    L518_value_01 = tkinter.ENTRY(frame_6,width = 3)
    L518_value_01.insert(tkinter.END, '0')
    L518_value_01.grid(row = 1, column = 5, sticky = 'en')
    
    L561_value= tkinter.Label(frame_6, height = 1,foreground= txtcolour, background=colour)
    L561_value.grid(row = 2, column = 0, sticky = 'wn')
    L561_value_01 = tkinter.ENTRY(frame_6,width = 3)
    L561_value_01.insert(tkinter.END, '50')
    L561_value_01.grid(row = 2, column = 5, sticky = 'en')
    
    L640_value= tkinter.Label(frame_6, height = 1,foreground= 'black', background=colour)
    L640_value.grid(row = 3, column = 0, sticky = 'wn')
    L640_value_01 = tkinter.ENTRY(frame_6,width = 3)
    L640_value_01.insert(tkinter.END, '3')
    L640_value_01.grid(row = 3, column = 5, sticky = 'en')
    
    L595_value= tkinter.Label(frame_6, height = 1,foreground= 'black', background= colour)
    L595_value.grid(row = 4, column = 0, sticky = 'wn')
    L595_value_01 = tkinter.ENTRY(frame_6,width = 3)
    L595_value_01.insert(tkinter.END, '0')
    L595_value_01.grid(row = 4, column = 5, sticky = 'en')
    
    L775_value= tkinter.Label(frame_6, height = 1,foreground= 'black', background=colour)
    L775_value.grid(row = 5, column = 0, sticky = 'wn')
    L775_value_01 = tkinter.ENTRY(frame_6,width = 3)
    L775_value_01.insert(tkinter.END, '50')
    L775_value_01.grid(row = 5, column = 5, sticky = 'en')
    
    THRES_value= tkinter.Label(frame_6, height = 1,foreground= 'black', background=colour)
    THRES_value.grid(row = 6, column = 0, sticky = 'wn')
    THRES_value_01 = tkinter.ENTRY(frame_6,width = 3)
    THRES_value_01.insert(tkinter.END, '0')
    THRES_value_01.grid(row = 6, column = 5, sticky = 'en')
    
    a=0
    
    out = [frame_top, tkinter.Label, frame_top2, frame_top3, frame_top4, labtext_1, T, style, nb, page1, page2, page3, page4, frame_spacer_01,frame_5, frame_6, frame_7, frame_8, a, colour, foldername, pxsize_01,ROIsize_01, dwell_01, frames_01, var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15, L485_value_01,L518_value_01,L561_value_01, L640_value_01, L595_value_01, L775_value_01, MultiRUN_01, laser_overview_value, laser_overview_entry, frames_overview_value,ROIsize_overview_value, dwell_overview_value, pxsize_overview_value] 
    return out    