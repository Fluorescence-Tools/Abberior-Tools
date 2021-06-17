# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:48:13 2018

@author: Jan-Hendrik Budde

Info   : The code is mainly splitted into three part.
            1. Define path as directory where all selff created modules are stored
            2. This script contains the main code for GUI. The main layout is imported from LAYOUT.py 
               which contains the coarse appearance of GUI.
            3. The function which are defined here ONLY defines the Buttons in GUI
         Due to dynamic of code some parameter need to be defined in this script. They should not be transfered elsewhere.
"""
# import packages and modules####
from FUNCTION import *
import FUNCTION as func
import tkinter

# Define path to main GUI directory
path = 'C:/Users/Abberior_admin/Desktop/GUI/'    #Type here the directory of GUI python file is stored: path = 'C:/Users/Abberior_admin/Desktop/GUI/'

# Create main graphical interface
root= tkinter.Tk()
frame_top, label, frame_top2, frame_top3, frame_top4, labtext_1, T, style, nb, page1, page2, page3, page4, frame_spacer_01,frame_5, frame_6, frame_7, frame_8, a, colour, foldername, pxsize_01,ROIsize_01, dwell_01, frames_01, var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15, L485_value_01,L518_value_01,L561_value_01, L640_value_01, L595_value_01, L775_value_01, MultiRUN_01,  laser_overview_value, laser_overview_entry, frames_overview_value,ROIsize_overview_value, dwell_overview_value, pxsize_overview_value = func.layout(path, root)

# Here the Buttons and the corresponding function are defined (Buttons with action)

def RELEASE(scale_03_value):
    Findpeak()
       
def SET_VALUE():
    global pixelsize, Roisize, dwelltime, frame_number, act485, act518, act561, act640, act595, act775, act485_02, act518_02, act561_02, act640_02, act595_02, act775_02, act_Autofocus, act_QFS, circle
    pixelsize, Roisize, dwelltime, frame_number, act485, act518, act561, act640, act595, act775, act485_02, act518_02, act561_02, act640_02, act595_02, act775_02, act_Autofocus, act_QFS, circle = func.set_value(pxsize_01,ROIsize_01, dwell_01, frames_01, var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14, var15)
   
    
def Connect():
    global a
    a = func.Connect(T)


def Overview(Multi, Pos, a):
    global roi_size, data, r1, r2,  pixelsize_global, circle
    circle = 0
    data, roi_size,r1, r2, pixelsize_global = func.Overview(Multi, Pos, path, foldername,  scale_01, frame_top, T,  laser_overview_value, laser_overview_entry, frames_overview_value, ROIsize_overview_value, dwell_overview_value, pxsize_overview_value, circle)    
     
    
def Findpeak():
    photo, number_peaks_new, CO = func.Findpeak(path, foldername, scale_01, scale_02, scale_03, roi_size, frame_top, T, frame_top3, a, circle, pixelsize_global)


def powerseries():
    func.powerseries()
      
    
def pinholeseries():
    pinholevector = [1,2,3,4,50]
    laser = 3
    func.pinholeseries(pinholevector, laser)
    
    
def Run_meas(Multi, Pos):
    global save_path
    mm = Multi
    save_path = func.Run_meas(pixelsize, Roisize, dwelltime, frame_number, act485, act518, act561, act640, act595, act775, act485_02, act518_02, act561_02, act640_02, act595_02, act775_02, act_Autofocus, act_QFS, L485_value_01,L518_value_01,L561_value_01, L640_value_01, L595_value_01, L775_value_01, T, mm, Pos, pixelsize_global)
    
    
def MultiRun_meas():
    import time
    runs = int(MultiRUN_01.get()) ### define the number of Rois you want to scan
    delaytime = 1 # here we slow down the loop in [s] in order to make sure all Rois will be identified                                    
    z_position =  5e-08 #float(pixelsize)*1e-09         # in meter
    x_pixelsize = float(pixelsize)*1e-09        # in meter
    y_pixelsize = float(pixelsize)*1e-09        # in meter
    z_pixelsize = float(pixelsize)*1e-09        # in meter
    RZ =    float(Roisize)*1e-06            # in meter    
    Dwelltime= float(dwelltime)*1e-06         # in seconds 
    number_frames = float(frame_number)                                
    time_wait = 1

    if a==0:    
        im = specpy.Imspector()
        meas= im.active_measurement()
        y_off = meas.parameters('ExpControl/scan/range/offsets/coarse/y/g_off') # current stage position in x [m]
        y_add = y_off + 1.1*roi_size * np.linspace(0,runs-1, runs) ### initial position 
        
        for i in y_add:
            print('Overview=',i)
            Overview(1,i,a)
            Findpeak()
            time.sleep(1)
            Run_meas(1,i) ### 1 = TRUE for MUltirun
            
            #print('sample=',y_add,'#peaks=',number_peaks_new, 'timewait=',(time_wait * number_peaks_new +1))
            #func.SAVING(save_path, a)
            im.close(im.measurement(im.measurement_names()[1]))
    else:
        print('Multi run not available! No connection')
        func.SAVING(save_path, a)

  
button_1 = Button(labtext_1, width = 10,            text = 'Connect',  activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = Connect,anchor = 'w'                  ).grid(row = 0, column = 0)
button_2 = Button(labtext_1, width = 9,             text = 'Overview', activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = partial(Overview,0,0, a), anchor = 'w').grid(row = 0, column = 1)
button_3 = Button(labtext_1, width = 9,             text = 'FindPeak', activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = Findpeak                              ).grid(row = 0,column = 2)
button_4 = Button(labtext_1, width = 9,             text = 'Run',      activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = partial(Run_meas,0,0)                 ).grid(row = 0, column = 3)
button_5 = Button(labtext_1, width = 10,height =1,  text = 'Set value',activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = SET_VALUE                             ).grid(row = 1, column = 0)
button_6 = Button(labtext_1, width = 9, height =1,  text = 'Power',    activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = powerseries                           ).grid(row = 1, column = 1)
button_7 = Button(labtext_1, width = 9, height =1,  text = 'Pinhole',  activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = pinholeseries                         ).grid(row = 1, column = 2)
button_8 = Button(labtext_1, width = 9, height =1,  text = 'MultiRun', activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = MultiRun_meas                         ).grid(row = 1, column = 3)


scale_01_label= Label(frame_7, text='Thres:', height = 1,foreground= 'white', background =colour, font = ('Sans','9','bold'))
scale_01_label.grid(row = 0, column = 0, sticky = W+N)
scale_01 = Scale(frame_7, from_=0.5, to=5, showvalue=1, background =colour)
scale_01_value = scale_01.get()
scale_01.set(20)
scale_01.bind("<ButtonRelease-1>", RELEASE)
scale_01.grid(row = 1, column = 0, sticky = W+N)

scale_02_label= Label(frame_7, text='Rmin:', height = 1,foreground= 'white', background =colour, font = ('Sans','9','bold'))
scale_02_label.grid(row = 0, column = 1, sticky = W+N)
scale_02 = Scale(frame_7, from_=1, to=50, showvalue=1, background =colour)
scale_02_value = scale_02.get()
scale_02.set(20)
scale_02.bind("<ButtonRelease-1>", RELEASE)
scale_02.grid(row = 1, column = 1, sticky = W+N)

scale_03_label= Label(frame_7, text='Rmax:', height = 1,foreground= 'white', background =colour, font = ('Sans','9','bold') )
scale_03_label.grid(row = 0, column = 2, sticky = W+N)
scale_03 = Scale(frame_7, from_=1, to=50, showvalue=1, background =colour)
scale_03_value = scale_03.get()
scale_03.set(10)
scale_03.bind("<ButtonRelease-1>", RELEASE)
scale_03.grid(row = 1, column = 2, sticky = W+N)


root.mainloop()