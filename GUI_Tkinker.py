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
#from FUNCTION import *
import FUNCTION as func
import tkinter as tk
import specpy
import numpy as np
from functools import partial
import time
#edits by NV
#this code has numerous serious issues
#it works, but it contains several features that are considered bad practice
#it had wildcard import - now fixed
#namespace names are overwritten with variable names, e.g.
#   import impspector as im; im = Image.image()
#a consequence is that in each function namespaces are reloaded
#global variables are used at random places
#button handles and variables are passed in long lists of function returns
#   making it easy to mistake the order.
#several pieces of code are repeated multiple times - they should go in functions
#there are a lot of pieces whose function seem unnesecary.
#
#as a solution one might:
#build the app into a class, solving the need for global variables and
#   passing so many variables
#going through the whole of the code to much-out unneeded parts.

# Define path to main GUI directory


# Create main graphical interface

#frame_topleft, frame_top2, frame_topright, frame_top4, labtext_1, T, frame_spacer_01,frame_5, frame_6, frame_7, frame_8, a, colour, foldername, pxsize_01,ROIsize_01, dwell_01, frames_01, var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15, L485_value_01,L518_value_01,L561_value_01, L640_value_01, L595_value_01, L775_value_01, MultiRUN_01,  laser_overview_value, laser_overview_entry, frames_overview_value,ROIsize_overview_value, dwell_overview_value, pxsize_overview_value = func.layout(path, root)
class AbberiorControl(tk.Tk):
    def __init__(self, parent, dataout, *args, **kwargs):
        #tk.Frame.__init__(self, parent, *args, **kwargs)
        self.abort_run = False
        self.parent = parent
        self.dataout = dataout #don't need in principle
        self.make_layout()
        self.foldername = 'testfolder' # why is this needed?
        button_1 = tk.Button(self.frame_buttons, width = 10,            text = 'Connect',  activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self.Connect,anchor = 'w'                  ).grid(row = 0, column = 0)
        button_2 = tk.Button(self.frame_buttons, width = 9,             text = 'Overview', activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = partial(self.Overview,0,0), anchor = 'w').grid(row = 0, column = 1)
        # button_3 = tk.Button(self.frame_buttons, width = 9,             text = 'FindPeak', activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = Findpeak                              , state = tk.DISABLED).grid(row = 0,column = 2)
        button_4 = tk.Button(self.frame_buttons, width = 9,             text = 'Run',      activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = partial(self.Run_meas,0,0)                 ).grid(row = 0, column = 3)
        button_5 = tk.Button(self.frame_buttons, width = 10,height =1,  text = 'Abort',    activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self.Abort                            ).grid(row = 1, column = 0)
        button_6 = tk.Button(self.frame_buttons, width = 9, height =1,  text = 'resetAbort',    activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self.reset_abort                           ).grid(row = 1, column = 1)
        # button_7 = tk.Button(self.frame_buttons, width = 9, height =1,  text = 'Pinhole',  activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = pinholeseries                         , state = tk.DISABLED).grid(row = 1, column = 2)
        button_8 = tk.Button(self.frame_buttons, width = 9, height =1,  text = 'MultiRun', activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self.MultiRun_meas                         ).grid(row = 1, column = 3)

        scale_01_label= tk.Label(self.frame_7, text='Thres:', height = 1,foreground= 'white', background =self.color, font = ('Sans','9','bold'))
        scale_01_label.grid(row = 0, column = 0, sticky = tk.W+tk.N)
        scale_01 = tk.Scale(self.frame_7, from_=0.5, to=5, showvalue=1, background =self.color)
        #scale_01_value = scale_01.get()
        scale_01.set(20)
        scale_01.bind("<ButtonRelease-1>", self.RELEASE)
        scale_01.grid(row = 1, column = 0, sticky = tk.W+tk.N)
        self.scale_01 = scale_01
        
        scale_02_label= tk.Label(self.frame_7, text='Rmin:', height = 1,\
                                 foreground= 'white', background =self.color,\
                                     font = ('Sans','9','bold'))
        scale_02_label.grid(row = 0, column = 1, sticky = tk.W+tk.N)
        scale_02 = tk.Scale(self.frame_7, from_=1, to=50, showvalue=1, background =self.color)
        #scale_02_value = scale_02.get()
        scale_02.set(20)
        scale_02.bind("<ButtonRelease-1>", self.RELEASE)
        scale_02.grid(row = 1, column = 1, sticky = tk.W+tk.N)
        self.scale_02 = scale_02
        
        scale_03_label= tk.Label(self.frame_7, text='Rmax:', height = 1,\
                                 foreground= 'white', background =self.color, \
                                     font = ('Sans','9','bold') )
        scale_03_label.grid(row = 0, column = 2, sticky = tk.W+tk.N)
        scale_03 = tk.Scale(self.frame_7, from_=1, to=50, showvalue=1, background = self.color)
        #scale_03_value = scale_03.get()
        scale_03.set(50)
        scale_03.bind("<ButtonRelease-1>", self.RELEASE)
        scale_03.grid(row = 1, column = 2, sticky = tk.W+tk.N)
        self.scale_03 = scale_03
        
    def make_layout(self): #bind function to this class
        func.layout(self)
    def Connect(self):
        func.Connect(self)
    def Overview(self,Multi, Pos):
        func.Overview(self, Multi, Pos)
    def Findpeak(self): # alias
        func.Findpeak(self)
    def RELEASE(self, scaleval):
        #when binding this function to a scale release, automatically the value is passed
        #it is given this scaleval dummy to avoid an error
        self.Findpeak() #findpeak is currently  a dummy setting 10 random numbers
    def Run_meas(self, Multi, Pos):
        self.y_coarse_offset = 0
        func._Run_meas(self)
    def MultiRun_meas(self):

        runs = int(self.multirun.get()) ### define the number of Rois you want to scan
        roisize  =    float(self.ROIsize.get())*1e-06            # in meter    
    
        #this re-connection is not needed in principle
        im = specpy.Imspector()
        meas= im.active_measurement()
        y_off = meas.parameters('ExpControl/scan/range/offsets/coarse/y/g_off') # current stage position in x [m]
        y_add = y_off + 1.1 * roisize * np.linspace(0,runs-1, runs) ### initial position 
        
        for i in y_add:
            print('Overview=',i)
            self.Overview(1,i)
            self.Findpeak()
            #time.sleep(1)
            self.Run_meas(1,i) ### 1 = TRUE for MUltirun
            if self.abort_run:
                break
            #print('sample=',y_add,'#peaks=',number_peaks_new, 'timewait=',(time_wait * number_peaks_new +1))
            #func.SAVING(save_path, a)
            im.close(im.measurement(im.measurement_names()[1]))
    def Abort(self):
        self.abort_run = True
    def reset_abort(self):
        self.abort_run = False



        
# Here the Buttons and the corresponding function are defined (Buttons with action)

# =============================================================================
# def RELEASE(scale_03_value):
#     Findpeak()
#        
# def SET_VALUE():
#     global pixelsize, Roisize, dwelltime, frame_number, act485, act518, act561, act640, act595, act775, act485_02, act518_02, act561_02, act640_02, act595_02, act775_02, act_Autofocus, act_QFS, circle
#     pixelsize, Roisize, dwelltime, frame_number, act485, act518, act561, act640, act595, act775, act485_02, act518_02, act561_02, act640_02, act595_02, act775_02, act_Autofocus, act_QFS, circle = func.set_value(pxsize_01,ROIsize_01, dwell_01, frames_01, var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14, var15)
#    
#     
# def Connect():
#     global a
#     a = func.Connect(T)
# 
# 
# def Overview(Multi, Pos, a):
#     global roi_size, data, r1, r2,  pixelsize_global, circle
#     circle = 0
#     data, roi_size,r1, r2, pixelsize_global = func.Overview(Multi, Pos, path, foldername,  scale_01, frame_topleft, T,  laser_overview_value, laser_overview_entry, frames_overview_value, ROIsize_overview_value, dwell_overview_value, pxsize_overview_value, circle)    
#      
#     
# def Findpeak():
#     photo, number_peaks_new, CO = func.Findpeak(path, foldername, scale_01, scale_02, scale_03, roi_size, frame_topleft, T, frame_top3, a, circle, pixelsize_global)
# 
# 
# def powerseries():
#     func.powerseries()
#       
#     
# def pinholeseries():
#     pinholevector = [1,2,3,4,50]
#     laser = 3
#     func.pinholeseries(pinholevector, laser)
#     
#     
# def Run_meas(Multi, Pos):
#     global save_path
#     mm = Multi
#     save_path = func.Run_meas(pixelsize, Roisize, dwelltime, frame_number, act485, act518, act561, act640, act595, act775, act485_02, act518_02, act561_02, act640_02, act595_02, act775_02, act_Autofocus, act_QFS, L485_value_01,L518_value_01,L561_value_01, L640_value_01, L595_value_01, L775_value_01, T, mm, Pos, pixelsize_global)
#     
#     
# def MultiRun_meas():
#     import time
#     runs = int(MultiRUN_01.get()) ### define the number of Rois you want to scan
#     delaytime = 1 # here we slow down the loop in [s] in order to make sure all Rois will be identified                                    
#     z_position =  5e-08 #float(pixelsize)*1e-09         # in meter
#     x_pixelsize = float(pixelsize)*1e-09        # in meter
#     y_pixelsize = float(pixelsize)*1e-09        # in meter
#     z_pixelsize = float(pixelsize)*1e-09        # in meter
#     RZ =    float(Roisize)*1e-06            # in meter    
#     Dwelltime= float(dwelltime)*1e-06         # in seconds 
#     number_frames = float(frame_number)                                
#     time_wait = 1
# 
#     if a==0:    
#         im = specpy.Imspector()
#         meas= im.active_measurement()
#         y_off = meas.parameters('ExpControl/scan/range/offsets/coarse/y/g_off') # current stage position in x [m]
#         y_add = y_off + 1.1*roi_size * np.linspace(0,runs-1, runs) ### initial position 
#         
#         for i in y_add:
#             print('Overview=',i)
#             Overview(1,i,a)
#             Findpeak()
#             time.sleep(1)
#             Run_meas(1,i) ### 1 = TRUE for MUltirun
#             
#             #print('sample=',y_add,'#peaks=',number_peaks_new, 'timewait=',(time_wait * number_peaks_new +1))
#             #func.SAVING(save_path, a)
#             im.close(im.measurement(im.measurement_names()[1]))
#     else:
#         print('Multi run not available! No connection')
#         func.SAVING(save_path, a)
# 

# 
# scale_01_label= tk.Label(frame_7, text='Thres:', height = 1,foreground= 'white', background =colour, font = ('Sans','9','bold'))
# scale_01_label.grid(row = 0, column = 0, sticky = tk.W+tk.N)
# scale_01 = tk.Scale(frame_7, from_=0.5, to=5, showvalue=1, background =colour)
# scale_01_value = scale_01.get()
# scale_01.set(20)
# scale_01.bind("<ButtonRelease-1>", RELEASE)
# scale_01.grid(row = 1, column = 0, sticky = tk.W+tk.N)
# 
# scale_02_label= tk.Label(frame_7, text='Rmin:', height = 1,foreground= 'white', background =colour, font = ('Sans','9','bold'))
# scale_02_label.grid(row = 0, column = 1, sticky = tk.W+tk.N)
# scale_02 = tk.Scale(frame_7, from_=1, to=50, showvalue=1, background =colour)
# scale_02_value = scale_02.get()
# scale_02.set(20)
# scale_02.bind("<ButtonRelease-1>", RELEASE)
# scale_02.grid(row = 1, column = 1, sticky = tk.W+tk.N)
# 
# scale_03_label= tk.Label(frame_7, text='Rmax:', height = 1,foreground= 'white', background =colour, font = ('Sans','9','bold') )
# scale_03_label.grid(row = 0, column = 2, sticky = tk.W+tk.N)
# scale_03 = tk.Scale(frame_7, from_=1, to=50, showvalue=1, background =colour)
# scale_03_value = scale_03.get()
# scale_03.set(50)
# scale_03.bind("<ButtonRelease-1>", RELEASE)
# scale_03.grid(row = 1, column = 2, sticky = tk.W+tk.N)
# =============================================================================
#Type here the directory of GUI python file is stored: path = 'C:/Users/Abberior_admin/Desktop/GUI/'
dataout = r'D:\current data' 
root= tk.Tk()
abberiorControl = AbberiorControl(root, dataout)#.pack(side="top", fill="both", expand=True)
root.mainloop()