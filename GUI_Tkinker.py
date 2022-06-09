# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:48:13 2018

@author: Jan-Hendrik Budde

Info   : The code is mainly splitted into three part.
            1. Define path as directory where all selff created modules are stored
            2. This script contains the main code for GUI. The main layout is imported from LAYOUT.py 
               which contains the coarse appearance of GUI.
            3. The function which are defined here ONLY defines the Buttons in GUI
         Due to dynamic of code some parameter need to be defined in this script. They should not be transferred elsewhere.

"""
# import packages and modules####
#from FUNCTION import *
import FUNCTION as func
import tkinter as tk
import specpy
import numpy as np
from functools import partial
import GUISpotFinding as spotFinding
import time
import warnings
import os
import threading
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
        self.dataout = dataout #used as a path for saving stuff
        self.make_layout()
        #self.foldername = 'testfolder' # why is this needed?
        button_1 = tk.Button(self.frame_buttons, width = 9,            text = 'Connect',  activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self.Connect,anchor = 'w'                  ).grid(row = 0, column = 0)
        button_2 = tk.Button(self.frame_buttons, width = 9,             text = 'Overview', activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self.makeOverview, anchor = 'w').grid(row = 0, column = 1)
        button_3 = tk.Button(self.frame_buttons, width = 9,             text = 'FindPeak', activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self.Findpeak                              ).grid(row = 1,column = 1)
        button_4 = tk.Button(self.frame_buttons, width = 9,             text = 'Run',      activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self.Run_meas                 ).grid(row = 0, column = 3)
        button_5 = tk.Button(self.frame_buttons, width = 9,height =1,  text = 'Abort',    activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self.Abort                            ).grid(row = 0, column = 2)
        button_6 = tk.Button(self.frame_buttons, width = 9, height =1,  text = 'resetAbort',    activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self.reset_abort                           ).grid(row = 1, column = 2)
        button_7 = tk.Button(self.frame_buttons, width = 9, height =1,  text = 'timeRun',  activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self.timeRun                         ).grid(row = 0, column = 4)
        button_8 = tk.Button(self.frame_buttons, width = 9, height =1,  text = 'MultiRun', activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self._MultiRun_meas                         ).grid(row = 1, column = 3)
        button_9 = tk.Button(self.frame_buttons, width = 9, height =1,  text = 'MultiTime', activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self._MultiTime                         ).grid(row = 1, column = 4)
        button_10 = tk.Button(self.frame_buttons, width = 14, height =1,  text = 'Set Positions', activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self.setPositions                         ).grid(row = 0, column = 5)
        button_11 = tk.Button(self.frame_buttons, width = 14, height =1,  text = 'Run Positions', activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self._runPositions                         ).grid(row = 1, column = 5)
        button_12 = tk.Button(self.frame_buttons, width = 14, height =1,  text = 'Multi Positions', activebackground= 'green',font = ('Sans','9','bold'),activeforeground= 'red', command = self._MultiPositions                         ).grid(row = 1, column = 6)
        
        scale_01_label= tk.Label(self.frame_7, text='Thres:', height = 1,foreground= 'white', background =self.color, font = ('Sans','9','bold'))
        scale_01_label.grid(row = 0, column = 0, sticky = tk.W+tk.N)
        scale_01 = tk.Scale(self.frame_7, from_=0, to=200, showvalue=1, background =self.color)
        scale_01.set(20)
        scale_01.bind("<ButtonRelease-1>", self.RELEASE)
        scale_01.grid(row = 1, column = 0, sticky = tk.W+tk.N)
        self.scale_01 = scale_01
        
        scale_02_label= tk.Label(self.frame_7, text='<area', height = 1,\
                                 foreground= 'white', background =self.color,\
                                     font = ('Sans','9','bold'))
        scale_02_label.grid(row = 0, column = 1, sticky = tk.W+tk.N)
        scale_02 = tk.Scale(self.frame_7, from_=0, to=2000, showvalue=1, background =self.color)
        scale_02.set(200)
        scale_02.bind("<ButtonRelease-1>", self.RELEASE)
        scale_02.grid(row = 1, column = 1, sticky = tk.W+tk.N)
        self.scale_02 = scale_02
        
        scale_03_label= tk.Label(self.frame_7, text='Rmin', height = 1,\
                                 foreground= 'white', background =self.color, \
                                     font = ('Sans','9','bold') )
        scale_03_label.grid(row = 0, column = 2, sticky = tk.W+tk.N)
        scale_03 = tk.Scale(self.frame_7, from_=0, to=200, showvalue=1, background = self.color)
        scale_03.set(0)
        scale_03.bind("<ButtonRelease-1>", self.RELEASE)
        scale_03.grid(row = 1, column = 2, sticky = tk.W+tk.N)
        self.scale_03 = scale_03
        
    def make_layout(self): #bind function to this class
        func.layout(self)
    def Connect(self):
        func.Connect(self)
    def makeOverview(self):
        self.y_coarse_offset = 0
        func.makeOverview(self)
    def Findpeak(self): # alias
        try:
            self.allpeaks, self.smoothimage = spotFinding.findPeaks(self.overview, smooth_sigma = 1)
            self.scale_01.config(to = np.max(self.smoothimage)*10)
        except RecursionError:
            self.T.delete('1.0', tk.END) 
            self.T.insert(tk.END, 'could not find peaks, try making an overview image first\n')        
        
    def RELEASE(self, scaleval):
        #when binding this function to a scale release, automatically the value is passed
        #it is given this scaleval dummy to avoid an error
        bglevel = float(self.scale_01.get()) / 10
        minarea = 0
        #minarea = float(self.scale_02.get()) / 10
        maxarea = float(self.scale_02.get())
        Rmin = float(self.scale_03.get()) / 10
        goodpeaks, counts, vals = spotFinding.filterPeaks(self.smoothimage, self.allpeaks, \
                                bglevel = bglevel, minarea = minarea, \
                                Rmin = Rmin, maxarea = maxarea)
        #this was implemented ad-hoc and might break easily
        countsvalsout = os.path.join(self.dataout, "peakproperties.txt")
        np.savetxt(countsvalsout, np.array([counts, vals]).T, delimiter = '\t', \
                   header = "peak area\tpeak height" )
        spotFinding.plotpeaks(self.smoothimage, goodpeaks, isshow = True)
        shape = self.smoothimage.shape
        #axis are swapped
        self.goodpeaks = goodpeaks #too lazy to do the converting, saving both
        self.roi_xs = goodpeaks[:,1] - shape[1] / 2
        self.roi_ys = goodpeaks[:,0] - shape[0] / 2 #center of image is 0
        
    def Run_meas(self):  
        """this function splits off Run meas into a separate thread, such
        that the GUI remains responsive and the run may be aborted"""
        self.runthread = threading.Thread(target=func.Run_meas, args = (self,) )
        self.runthread.start()  
        
    def _MultiRun_meas(self):
        """this function splits off MultiRun meas into a separate thread, such
        that the GUI remains responsive and the run may be aborted"""
        self.runthread = threading.Thread(target = self.MultiRun_meas, args = () )
        self.runthread.start()  
        
    def MultiRun_meas(self):
        #raise NotImplementedError("the global positioning is not working, the command to imspector does not seem to work")

        runs = int(self.multirun.get()) ### define the number of Rois you want to scan
        roisize = float(self.ROIsize_overview_value.get())*1e-06# in meter
        
        coarse_y_start = func.getYOffset()

        for ii in range(runs):
            #get  current measurement handles
            im = specpy.Imspector()
            try:
                msr = im.active_measurement()
            except:
                self.T.insert(tk.END, 'creating new measurement\n')
                msr=im.create_measurement()
            config = msr.active_configuration()
            
            # set desired coarse stage positions in y [m]
            #this does not work maybe, because each time the measurement is deleted, so it must be set globally
            ypos = coarse_y_start + 1.1 * roisize * ii
            config.set_parameters('ExpControl/scan/range/offsets/coarse/y/g_off', ypos) 
            print('Overview=',ypos)
            
            self.makeOverview()
            #find peak runs in two steps, first Findpeak, than RELEASE
            self.Findpeak()
            self.RELEASE(1)#1 is a dummy value to avoid an error
            #this function does the work
            func.Run_meas(self)
            #_Run_meas creates self.runthread
            #the next overview must start after the first run has ended
            #calling runthread.join() halts further execution untill runthread has terminated
            #self.runthread.join()
            if self.abort_run:
                break
            #Imspector does not like it when measurements are too fast after one another
            time.sleep(1)
            #print('sample=',y_add,'#peaks=',number_peaks_new, 'timewait=',(time_wait * number_peaks_new +1))
            #func.SAVING(save_path, a)
    def _MultiTime(self):
        """this function splits off MultiRun meas into a separate thread, such
        that the GUI remains responsive and the run may be aborted"""
        self.runthread = threading.Thread(target = self.MultiTime, args = () )
        self.runthread.start()  
        
    def MultiTime(self):
        #raise NotImplementedError("the global positioning is not working, the command to imspector does not seem to work")

        runs = int(self.multirun.get()) ### define the number of Rois you want to scan
        roisize = float(self.ROIsize_overview_value.get())*1e-06# in meter
        
        coarse_y_start = func.getYOffset()

        for ii in range(runs):
            #adding some random time delays to hope it prevents crashing
            time.sleep(1)
            #get  current measurement handles
            im = specpy.Imspector()
            try:
                msr = im.active_measurement()
            except:
                self.T.insert(tk.END, 'creating new measurement\n')
                msr=im.create_measurement()
            config = msr.active_configuration()
            
            # set desired coarse stage positions in y [m]
            #this does not work maybe, because each time the measurement is deleted, so it must be set globally
            ypos = coarse_y_start + 1.1 * roisize * ii
            config.set_parameters('ExpControl/scan/range/offsets/coarse/y/g_off', ypos) 
            print('Overview=',ypos)
            
            self.makeOverview()
            #adding some random time delays to hope it prevents crashing
            time.sleep(1)
            #find peak runs in two steps, first Findpeak, than RELEASE
            self.Findpeak()
            #adding some random time delays to hope it prevents crashing
            time.sleep(1)
            self.RELEASE(1)#1 is a dummy value to avoid an error
            #function that does the work
            func.timeRun(self)
            #_Run_meas creates self.runthread
            #the next overview must start after the first run has ended
            #calling runthread.join() halts further execution untill runthread has terminated
            #self.runthread.join()
            if self.abort_run:
                break
            #Imspector does not like it when measurements are too fast after one another
            time.sleep(1)
            #print('sample=',y_add,'#peaks=',number_peaks_new, 'timewait=',(time_wait * number_peaks_new +1))
            #func.SAVING(save_path, a)
    def Abort(self):
        self.abort_run = True
    def reset_abort(self):
        self.abort_run = False
    def timeRun(self):
        self.y_coarse_offset = 0 #to change later, badly implemented
        func._timeRun(self)
    def setPositions(self):
        #open a new window where a list of positions can be set.
        func.setPositions(self)
    def _runPositions(self):
        """this function splits off runPositions meas into a separate thread, such
        that the GUI remains responsive and the run may be aborted"""
        self.runthread = threading.Thread(target = func.runPositions, \
                                          args = (self,) )
        self.runthread.start()
    def _MultiPositions(self):
        self.runthread = threading.Thread(target = self.MultiPositions, \
                                          args = () )
        self.runthread.start()
    def MultiPositions(self):
        runs = int(self.multirun.get())
        for run in range(runs):
            self.T.delete("1.0", tk.END)
            self.T.insert(tk.END, 'measuring all positions, loop %i\n' % run)
            func.runPositions(self)
            if self.abort_run:
                break
        pass

#Type here the directory of GUI python file is stored: path = 'C:/Users/Abberior_admin/Desktop/GUI/'
dataout = r'D:\current data' 
root= tk.Tk()
abberiorControl = AbberiorControl(root, dataout)#.pack(side="top", fill="both", expand=True)
root.mainloop()