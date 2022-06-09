# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:32:04 2022

@author: Abberior_admin
"""
import specpy
import threading
import tkinter
import time
#%%
im = specpy.Imspector()
msr = im.active_measurement()
#%%
im.run(msr)
#%%
im.pause(msr)
#%%
im.stop(msr)
#%%

def stopRunthreadAfter(timeout):
    time.sleep(timeout)
    if thread.is_alive():
        im = specpy.Imspector()
        msr = im.active_measurement()
        im.pause(msr)
        print("thread id %i stopped" % id(thread))
def stopThread(thread):
    im = specpy.Imspector()
    msr = im.active_measurement()
    im.pause(msr)
    print("thread id %i stopped" % id(thread))
def hello()):
    print("hello_world")
#%%
while True:
    timeout = 1
    im = specpy.Imspector()
    msr = im.active_measurement()
    runthread = threading.Thread(target = im.run, args = (msr,) )
    print("starting runthread with id %i" % id(runthread))
    runthread.start()
    #waitthread gets the pointer from runthread and halts measurement if that measurement is still alive after timeout
    #in case runthread finishes early, another runthread and associated waitthread is made. In this way multiple waitthreads are on at the same time.
    #it would be better if the timer gets reset each time. How could I do this?
    # waitthread = threading.Thread(target = stopThreadAfter, args = (timeout, runthread))
    # waitthread.start()

# runthread.join(5)
# if runthread.is_alive():
#     im.stop(msr)
#     print("stopped the running")
