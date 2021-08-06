# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 13:06:40 2021

@author: Abberior_admin
"""
import specpy
import numpy as np
import threading
import time

#%%
def hello_world():
    print('hello world')
def counttoNumber(N = 3):
    for i in range(N):
        print('counting %i' % i)
        time.sleep(1)
#%%
thread1 = threading.Thread(target = hello_world)
thread2 = threading.Thread(target = counttoNumber)
thread1.start()
thread2.start()
thread2.join()
print('are both threads finished?')