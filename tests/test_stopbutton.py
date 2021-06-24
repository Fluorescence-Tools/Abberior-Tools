# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 14:19:57 2021

@author: Abberior_admin
"""
import tkinter as tk    
import time  
import threading  
  
switch = True  
root = tk.Tk()  
  
def blink(*args):  

 thread = threading.Thread(target=run, args = args)  
 thread.start()  
 
def run(message):  
    while (switch == True):  
     print(message)  
     time.sleep(0.5)  
     if switch == False:  
      break  
  
def switchon():    
 global switch  
 switch = True  
 print ('switch on'   )
 blink('hi there')    
        
def switchoff():    
 print ('switch off'  )
 global switch  
 switch = False      
        
def kill():    
 root.destroy()    

def clear():
    nchars = len(entry.get())
    entry.delete(0, nchars)
        
onbutton = tk.Button(root, text = "Blink ON", command = switchon)    
onbutton.pack()    
offbutton =  tk.Button(root, text = "Blink OFF", command = switchoff)    
offbutton.pack()    
killbutton = tk.Button(root, text = "EXIT", command = kill)    
killbutton.pack()   
clearbutton = tk.Button(root, text = "Clear", command = clear)    
clearbutton.pack()  
entry = tk.Entry(root) 
entry.insert(tk.END, 'some value')

entry.pack()
        
root.mainloop()