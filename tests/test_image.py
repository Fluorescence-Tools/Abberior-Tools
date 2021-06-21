# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:27:50 2021

@author: Abberior_admin
"""
import tkinter
from PIL import Image, ImageTk


image_ID = Image.open(r'C:\Users\Abberior_admin\Desktop\Abberior-Tools\Alphatubulin.tif')
resized = image_ID.resize((300, 300), Image.ANTIALIAS)

colour = 'grey'

root= tkinter.Tk()
photo = ImageTk.PhotoImage(resized, master = root)
root1 = tkinter.Frame(root, width = 350, height = 350, bg = 'grey')
root1.grid()
frame_top = tkinter.Frame(root1, width = 300, height = 300, bg = colour)
frame_top.grid(row=1, column=1)
label = tkinter.Label(frame_top,image=photo)
label.image = photo

#root.mainloop()