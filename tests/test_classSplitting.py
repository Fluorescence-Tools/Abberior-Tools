# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 11:02:57 2021

@author: Abberior_admin
"""
import _test_classSplitting
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
colour = 'grey'
class MyGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        #frame_top = tk.Frame(self, width = 400, height = 400)
        #frame_top.grid(row=1, column=1)
        self.value = 2
        #frame_top2 = tk.Frame(self, width = 400, height = 300, bg = colour)
        #frame_top2.grid(row=3, column=1)
        #frame_top2.grid_propagate(0)
        
        frame_top3 = tk.Frame(self, width = 400, height = 400, bg = colour)
        frame_top3.grid(row= 1, column=2, sticky = 'n')
        
        frame_top4 = tk.Frame(self, width = 400, height = 300, bg = colour)
        frame_top4.grid(row=3, column=2, sticky = 'n')
    
        f = plt.figure(figsize=(2,2), dpi=150, edgecolor='k')
        canvas = FigureCanvasTkAgg(f, master = frame_top3)
        canvas.get_tk_widget().grid(row=0, column=0)
        
        labtext_1 = tk.Label(frame_top4,width = 300, height = 200,bg = colour)
        labtext_1.grid(row=1, column = 0, sticky = 's')
        
        T = tk.Text(frame_top4, height=10, width=37)
        T.grid()
    def print_value(self):
        _test_classSplitting.print_value(self)
        
myGUI = MyGUI()
myGUI.print_value()
myGUI.mainloop()

#%%
import tkinter as tk 
 
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", 
        "Friday", "Saturday", "Sunday"] 
MODES = [tk.SINGLE, tk.BROWSE, tk.MULTIPLE, tk.EXTENDED] 
 
class ListApp(tk.Tk): 
    def __init__(self): 
        super().__init__() 
        self.list = tk.Listbox(self)  
        self.list.insert(0, *DAYS) 
        self.print_btn = tk.Button(self, text="Print selection", 
                                   command=self.print_selection) 
        self.btns = [self.create_btn(m) for m in MODES] 
 
        self.list.pack() 
        self.print_btn.pack(fill=tk.BOTH) 
        for btn in self.btns: 
            btn.pack(side=tk.LEFT) 
 
    def create_btn(self, mode): 
        cmd = lambda: self.list.config(selectmode=mode) 
        return tk.Button(self, command=cmd, 
                         text=mode.capitalize()) 
 
    def print_selection(self): 
        selection = self.list.curselection() 
        print([self.list.get(i) for i in selection]) 
 
if __name__ == "__main__": 
    app = ListApp() 
    app.mainloop() 