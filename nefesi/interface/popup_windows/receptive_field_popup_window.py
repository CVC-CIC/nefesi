import tkinter as tk# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3

try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk
import numpy as np
from nefesi.util.general_functions import clean_widget
from nefesi.util.interface_plotting import ORDER
from PIL import ImageTk, Image
import os
from math import ceil


class ReceptiveFieldPopupWindow(object):
    def __init__(self, master, image_complete, image_cropped,x_len,y_len):
        self.window=Toplevel(master)
        self.window.title('Receptive Field of '+str(x_len)+'X'+str(y_len))
        self.complete_image_frame = Frame(master=self.window)
        Label(master=self.complete_image_frame, text="Original image").pack(side=TOP)
        self.cropped_image_frame = Frame(master=self.window)
        Label(master=self.cropped_image_frame, text="Receptive field").pack(side=TOP)
        self.put_image(master=self.complete_image_frame, img=image_complete)
        self.put_image(master=self.cropped_image_frame, img=image_cropped)
        self.complete_image_frame.pack(side=LEFT)
        self.cropped_image_frame.pack(side=RIGHT)
    def put_image(self, master, img):
        panel = Label(master=master, image=img)
        panel.image = img
        panel.pack(side=BOTTOM, fill=BOTH, expand=True)