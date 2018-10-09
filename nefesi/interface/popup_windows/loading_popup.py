
try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk


class LoadingPopup(object):
    def __init__(self, master):
        self.top=Toplevel(master)
        self.top.title('Loading')
        self.text_label=Label(self.top, text='Loading...')
        self.text_label.pack(side=TOP)