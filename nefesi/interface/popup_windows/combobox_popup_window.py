
try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk


class ComboboxPopupWindow(object):
    def __init__(self, master,values, text, default = None):
        self.value = -1 #Returned value if user clicks on X

        self.top=Toplevel(master)
        self.top.title('Select Value')
        self.default = default
        self.text_label= ttk.Label(self.top, text=text)
        self.text_label.pack()
        self.ok_button = ttk.Button(self.top, text='Ok', command=self.cleanup)
        self.combo_box = ttk.Combobox(master=self.top, values=values, state='readonly', width=15, justify=CENTER)
        idx_of_default = 0 if default is None else values.index(default)
        self.combo_box.current(idx_of_default)
        self.combo_box.pack()
        self.ok_button.pack()
    def cleanup(self):
        self.value=self.combo_box.get()
        self.top.destroy()
