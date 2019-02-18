
try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk


class ComboboxPopupWindow(object):
    def __init__(self, master,values, text):
        self.value = -1 #Returned value if user clicks on X

        self.top=Toplevel(master)

        self.top.title('Select Value')

        self.text_label=Label(self.top, text=text)
        self.text_label.pack()
        self.ok_button = Button(self.top, text='Ok', command=self.cleanup)
        self.combo_box = ttk.Combobox(master=self.top, values=values, state='readonly', width=15, justify=CENTER)
        self.combo_box.current(0)
        self.combo_box.pack()
        self.ok_button.pack()
    def cleanup(self):
        self.value=self.combo_box.get()
        self.top.destroy()
