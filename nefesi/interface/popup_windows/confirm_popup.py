
try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk


class ConfirmPopup(object):
    def __init__(self, master,text='Are you sure?'):
        self.value = False
        self.top=Toplevel(master)
        self.top.title('Confirm selection')
        self.text_label=Label(self.top, text=text)
        self.text_label.pack(side=TOP)
        buttons_frame = Frame(master = self.top)
        buttons_frame.pack(side=BOTTOM)
        self.ok_button = Button(buttons_frame, text='Confirm', command=self.cleanup_ok)
        self.cancel_button = Button(buttons_frame, text='Cancel', command=self.cleanup_cancel)
        self.ok_button.pack(side=LEFT,pady=(8, 5),padx=3, ipadx=5)
        self.cancel_button.pack(side=RIGHT,pady=(8, 5),padx=3, ipadx=5)

    def cleanup_ok(self):
        self.value=True
        self.top.destroy()
    def cleanup_cancel(self):
        self.value=False
        self.top.destroy()
