
try:
    from tkinter import *
    from tkinter import ttk
    from tkinter import filedialog
except ImportError:
    from Tkinter import *
    from Tkinter import ttk


MAX_VALUES_VISIBLES_IN_LISTBOX = 6
ALL_INDEX_NAMES = ['symmetry', 'orientation', 'color', 'class', 'object', 'part']

class SelectIndexWindow():
    def __init__(self, master, actual_indexes):
        self.top = Toplevel(master)
        self.top.title("Select the indexes to use")
        self.new_indexes = -1
        ttk.Label(master=self.top, text="Indexes to take into account: ").pack()
        self.ok_button = Button(self.top, text='Ok', command=self.cleanup)
        self.boolean_vars = []
        for index in ALL_INDEX_NAMES:
            checkbox_frame= ttk.Frame(master=self.top)
            self.boolean_vars.append(self.set_check_box(checkbox_frame, text=index, value =index in actual_indexes))
            checkbox_frame.pack()

        self.ok_button.pack(pady=(8, 5), ipadx=10)

    def cleanup(self):
        self.new_indexes = [name for value, name in zip(self.boolean_vars,ALL_INDEX_NAMES) if value.get()]
        self.top.destroy()

    def set_check_box(self,master,text, value = False):
        var = BooleanVar(master=master,value=value)
        checkbox = ttk.Checkbutton(master=master, text=text, variable=var)
        checkbox.pack()
        return var

