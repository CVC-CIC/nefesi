import os
from os.path import relpath

from ..interface.calc_indexs_interface import CalcIndexsInterface

STATES = ['init']
MAX_PLOTS_VISIBLES_IN_WINDOW = 4
MAX_VALUES_VISIBLES_IN_LISTBOX = 6

try:
    from tkinter import *
    from tkinter import ttk
    from tkinter import filedialog
except ImportError:
    from Tkinter import *
    from Tkinter import ttk

from ..network_data import NetworkData
from ..interface.interface import Interface
from ..interface.make_analysis_interface import MakeAnalysisInterface

class SelectionInterface():
    def __init__(self):
        self.window = Tk()
        self.window.title("Nefesi")
        #TOP Part with general info of viewing and some setteables
        self.set_make_analysis_frame()
        self.set_make_indexs_calc_frame()
        self.set_select_action_frame()

        #self.plot_general_index(index='orientation')
        self.window.mainloop()


    def set_make_analysis_frame(self):
        label = Label(master=self.window, text="Make script for do an analysis")
        button = Button(master=self.window, text="Select Parameters", command=self._on_click_make_analysis_button)
        label.grid (row=0, column=0, sticky=E, pady=(6,3), padx=(6,1))
        button.grid(row=0, column=1, sticky=W, pady=(6,3), padx=(1,6))

    def set_make_indexs_calc_frame(self):
        label = Label(master=self.window, text="Make script for calc indexs")
        button = Button(master=self.window, text="Select Parameters", command=self._on_click_make_indexs_calcs_button)
        label.grid (row=1, column=0, sticky=E, pady=(3,3), padx=(6,1))
        button.grid(row=1, column=1, sticky=W, pady=(3,3), padx=(1,6))

    def set_select_action_frame(self):
        label = Label(master=self.window, text="Visualize an existent analysis")
        button = Button(master=self.window, text="Select file", command=self._on_click_visualize_analysis_button)
        label.grid (row=2, column=0, sticky=E, pady=(3,6), padx=(6,1))
        button.grid(row=2, column=1, sticky=W, pady=(3,6), padx=(1,6))




    def ask_for_file(self, title="Select file", type='obj', initialdir=None, initialfile=None):
        filename = filedialog.askopenfilename(title=title,
                                              filetypes=((type, '*.' + type), ("all files", "*.*")),
                                              initialdir=initialdir, initialfile=initialfile)
        #filename = relpath(filename)
        return filename


    def _on_click_visualize_analysis_button(self):
        network_data_file = self.ask_for_file(title="Select NetworkData object (.obj file)", initialdir = '/home/eric/Nefesi/Data/WithImageNet', initialfile='vgg16Copy.obj')
        if network_data_file != '':
            model_file = self.ask_for_file(title="Select model (.h5 file)", type="h5", initialfile='vgg16.h5')
            model_file = model_file if model_file != '' else None
            network_data = NetworkData.load_from_disk(file_name=network_data_file, model_file=model_file)
            last_dir_pos = network_data_file.rfind(os.path.sep)
            network_data.save_path = network_data_file[:last_dir_pos]
            network_data.default_file_name = network_data_file[last_dir_pos+1:network_data_file.rfind('.')]
            self.window.destroy()
            Interface(network_data=network_data)

    def _on_click_make_analysis_button(self):
        self.window.destroy()
        MakeAnalysisInterface()


    def _on_click_make_indexs_calcs_button(self):
        self.window.destroy()
        CalcIndexsInterface()