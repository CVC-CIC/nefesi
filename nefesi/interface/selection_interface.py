import os
from os.path import relpath
import dill as pickle
from ..interface.calc_indexes_interface import CalcIndexesInterface

STATES = ['init']
MAX_PLOTS_VISIBLES_IN_WINDOW = 4
MAX_VALUES_VISIBLES_IN_LISTBOX = 6

try:
    from tkinter import *
    from tkinter import filedialog
    import tkinter.ttk as ttk
except ImportError:
    from Tkinter import *
    from Tkinter import ttk

from ..network_data import NetworkData
from ..interface.interface import Interface
from ..interface.make_analysis_interface import MakeAnalysisInterface

STYLE = 'clam'
class SelectionInterface():
    def __init__(self):
        self.window = Tk()
        ttk.Style().theme_use(STYLE)
        self.window.title("Nefesi")
        #TOP Part with general info of viewing and some setteables
        self.set_make_analysis_frame()
        self.set_make_indexes_calc_frame()
        self.set_select_action_frame()

        #self.plot_general_index(index='orientation')
        self.window.mainloop()


    def set_make_analysis_frame(self):
        label = ttk.Label(master=self.window, text="Prepare Analysis: ")
        button = ttk.Button(master=self.window, text="Select Parameters",width=15, command=self._on_click_make_analysis_button)
        label.grid (row=0, column=0, sticky=E, pady=(6,3), padx=(6,1))
        button.grid(row=0, column=1, sticky=W, pady=(6,3), padx=(1,6))

    def set_make_indexes_calc_frame(self):
        label = ttk.Label(master=self.window, text="Prepare Indexes Calc.: ")
        button = ttk.Button(master=self.window, text="Select Parameters",width=15, command=self._on_click_make_indexes_calcs_button)
        label.grid (row=1, column=0, sticky=E, pady=(3,3), padx=(6,1))
        button.grid(row=1, column=1, sticky=W, pady=(3,3), padx=(1,6))

    def set_select_action_frame(self):
        label = ttk.Label(master=self.window, text="Visualize Analysis: ")
        button = ttk.Button(master=self.window, text="Select File", width=15, command=self._on_click_visualize_analysis_button)
        label.grid (row=2, column=0, sticky=E, pady=(3,6), padx=(6,1))
        button.grid(row=2, column=1, sticky=W, pady=(3,6), padx=(1,6))




    def ask_for_file(self, title="Select File", type='obj', initialdir=None, initialfile=None):
        filename = filedialog.askopenfilename(title=title,
                                              filetypes=((type, '*.' + type), ("all files", "*.*")),
                                              initialdir=initialdir, initialfile=initialfile)
        #filename = relpath(filename)
        return filename


    def _on_click_visualize_analysis_button(self):
        cfg_file = os.path.join(os.path.dirname(sys.argv[0]),"init.cfg")
        if os.path.isfile(cfg_file):
            # Getting back the objects:
            with open(cfg_file, 'rb') as f:
                init_folder, init_network = pickle.load(f)
        else:
            init_folder, init_network = ('/data/114-1/users/nefesi/Data', '')

        network_data_file = self.ask_for_file(title="Select Nefesi object (.obj)",
                                              initialdir=init_folder, initialfile=init_network )
        if network_data_file != '':
            name = os.path.splitext(os.path.basename(network_data_file))[0]
            model_file = self.ask_for_file(title="Select Model (.h5)", type="h5", initialfile=name+'.h5')
            model_file = model_file if model_file != '' else None
            network_data = NetworkData.load_from_disk(file_name=network_data_file, model_file=model_file)
            init_folder = os.path.dirname(network_data_file)
            init_network = os.path.basename(network_data_file)

            # Saving the objects:
            with open(cfg_file, 'wb') as f:
                pickle.dump([init_folder, init_network ], f)

            self.window.destroy()
            Interface(network_data=network_data, window_style = STYLE)

    def _on_click_make_analysis_button(self):
        self.window.destroy()
        MakeAnalysisInterface(window_style=STYLE)


    def _on_click_make_indexes_calcs_button(self):
        self.window.destroy()
        CalcIndexesInterface(window_style=STYLE)