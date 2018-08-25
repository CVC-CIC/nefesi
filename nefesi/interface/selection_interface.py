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
        self.make_analysis_frame = Frame(master=self.window)
        self.set_make_analysis_frame(master=self.make_analysis_frame)
        self.make_analysis_frame.pack(pady=3)
        self.calc_indexs_frame = Frame(master=self.window)
        self.set_make_indexs_calc_frame(master=self.calc_indexs_frame)
        self.calc_indexs_frame.pack(pady=3)
        self.select_action_frame = Frame(master=self.window, borderwidth=1)
        self.set_select_action_frame(master=self.select_action_frame)
        self.select_action_frame.pack(pady=3)
        #self.plot_general_index(index='class')
        #self.plot_general_index(index='class')
        #self.plot_general_index(index='class')
        #self.plot_general_index(index='orientation')
        self.window.mainloop()

    def set_select_action_frame(self, master):
        label = Label(master=master, text="Visualize an existent analysis")
        button = Button(master=master, text="Select file", command=self._on_click_visualize_analysis_button)
        label.pack(side=LEFT)
        button.pack(side=RIGHT)

    def set_make_analysis_frame(self, master):
        label = Label(master=master, text="Make script for do an analysis")
        button = Button(master=master, text="Select Parameters", command=self._on_click_make_analysis_button)
        label.pack(side=LEFT)
        button.pack(side=RIGHT)

    def set_make_indexs_calc_frame(self, master):
        label = Label(master=master, text="Make script for calc indexs")
        button = Button(master=master, text="Select Parameters", command=self._on_click_make_indexs_calcs_button)
        label.pack(side=LEFT)
        button.pack(side=RIGHT)

    def ask_for_file(self, title="Select file", type='obj'):
        filename = filedialog.askopenfilename(title=title,
                                              filetypes=((type, '*.' + type), ("all files", "*.*")))
        return filename


    def _on_click_visualize_analysis_button(self):
        network_data_file = self.ask_for_file(title="Select NetworkData object (.obj file)")
        if network_data_file != '':
            model_file = self.ask_for_file(title="Select model (.h5 file)", type="h5")
            model_file = model_file if model_file != '' else None
            network_data = NetworkData.load_from_disk(file_name=network_data_file, model_file=model_file)
            self.window.destroy()
            Interface(network_data=network_data)

    def _on_click_make_analysis_button(self):
        self.window.destroy()
        MakeAnalysisInterface()


    def _on_click_make_indexs_calcs_button(self):
        self.window.destroy()
        CalcIndexsInterface()