from ..evaluation_scripts.calculate_indexes import CalculateIndexes, ALL_INDEX_NAMES
from .popup_windows.confirm_popup import ConfirmPopup
from os.path import relpath

try:
    from tkinter import *
    from tkinter import ttk
    from tkinter import filedialog
except ImportError:
    from Tkinter import *
    from Tkinter import ttk
import dill as pickle


MAX_VALUES_VISIBLES_IN_LISTBOX = 6

class CalcIndexesInterface():
    def __init__(self, window_style = 'default'):
        self.window = Tk()
        ttk.Style().theme_use(window_style)
        self.window.title("Nefesi - Prepare Index Calculation")
        self.network_data_file = None
        self.model_file = None
        self.verbose =False
        self.lstbox_last_selection = [0]
        self.select_parameters_frame = ttk.Frame(master=self.window, borderwidth=1)
        self.set_parameters_frame(master=self.select_parameters_frame)
        self.set_index_listbox(self.select_parameters_frame)
        self.set_verbose_check_box(self.select_parameters_frame)
        self.select_parameters_frame.pack()
        self.ok_button = ttk.Button(self.window, text='Ok', command=self.cleanup)
        self.ok_button['state'] = 'disabled'
        self.ok_button.pack(pady=(8, 5), ipadx=10)
        self.set_footers(master=self.window)
        self.window.mainloop()

    def cleanup(self):
        if self.user_confirm():
            network_data_file = self.network_data_file
            model_file = self.model_file
            verbose = self.verbose.get()
            if len(self.lstbox_last_selection) == 1 and self.lstbox_last_selection[0] == 0:
                sel_indexes = ALL_INDEX_NAMES
            else:
                sel_indexes = [ALL_INDEX_NAMES[i-1] for i in self.lstbox_last_selection]
            self.window.destroy()
            indexes_eval = CalculateIndexes(network_data_file=network_data_file, model_file=model_file,
                                           sel_indexes=sel_indexes, verbose=verbose)
            with open('../nefesi/evaluation_scripts/indexes_config.cfg', 'wb') as f:
	            pickle.dump(indexes_eval, f)

    def set_index_listbox(self, master):
        label = ttk.Label(master=master, text="Indexes to Calculate: ")
        label.grid(row=2, column=0, sticky=E, pady=2, padx=(1, 2))
        index_listbox_frame = ttk.Frame(master=master)
        scrollbar = ttk.Scrollbar(master=index_listbox_frame, orient="vertical")
        self.index_to_calc_lstbox = Listbox(master=index_listbox_frame, selectmode=EXTENDED, yscrollcommand=scrollbar.set,
                                            height=MAX_VALUES_VISIBLES_IN_LISTBOX, width=20)
        scrollbar.config(command=self.index_to_calc_lstbox.yview)
        self.index_to_calc_lstbox.pack(side=LEFT)
        scrollbar.pack(side=RIGHT, fill="y")
        self.index_to_calc_lstbox.delete(0,END)
        self.index_to_calc_lstbox.insert(END, 'all')
        for item in ALL_INDEX_NAMES:
            self.index_to_calc_lstbox.insert(END, item)
        self.index_to_calc_lstbox.select_set(0)
        self.index_to_calc_lstbox.\
            bind('<<ListboxSelect>>', lambda event: self._on_listbox_change_selection(event, self.index_to_calc_lstbox))
        index_listbox_frame.grid(row=2, column=1, sticky=W, columnspan=2)
        return self.index_to_calc_lstbox

    def _on_listbox_change_selection(self,event,lstbox):
        selection = lstbox.curselection()
        if len(selection) <= 0:
            selection = self.lstbox_last_selection
            for idx in self.lstbox_last_selection:
                lstbox.select_set(idx)
        #'all' not have sense to be selected with more layer_names
        if 0 in selection and len(selection)>1:
            lstbox.select_clear(selection[1],END)
        self.lstbox_last_selection = selection

    def set_footers(self, master):
        frame = ttk.Frame(master=master)
        label = ttk.Label(master=frame, text='*(calculate indexes) Nefesi/main>>'
                                         ' nohup python calculate_indexes.py &', font=("Times New Roman", 8))
        label.grid(row=0)
        frame.pack(side=BOTTOM)

    def user_confirm(self):
        text = self.get_override_text()
        if text is not None:
            popup_window = ConfirmPopup(self.window, text=text)
            self.window.wait_window(popup_window.top)
            return popup_window.value
        else:
            return True

    def get_override_text(self):
        try:
            text = 'This action will override the following files:\n'
            with  open('../nefesi/evaluation_scripts/indexes_config.cfg', 'rb') as f:
                indexes = pickle.load(f)
            text += '\n' \
                    '../nefesi/evaluation_scripts/indexes_config.cfg\n' \
                    'model = ' + indexes.model_file + '\n' \
                                                      'network_data = ' + indexes.network_data_file + '\n' \
                                                      'verbose = ' + str(indexes.verbose) + '\n'
            return text
        except:
            return None



    def update_ok_button_state(self):
        if self.network_data_file is not None and self.model_file is not None:
            self.ok_button['state'] = 'normal'
        else:
            self.ok_button['state'] = 'disabled'

    def set_model_frame(self, master):
        label = ttk.Label(master=master, text="Model: ")
        label_selection = ttk.Label(master=master, text="No Model Selected")
        button = ttk.Button(master=master, text="Select File", command=lambda : self._on_click_set_model(label_selection) )
        label.grid(row=0, column=0, sticky=E, pady=2)
        button.grid(row=0, column=1, sticky=E, pady=2)
        label_selection.grid(row=0, column=2, sticky=W, pady=2)

    def set_network_data_file(self, master):
        label = ttk.Label(master=master, text="Nefesi File: ")
        label_selection = ttk.Label(master=master, text="No File Selected")
        button = ttk.Button(master=master, text="Select File",
                        command=lambda: self._on_click_set_network_data_file(label_selection))
        label.grid(row=1, column=0, sticky=E, pady=2)
        button.grid(row=1, column=1, sticky=E, pady=2)
        label_selection.grid(row=1, column=2, sticky=W, pady=2)

    def set_parameters_frame(self, master):
        self.set_network_data_file(master)
        self.set_model_frame(master)
    def set_verbose_check_box(self,master):
        self.verbose = BooleanVar(master=master,value=True)
        checkbox = ttk.Checkbutton(master=master, text="Verbose", variable=self.verbose)
        checkbox.grid(row=3, column=1, sticky=W, pady=2, columnspan=3)

    def ask_for_file(self, title="Select file", type='obj'):
        filename = filedialog.askopenfilename(title=title,
                                              filetypes=((type, '*.' + type), ("all files", "*.*")))
        if filename != '':
            filename = relpath(filename)
        return filename

    def ask_for_directory(self, title="Select Directory"):
        dir_name = filedialog.askdirectory(title=title)
        if dir_name != '':
            dir_name = relpath(dir_name)
        return dir_name

    def _on_click_set_model(self, label_selection):
        model_file = self.ask_for_file(title="Select model (.h5 file)",type='h5')
        if model_file != '':
            self.model_file = model_file
            label_selection.configure(text=model_file)
        self.update_ok_button_state()

    def _on_click_set_network_data_file(self, label_selection):
        network_data_file = self.ask_for_file(title="Select network data", type='obj')
        if network_data_file != '':
            self.network_data_file = network_data_file
            label_selection.configure(text=network_data_file)
        self.update_ok_button_state()