import sys

VALID_LAYERS = '.*' #'.*conv*'
sys.path.append('..')
from nefesi.interface.popup_windows.confirm_popup import ConfirmPopup
from nefesi.util.general_functions import get_listbox_selection
from nefesi.util.image import ImageDataset
from nefesi.network_data import NetworkData
from nefesi.network_data import get_model_layer_names
from nefesi.evaluation_scripts.calculate_indexes import CalculateIndexes
from nefesi.evaluation_scripts.evaluate_with_config import EvaluationWithConfig

import os
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

from keras.models import load_model
import dill as pickle
import warnings
class MakeAnalysisInterface():
    def __init__(self, window_style = 'default'):
        self.window = Tk()
        ttk.Style().theme_use(window_style)
        self.window.title("Nefesi - Prepare Analysis")
        self.model = None
        self.dataset_dir = None
        self.save_path_dir = None
        self.lstbox_last_selection = [0]
        self.default_labels_dict = None
        #TOP Part with general info of viewing and some setteables
        self.buttons_frame = ttk.Frame(master=self.window, borderwidth=1)
        self.set_model_frame(master=self.buttons_frame)
        self.set_parameters_frame(master=self.buttons_frame)
        self.set_select_dataset_frame(master=self.buttons_frame)
        self.set_layers_and_options_frame(master=self.buttons_frame)
        self.buttons_frame.pack()
        self.ok_button = ttk.Button(self.window, text='Ok', command=self.cleanup)
        self.ok_button['state'] = 'disabled'
        self.ok_button.pack(pady=(8, 5), ipadx=10)
        self.set_footers(master=self.window)
        self.window.mainloop()

    def cleanup(self):
        if self.user_confirm():
            _,h, w, c = self.model.input_shape
            image_size = (h,w)
            model = self.model
            dataset_dir = self.dataset_dir
            segmented_dataset_dir = self.dataset_dir+'Segmented'
            model_file = self.model_file
            color_mode = 'rgb' if c == 3 else 'grayscale'
            save_path_dir = self.save_path_dir
            layers_to_evaluate = get_listbox_selection(self.layers_to_evaluate_lstbox)
            default_labels_dict = self.default_labels_dict
            also_calc_index = self.also_calc_index.get()
            verbose = self.verbose.get()
            self.window.destroy()
            dataset = ImageDataset(src_dataset=dataset_dir, target_size=image_size,preprocessing_function=None,
                                   color_mode=color_mode,src_segmentation_dataset=segmented_dataset_dir)
            network_data = NetworkData(model=model, layer_data=layers_to_evaluate, save_path=save_path_dir,dataset=dataset,
                                       save_changes=True, default_labels_dict=default_labels_dict)
            network_data.model = None

            evaluation = EvaluationWithConfig(network_data=network_data, model_file=model_file,evaluate_index=also_calc_index,
                                              verbose=verbose)
            with open('../nefesi/evaluation_scripts/evaluation_config.cfg', 'wb') as f:
                pickle.dump(evaluation, f)
            #saving the cfg for do it after
            network_data_file = os.path.join(network_data.save_path, model.name+'.obj')
            indexes_eval = CalculateIndexes(network_data_file=network_data_file, model_file=model_file, verbose=verbose)
            with open('../nefesi/evaluation_scripts/indexes_config.cfg', 'wb') as f:
                pickle.dump(indexes_eval, f)

    def set_footers(self, master):
        frame = ttk.Frame(master=master)
        label = ttk.Label(master=frame, text='*(eval network) Nefesi/main>> nohup python evaluate_with_config.py &',
                      font=("Times New Roman", 8))
        label.grid(row=0)
        label = ttk.Label(master=frame, text='**(calculate indexes) Nefesi/main>>'
                                         ' nohup python calculate_indexes.py &', font=("Times New Roman", 8))
        label.grid(row=1)
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
            f = open('../nefesi/evaluation_scripts/evaluation_config.cfg', 'rb')
            evaluation = pickle.load(f)
            f.close()
        except:
            evaluation = None
        text = 'This action will override the following files:\n'
        if evaluation is not None:
            text+= '\n' \
                   '../nefesi/evaluation_scripts/evaluation_config.cfg\n' \
                   'model = '+evaluation.model_file+'\n' \
                    'save_path = '+evaluation.network_data.save_path+'\n' \
                    'dataset = ' + evaluation.network_data.dataset.src_dataset + '\n' \
                    'verbose = ' + str(evaluation.verbose) + '\n'
        try:
            f = open('../nefesi/evaluation_scripts/indexes_config.cfg', 'rb')
            indexes = pickle.load(f)
            f.close()
        except:
            indexes = None
        if indexes is not None:
            text += '\n' \
                    '../nefesi/evaluation_scripts/indexes_config.cfg\n' \
                    'model = ' + indexes.model_file + '\n'\
                    'network_data = ' + indexes.network_data_file + '\n' \
                    'verbose = ' + str(indexes.verbose) + '\n'

        if evaluation is None and indexes is None:
            text = None
        return text


    def update_ok_button_state(self):
        if self.model is not None and self.dataset_dir is not None and self.save_path_dir is not None:
            self.ok_button['state'] = 'normal'
        else:
            self.ok_button['state'] = 'disabled'
    def set_model_frame(self, master):
        label = ttk.Label(master=master, text="Model: ")
        label_selection = ttk.Label(master=master, text="No Model Selected")
        button = ttk.Button(master=master, text="Select File", command=lambda : self._on_click_set_model(label_selection) )
        label.grid(row=0, column=0, sticky=E,pady=2)
        button.grid(row=0, column=1, sticky=E, pady=2)
        label_selection.grid(row=0, column=2, sticky=W, pady=2)


    def set_save_path_dir(self, master):
        label = ttk.Label(master=master, text="Save Directory: ")
        label_selection = Label(master=master, text="No Directory Selected")
        button = ttk.Button(master=master, text="Select Dir",
                        command=lambda: self._on_click_select_save_path_dir(label_selection))
        label.grid(row=1, column=0, sticky=E, pady=2)
        button.grid(row=1, column=1, sticky=E, pady=2)
        label_selection.grid(row=1, column=2, sticky=W, pady=2)

    def set_dataset_dir(self, master):
        label = ttk.Label(master=master, text="Dataset Directory: ")
        label_selection = ttk.Label(master=master, text="No Model Selected")
        button = ttk.Button(master=master, text="Select Dir",
                            command=lambda: self._on_click_select_dataset_dir(label_selection))
        label.grid(row=2, column=0, sticky=E, pady=2)
        button.grid(row=2, column=1, sticky=E, pady=2)
        label_selection.grid(row=2, column=2, sticky=W, pady=2)

    def set_default_labels_dict_dir(self, master):
        label = ttk.Label(master=master, text="Class Labels Dict: ")
        label_selection = ttk.Label(master=master, text="No Dictionary Selected")
        button = ttk.Button(master=master, text="Select File",
                        command=lambda: self._on_click_select_labels_dict_dir(label_selection))
        label.grid(row=3, column=0, sticky=E, pady=2)
        button.grid(row=3, column=1, sticky=E, pady=2)
        label_selection.grid(row=3, column=2, sticky=W, pady=2)

    def set_parameters_frame(self, master):
        self.set_save_path_dir(master)

    def set_layers_to_evaluate(self, master):
        label = ttk.Label(master=master, text="Layers to Evaluate: ")
        label.grid(row=4, column=0, sticky=E, pady=2, padx=(1,2))
        listbox_frame = ttk.Frame(master=master)
        self.set_layers_listbox(master=listbox_frame)
        listbox_frame.grid(row=4, column=1, sticky=W, columnspan=2)

    def set_layers_listbox(self, master):
        scrollbar = ttk.Scrollbar(master=master, orient="vertical")
        self.layers_to_evaluate_lstbox = Listbox(master=master, selectmode=EXTENDED, yscrollcommand=scrollbar.set,
                         height=MAX_VALUES_VISIBLES_IN_LISTBOX, width=20)
        scrollbar.config(command=self.layers_to_evaluate_lstbox.yview)
        self.layers_to_evaluate_lstbox.pack(side=LEFT)
        scrollbar.pack(side=RIGHT, fill="y")
        self.layers_to_evaluate_lstbox.\
            bind('<<ListboxSelect>>', lambda event: self._on_listbox_change_selection(event, self.layers_to_evaluate_lstbox))
        return self.layers_to_evaluate_lstbox


    def set_layers_and_options_frame(self, master):
        self.set_layers_to_evaluate(master)
        also_calc_index = ttk.Frame(master=master)
        self.set_calc_index_check_box(also_calc_index)
        also_calc_index.grid(row=5,column=1, sticky=W, pady=2, columnspan=3)
        verbose = ttk.Frame(master=master)
        self.set_verbose_check_box(verbose)
        verbose.grid(row=6, column=1, sticky=W, pady=2, columnspan=3)

    def set_verbose_check_box(self,master):
        self.verbose = BooleanVar(master=master,value=True)
        checkbox = ttk.Checkbutton(master=master, text="Verbose", variable=self.verbose)
        checkbox.pack()

    def set_calc_index_check_box(self,master):
        self.also_calc_index = BooleanVar(master=master,value=False)
        checkbox = ttk.Checkbutton(master=master, text="Also Calculate Indexes", variable=self.also_calc_index)
        checkbox.pack()

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

    def update_layers_to_evaluate(self, model):
        list_values = get_model_layer_names(model=self.model, regEx=VALID_LAYERS)
        self.layers_to_evaluate_lstbox.delete(0,END)
        self.layers_to_evaluate_lstbox.insert(END, 'all')
        for item in list_values:
            self.layers_to_evaluate_lstbox.insert(END, item)
        self.layers_to_evaluate_lstbox.select_set(0)



    def set_select_dataset_frame(self, master):
        self.set_dataset_dir(master)
        self.set_default_labels_dict_dir(master)

    def set_color_mode_frame(self, master):
        label = ttk.Label(master=master, text="Select ColorMode")
        self.combo_color_mode = ttk.Combobox(master=master, values=['rgb', 'grayscale'], state='readonly', width=9, justify=CENTER)
        self.combo_color_mode.set('rgb')
        label.pack(side=LEFT)
        self.combo_color_mode.pack(side=RIGHT)


    def ask_for_file(self, title="Select file", type='obj'):
        filename = filedialog.askopenfilename(title=title,
                                              filetypes=((type, '*.' + type), ("all files", "*.*")))
        #if filename != '':
        #    filename = relpath(filename)
        return filename

    def ask_for_directory(self, title="Select Directory"):
        dir_name = filedialog.askdirectory(title=title)
        #if dir_name != '':
        #    dir_name = relpath(dir_name)
        return dir_name

    def _on_click_set_model(self, label_selection):
        model_file = self.ask_for_file(title="Select model (.h5 file)",type='h5')
        if model_file != '':
            self.model_file = model_file
            self.model = load_model(model_file)
            label_selection.configure(text = self.model.name)
            self.update_layers_to_evaluate(model=self.model)
        self.update_ok_button_state()

    def _on_click_select_dataset_dir(self, label_selection):
        dataset_dir = self.ask_for_directory(title="Select dataset")
        if dataset_dir != '':
            self.dataset_dir = dataset_dir
            label_selection.configure(text=dataset_dir)
        self.update_ok_button_state()

    def _on_click_select_labels_dict_dir(self, label_selection):
        labels_dict = self.ask_for_file(title="Select labels dict", type='obj')
        if labels_dict != '':
            self.labels_dict = labels_dict
            label_selection.configure(text=labels_dict)



    def _on_click_select_save_path_dir(self, label_selection):
        save_path_dir = self.ask_for_directory(title="Select save path")
        if save_path_dir != '':
            self.save_path_dir = save_path_dir
            label_selection.configure(text=save_path_dir)
        self.update_ok_button_state()