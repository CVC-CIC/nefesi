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
        self.window.title("Nefesi")
        self.model = None
        self.dataset_dir = None
        self.save_path_dir = None
        self.lstbox_last_selection = [0]
        self.default_labels_dict = None
        #TOP Part with general info of viewing and some setteables
        self.select_model_frame = ttk.Frame(master=self.window, borderwidth=1)
        self.set_model_frame(master=self.select_model_frame)
        self.select_model_frame.pack()
        self.select_parameters_frame = ttk.Frame(master=self.window, borderwidth=1)
        self.set_parameters_frame(master=self.select_parameters_frame)
        self.select_parameters_frame.pack()
        self.select_dataset_frame = ttk.Frame(master=self.window, borderwidth=1)
        self.set_select_dataset_frame(master=self.select_dataset_frame)
        self.select_dataset_frame.pack()
        self.layers_and_options_frame = ttk.Frame(master=self.window, borderwidth=1)
        self.set_layers_and_options_frame(master=self.layers_and_options_frame)
        self.layers_and_options_frame.pack()
        self.ok_button = ttk.Button(self.window, text='Ok', command=self.cleanup)
        self.ok_button['state'] = 'disabled'
        self.ok_button.pack(pady=(8, 5), ipadx=10)
        self.set_footers(master=self.window)
        self.window.mainloop()

    def cleanup(self):
        if self.user_confirm():
            image_size = (int(self.entry_h.get()), int(self.entry_w.get()))
            model = self.model
            dataset_dir = self.dataset_dir
            segmented_dataset_dir = self.dataset_dir+'Segmented'
            model_file = self.model_file
            color_mode = self.combo_color_mode.get()
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
        if self.model is not None and self.dataset_dir is not None and self.save_path_dir is not None\
            and self.entry_h.get() != '' and self.entry_w.get() != '':
            self.ok_button['state'] = 'normal'
        else:
            self.ok_button['state'] = 'disabled'
    def set_model_frame(self, master):
        label = ttk.Label(master=master, text="Select the model")
        label_selection = ttk.Label(master=master, text="No model selected")
        button = ttk.Button(master=master, text="Select file", command=lambda : self._on_click_set_model(label_selection) )
        label.pack(side=LEFT)
        label_selection.pack(side=RIGHT)
        button.pack(side=RIGHT)

    def set_parameters_frame(self, master):
        save_path_frame = ttk.Frame(master=master)
        self.set_save_path_dir(save_path_frame)
        save_path_frame.pack()


    def set_layers_and_options_frame(self, master):
        layers_to_evaluate = ttk.Frame(master=master)
        self.set_layers_to_evaluate(layers_to_evaluate)
        layers_to_evaluate.pack()
        also_calc_index = ttk.Frame(master=master)
        self.set_calc_index_check_box(also_calc_index)
        also_calc_index.pack()
        verbose = ttk.Frame(master=master)
        self.set_verbose_check_box(verbose)
        verbose.pack()

    def set_verbose_check_box(self,master):
        self.verbose = BooleanVar(master=master,value=False)
        checkbox = ttk.Checkbutton(master=master, text="Verbose", variable=self.verbose)
        checkbox.pack()

    def set_calc_index_check_box(self,master):
        self.also_calc_index = BooleanVar(master=master,value=False)
        checkbox = ttk.Checkbutton(master=master, text="Also calc indexes in same script", variable=self.also_calc_index)
        checkbox.pack()


    def set_default_labels_dict_dir(self, master):
        label = ttk.Label(master=master, text="Select default Labels Dict")
        label_selection = ttk.Label(master=master, text="No translation dict")
        button = ttk.Button(master=master, text="Select file",
                        command=lambda: self._on_click_select_labels_dict_dir(label_selection))
        label.pack(side=LEFT)
        label_selection.pack(side=RIGHT)
        button.pack(side=RIGHT)

    def set_layers_to_evaluate(self, master):
        label = ttk.Label(master=master, text="Select layers to evaluate")
        label.pack(side=LEFT)
        self.set_layers_listbox(master=master)

    def set_layers_listbox(self, master):
        scrollbar = ttk.Scrollbar(master=master, orient="vertical")
        self.layers_to_evaluate_lstbox = Listbox(master=master, selectmode=EXTENDED, yscrollcommand=scrollbar.set,
                         height=MAX_VALUES_VISIBLES_IN_LISTBOX)
        scrollbar.config(command=self.layers_to_evaluate_lstbox.yview)
        self.layers_to_evaluate_lstbox.pack(side=LEFT)
        scrollbar.pack(side=RIGHT, fill="y")
        self.layers_to_evaluate_lstbox.\
            bind('<<ListboxSelect>>', lambda event: self._on_listbox_change_selection(event, self.layers_to_evaluate_lstbox))
        return self.layers_to_evaluate_lstbox

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

    def set_save_path_dir(self, master):
        label = ttk.Label(master=master, text="Select Save Directory")
        label_selection = Label(master=master, text="No directory selected")
        button = ttk.Button(master=master, text="Select dir",
                        command=lambda: self._on_click_select_save_path_dir(label_selection))
        label.pack(side=LEFT)
        label_selection.pack(side=RIGHT)
        button.pack(side=RIGHT)




    def set_select_dataset_frame(self, master):
        directory_frame = ttk.Frame(master=master)
        self.set_dataset_dir(directory_frame)
        directory_frame.pack()
        default_labels_dict_frame = ttk.Frame(master=master)
        self.set_default_labels_dict_dir(default_labels_dict_frame)
        default_labels_dict_frame.pack()
        images_size_frame = ttk.Frame(master=master)
        self.set_select_images_size(images_size_frame)
        images_size_frame.pack()
        color_mode_frame = ttk.Frame(master=master)
        self.set_color_mode_frame(color_mode_frame)
        color_mode_frame.pack()

    def set_color_mode_frame(self, master):
        label = ttk.Label(master=master, text="Select ColorMode")
        self.combo_color_mode = ttk.Combobox(master=master, values=['rgb', 'grayscale'], state='readonly', width=9, justify=CENTER)
        self.combo_color_mode.set('rgb')
        label.pack(side=LEFT)
        self.combo_color_mode.pack(side=RIGHT)

    def set_dataset_dir(self, master):
        label = ttk.Label(master=master, text="Select Dataset Directory")
        label_selection = ttk.Label(master=master, text="No model selected")
        button = ttk.Button(master=master, text="Select dir",
                        command=lambda: self._on_click_select_dataset_dir(label_selection))
        label.pack(side=LEFT)
        label_selection.pack(side=RIGHT)
        button.pack(side=RIGHT)

    def set_select_images_size(self, master):
        ttk.Label(master=master, text="Select images size").pack(side=LEFT)
        validate_command = (master.register(self._on_entry_updated_check_validity),
                            '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        ttk.Label(master=master, text=" H:").pack(side=LEFT)
        self.entry_h = Entry(master=master, validate='key', validatecommand=validate_command,
                                           textvariable=StringVar(master=self.window), justify=CENTER, width=5)
        self.entry_h.pack(side=LEFT)
        ttk.Label(master=master, text=" W:").pack(side=LEFT)
        self.entry_w = Entry(master=master, validate='key', validatecommand=validate_command,
                        textvariable=StringVar(master=self.window), justify=CENTER, width=5)
        self.entry_w.pack(side=RIGHT)
        self.entry_h.bind('<KeyRelease>', self._on_entry_updated)
        self.entry_w.bind('<KeyRelease>', self._on_entry_updated)

    def _on_entry_updated(self, event):
        self.update_ok_button_state()

    def update_image_sizes_entry(self,model):
        _,h,w,_ = model.input.shape
        h,w = str(h), str(w)
        self.entry_h.delete(0,END)
        self.entry_w.delete(0,END)
        for i in reversed(range(len(h))):
            self.entry_h.insert(0, h[i])
        for i in reversed(range(len(w))):
            self.entry_w.insert(0, w[i])

    def update_color_mode(self, model):
        channels = model.input.shape[-1]
        if channels == 3:
            self.combo_color_mode.set('rgb')
        elif channels == 1:
            self.combo_color_mode.set('grayscale')
        else:
            warnings.warn('Nefesi not prepared for networks with '+str(channels)+' channels. Only 3 (rgb) or 1 (grayscale)')

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
            self.update_image_sizes_entry(model=self.model)
            self.update_color_mode(model=self.model)
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

    def _on_entry_updated_check_validity(self, action, index, value_if_allowed,
                                         prior_value, text, validation_type, trigger_type, widget_name):
        # action=1 -> insert
        if (action == '1'):
            if text in '0123456789':
                try:
                    int(value_if_allowed)
                    return True
                except ValueError:
                    return False
            else:
                return False
        else:
            return True
