from ..evaluation_scripts.calculate_indexs import CalculateIndexs
from .popup_windows.confirm_popup import ConfirmPopup
from os.path import relpath

try:
    from tkinter import *
    from tkinter import ttk
    from tkinter import filedialog
except ImportError:
    from Tkinter import *
    from Tkinter import ttk
import pickle

class CalcIndexsInterface():
    def __init__(self):
        self.window = Tk()
        self.window.title("Nefesi")
        self.network_data_file = None
        self.model_file = None
        self.verbose =False
        self.select_parameters_frame = Frame(master=self.window, borderwidth=1)
        self.set_parameters_frame(master=self.select_parameters_frame)
        self.select_parameters_frame.pack()
        self.ok_button = Button(self.window, text='Ok', command=self.cleanup)
        self.ok_button['state'] = 'disabled'
        self.ok_button.pack(pady=(8, 5), ipadx=10)
        self.set_footers(master=self.window)
        self.window.mainloop()

    def cleanup(self):
        if self.user_confirm():
            network_data_file = self.network_data_file
            model_file = self.model_file
            verbose = self.verbose.get()
            self.window.destroy()
            indexs_eval = CalculateIndexs(network_data_file=network_data_file,model_file=model_file, verbose=verbose)
            with open('../nefesi/evaluation_scripts/indexs_config.cfg', 'wb') as f:
	            pickle.dump(indexs_eval, f)

    def set_footers(self, master):
        frame = Frame(master=master)
        label = Label(master=frame, text='*(calculate indexs) Nefesi/main>>'
                                         ' nohup python ./calculate_indexs.py &', font=("Times New Roman", 8))
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

        text = 'This action will override the following files:\n'
        try:
            f = open('../nefesi/evaluation_scripts/indexs_config.cfg', 'rb')
            indexs = pickle.load(f)
            f.close()
        except:
            indexs = None
        text += '\n' \
                '../nefesi/evaluation_scripts/indexs_config.cfg\n' \
                'model = ' + indexs.model_file + '\n'\
                'network_data = ' + indexs.network_data_file + '\n' \
                'verbose = ' + str(indexs.verbose) + '\n'
        return text


    def update_ok_button_state(self):
        if self.network_data_file is not None and self.model_file is not None:
            self.ok_button['state'] = 'normal'
        else:
            self.ok_button['state'] = 'disabled'

    def set_model_frame(self, master):
        label = Label(master=master, text="Select the model")
        label_selection = Label(master=master, text="No model selected")
        button = Button(master=master, text="Select file", command=lambda : self._on_click_set_model(label_selection) )
        label.pack(side=LEFT)
        label_selection.pack(side=RIGHT)
        button.pack(side=RIGHT)

    def set_parameters_frame(self, master):
        save_network_data_file_frame = Frame(master=master)
        self.set_network_data_file(save_network_data_file_frame)
        save_network_data_file_frame.pack()
        model_frame = Frame(master=master)
        self.set_model_frame(model_frame)
        model_frame.pack()
        verbose = Frame(master=master)
        self.set_verbose_check_box(verbose)
        verbose.pack()

    def set_verbose_check_box(self,master):
        self.verbose = BooleanVar(master=master,value=False)
        checkbox = ttk.Checkbutton(master=master, text="Verbose", variable=self.verbose)
        checkbox.pack()

    def set_network_data_file(self, master):
        label = Label(master=master, text="Select network_data file")
        label_selection = Label(master=master, text="No .obj selected")
        button = Button(master=master, text="Select file",
                        command=lambda: self._on_click_set_network_data_file(label_selection))
        label.pack(side=LEFT)
        label_selection.pack(side=RIGHT)
        button.pack(side=RIGHT)

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