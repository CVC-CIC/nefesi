import tkinter as tk# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3
import warnings

try:
    from tkinter import *
    from tkinter import ttk
    from tkinter import filedialog
except ImportError:
    from Tkinter import *
    from Tkinter import ttk

import threading
import multiprocessing
import time
import pickle
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from nefesi.layer_data import ALL_INDEX_NAMES
from nefesi.util.interface_plotting import get_one_layer_plot, get_plot_net_summary_figure
from nefesi.interface.popup_windows.special_value_popup_window import SpecialValuePopupWindow
from nefesi.interface.popup_windows.one_layer_popup_window import OneLayerPopupWindow
from nefesi.interface.popup_windows.neuron_window import NeuronWindow
import nefesi.interface.EventController as events
from nefesi.util.general_functions import clean_widget, destroy_canvas_subplot_if_exist
import nefesi.util.plotting as plotting

STATES = ['init']
MAX_PLOTS_VISIBLES_IN_WINDOW = 4
MAX_VALUES_VISIBLES_IN_LISTBOX = 6

class Interface():
    def __init__(self, network_data, title = 'Nefesi'):
        self.event_controller = events.EventController(self)
        self.network_data = network_data
        self.visible_plots_canvas = np.zeros(MAX_PLOTS_VISIBLES_IN_WINDOW,
                                dtype=np.dtype([('canvas', np.object), ('used',np.bool),
                                                ('index', 'U64'), ('special_value',np.object)]))
        self.current_layers_in_view = '.*'
        #Window element
        self.window = tk.Tk()
        self.window.title(title)
        #Style, defined here all common styles of buttons, frames....
        self.style = ttk.Style()
        self.style.configure("TButton", foreground="black", background="white")
        #TOP Part with general info of viewing and some setteables
        self.general_info_frame = Frame(master=self.window, borderwidth=1)
        #LEFT-BOTTOM part with all graphics and plots
        #self.plots_frame = Frame(master=self.window, borderwidth=1)
        self.plots_canvas = Canvas(master=self.window, background='white')
        #RIGHT part with general buttons of interface
        self.general_buttons_frame = Frame(master=self.window, borderwidth=1)
        self.set_general_buttons_frame()


        tk.Label(self.general_info_frame, text="Place to put texts").pack()
        self.plot_general_index(index=None)
        self.set_menu_bar()
        #self.plot_general_index(index='symmetry')
        #self.plot_general_index(index='class')
        #self.plot_general_index(index='class')
        #self.plot_general_index(index='class')
        #self.plot_general_index(index='orientation')
        self.window.mainloop()

    @property
    def network_data(self):
        return self._network_data
    @network_data.setter
    def network_data(self, network_data):
        self._network_data = network_data

    @property
    def general_info_frame(self):
        return self._general_info_frame
    @general_info_frame.setter
    def general_info_frame(self,general_info_frame):
        self._general_info_frame = general_info_frame
        self._general_info_frame.pack(side=TOP, expand=True)

    @property
    def window(self):
        return self._window
    @window.setter
    def window(self, window):
        self._window = window

    @property
    def style(self):
        return self._style
    @style.setter
    def style(self, style):
        self._style = style

    @property
    def plots_frame(self):
        return self._plots_frame
    @plots_frame.setter
    def plots_frame(self,plots_frame):
        self._plots_frame = plots_frame
        self._plots_frame.pack(side=LEFT, expand=True)

    @property
    def plots_canvas(self):
        return self._plots_canvas

    @plots_canvas.setter
    def plots_canvas(self, plots_canvas):
        self._plots_canvas = plots_canvas
        #In order to make the plots resizables on resize window
        self.addapt_widget_for_grid(self._plots_canvas)
        self._plots_canvas.pack(side=RIGHT, expand=True,padx=(150,0))

    @property
    def general_buttons_frame(self):
        return self._general_buttons_frame

    @general_buttons_frame.setter
    def general_buttons_frame(self, general_buttons_frame):
        self._general_buttons_frame = general_buttons_frame
        self._general_buttons_frame.place(rely=0.45)#relx=0.8,rely=0.45)#pack(side=RIGHT, expand=True)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        """
        Implements the state machine and is the setter called when variable state is changed
        :param state: the new state of the machine
        :raise ValueError: if the new state is a non-valid state
        """
        state = state.lower()

        if state == 'init':
            clean_widget(self.plots_canvas)

    def raise_neuron_window(self, layer, neuron_idx):

        NeuronWindow(master=self.window, network_data=self.network_data, layer_to_evaluate=layer, neuron_idx=neuron_idx)

    def set_menu_bar(self):
        menubar = Menu(master=self.window)
        self.window.config(menu=menubar)
        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Set Label traduction", command=self.set_labels_dict)
        menubar.add_cascade(label="Configuration", menu=fileMenu)

    def ask_for_file(self, title="Select file", type='obj'):
        filename = filedialog.askopenfilename(initialdir="/", title=title,
                                   filetypes=((type, '*.'+type), ("all files", "*.*")))
        return filename

    def set_labels_dict(self):
        labels_dict_file = self.ask_for_file(title="Select labels traduction (Python dict object)")
        self.network_data.default_labels_dict = labels_dict_file

    def set_general_buttons_frame(self):
        self.set_save_changes_check_box(master=self.general_buttons_frame)
        self.layers_listbox = self.set_info_to_show_listbox(master=self.general_buttons_frame)
        self.graphics_to_show_combo = self.set_grafics_to_show_combobox(master=self.general_buttons_frame)

    def set_info_to_show_listbox(self, master):
        # Title just in top of selector
        lstbox_frame = Frame(master=master)
        lstbox_tittle = ttk.Label(master=lstbox_frame, text="Layers to show")
        list_values = [layer.layer_id for layer in self.network_data.layers_data]
        scrollbar = tk.Scrollbar(master=lstbox_frame, orient="vertical")
        lstbox = Listbox(master=lstbox_frame, selectmode=EXTENDED, yscrollcommand=scrollbar.set,
                         height=min(len(list_values)+2,MAX_VALUES_VISIBLES_IN_LISTBOX))
        scrollbar.config(command=lstbox.yview)
        lstbox.insert(END, 'all')
        for item in list_values:
            lstbox.insert(END, item)
        self.lstbox_last_selection = (0,)
        lstbox.bind('<<ListboxSelect>>',lambda event: self.event_controller._on_listbox_change_selection(event, lstbox))
        lstbox.selection_set(0)
        ok_button = ttk.Button(master=master, text="Apply on all", style="TButton",
                               command=self.event_controller._on_click_proceed_button)
        lstbox_frame.pack()
        lstbox_tittle.pack(side=TOP)
        lstbox.pack(side=LEFT)
        scrollbar.pack(side=RIGHT, fill="y")
        ok_button.pack()
        return lstbox

    def get_listbox_selection(self, lstbox):
        selection = lstbox.curselection()
        layers_selected = [lstbox.get(first=selection[i]) for i in range(len(selection))]
        if len(layers_selected) == 1 and layers_selected[0] == 'all':
            layers_selected = '.*'
        return layers_selected

    def set_save_changes_check_box(self,master):
        checkbox_value = tk.BooleanVar(master=master)
        checkbox = ttk.Checkbutton(master=master, text="Save all index updated", variable=checkbox_value,
                                    command= lambda: self.event_controller._on_checkbox_clicked(checkbox_value))
        checkbox.pack()

    def set_grafics_to_show_combobox(self, master):
        # Title just in top of selector
        combo_title = ttk.Label(self.general_buttons_frame, text="Graphics to show")
        combo_title.pack(expand=False)
        # Options will be 1,2,3,4
        options = [i for i in range(1, MAX_PLOTS_VISIBLES_IN_WINDOW + 1)]
        combo = ttk.Combobox(master=self.general_buttons_frame, values=options, state='readonly', width=4)
        # When selection is changed, calls the function _on_number_of_plots_to_show_changed
        combo.bind("<<ComboboxSelected>>", self.event_controller._on_number_of_plots_to_show_changed)
        combo.set(options[0])
        combo.pack(expand=False)
        return combo

    """
    def thread_show_progress_bar(self, master, seconds_between_updates=1):
        print("PACKED")
        while self.is_occuped:
            print("a")
            #self.progress_bar["value"] = self.network_data.get_progress()
            #time.sleep(seconds_between_updates)
        print("END WHILE")
        self.progress_bar.destroy()

    def show_progress_bar(self, master, seconds_after_progress_bar=1):
        self.progress_bar = ttk.Progressbar(master=self.window, orient="horizontal", length=200, mode="determinate",maximum=100)
        self.progress_bar.pack()
        threading.Thread(target=self.thread_show_progress_bar, args=(master,)).start()
    """
    def plot_general_index(self, index, master_canvas=None, layers='.*', neuron=-1, special_value = 180):
        """
        Plots a general graphic of specified index in the general plots section
        :param index: String representing the index to plot. Needs to be one of prensents in
         nefesi.layer_data.ALL_INDEX_NAMES ('color', 'orientation', 'symmetry', 'class' or 'population code')
        :return:
        """
        if index is None:
            self.add_figure_to_frame(master_canvas=master_canvas, figure=None)
        else:
            if type(layers) in [str, np.str_]:
                    layers = self.network_data.get_layers_analyzed_that_match_regEx(layers)
            if type(layers) is list:
                if len(layers) == 0:
                    warnings.warn("layers is a 0-lenght list. Check why. Value: "+
                                  str(layers), RuntimeWarning)
                    figure = hidden_annotations = None
                elif len(layers) == 1:
                    if neuron < 0:
                        min, condition1, max, condition2, order, max_neurons = self.get_one_layer_params_popup(index=index,
                                                                                                    layer_to_evaluate=layers,
                                                                                                special_value=special_value)
                        figure, hidden_annotations = get_one_layer_plot(index, network_data=self.network_data,
                                                                    layer_to_evaluate=layers,
                                                                    special_value=special_value,min=min, max=max,
                                                                    condition1=condition1, condition2=condition2,
                                                                    order=order, max_neurons=max_neurons)
                    else:
                        raise ValueError('WHY? Is entering here, when is a neuron specific plot? Neuron: '+str(neuron))

                else:
                    figure, hidden_annotations = get_plot_net_summary_figure(index, layersToEvaluate=layers,
                                                                             special_value=special_value, network_data=self.network_data)
            else:
                warnings.warn("self.current_layers_in_view is not a list. Check why. Value: "+
                                  str(layers)+str(type(layers)), RuntimeWarning)
                figure = hidden_annotations = None

            self.add_figure_to_frame(master_canvas=master_canvas, figure=figure, hidden_annotations=hidden_annotations,
                                     index=index, special_value=special_value)


    def add_figure_to_frame(self,master_canvas=None, figure=None, hidden_annotations=None, index=None, special_value=None):
        if master_canvas is None:
            first_util_place = np.where(self.visible_plots_canvas['used'] == False)[0][0]
            master_canvas = Canvas(master=self.plots_canvas)
            self.addapt_widget_for_grid(master_canvas)
            master_canvas.configure(width=800, height=450)
            master_canvas.grid(column=first_util_place%2, row=(first_util_place//2)+1, sticky=SW)
            self.visible_plots_canvas[first_util_place] = (master_canvas, True, index, special_value)
        if figure is not None:
            self.put_figure_plot(master=master_canvas, figure=figure, hidden_annotations=hidden_annotations,index=index,
                                 special_value=special_value)
            visible_plot_idx = np.where(self.visible_plots_canvas['canvas'] == master_canvas)[0][0]
            self.visible_plots_canvas[visible_plot_idx]['index'] = index
            self.visible_plots_canvas[visible_plot_idx]['special_value'] = special_value
        selector = self.get_index_button_general(master_canvas,default_index=index)
        selector.place(relx=0.4,rely=0)#grid(row=0,column=0, columnspan=2)
        erase_button = self.get_erase_plot_button(master=master_canvas)
        erase_button.place(relx=0.85,rely=0)#((row=0,column=3)

    def put_figure_plot(self, master, figure,index, hidden_annotations, special_value = 180):
        destroy_canvas_subplot_if_exist(master_canvas=master)
        plot_canvas = FigureCanvasTkAgg(figure, master=master)
        if hidden_annotations is not None:
            plot_canvas.mpl_connect('motion_notify_event',
                                    lambda event: self.event_controller._on_in_plot_element_hover(event, hidden_annotations))

        plot_canvas.mpl_connect('button_press_event',
                                lambda event: self.event_controller._on_in_plot_element_double_click(event, hidden_annotations,
                                                                                                     master, index, special_value))
        self.addapt_widget_for_grid(plot_canvas.get_tk_widget())
        plot_canvas.get_tk_widget().configure(width=800, height=450)
        # plot_canvas.draw()
        plot_canvas.get_tk_widget().grid(row=1, sticky=SW)

    def addapt_widget_for_grid(self, widget):
        for i in range(3):
            Grid.columnconfigure(widget, i, weight=1)
            Grid.rowconfigure(widget, i, weight=1)

    def get_index_button_general(self, master, default_index = None):
        """
        Gets a general button to select wich graphic to plot
        :return: A select button with each index possible, and the event to plot it when called
        """
        combo = ttk.Combobox(master=master, values=ALL_INDEX_NAMES, state='readonly',justify=CENTER,width=15)
        combo.bind("<<ComboboxSelected>>", self.event_controller._on_general_plot_selector_changed)
        if default_index is not None:
            combo.set(default_index)
        else:
            combo.set('Select Index')
        return combo

    def get_erase_plot_button(self, master):
        button = ttk.Button(master=master, text="X", style="TButton")
        button.bind('<Button-1>', self.event_controller._on_click_destroy_subplot)
        return button


    def get_value_from_popup(self,index=''):
        popup_window = SpecialValuePopupWindow(self.window, index=index)
        self.window.wait_window(popup_window.top)
        return popup_window.value

    def get_one_layer_params_popup(self,index='',layer_to_evaluate='unknow', special_value=0.1):
        if type(layer_to_evaluate) is list:
            layer_to_evaluate = layer_to_evaluate[0]
        popup_window = OneLayerPopupWindow(self.window, layer_to_evaluate=layer_to_evaluate, index=index,
                                           special_value=special_value)
        self.window.wait_window(popup_window.top)
        return popup_window.value1,popup_window.condition1,popup_window.value2,popup_window.condition2,\
               popup_window.order, popup_window.neurons_to_show


    def destroy_plot_canvas(self, plot_canvas):
        clean_widget(plot_canvas)
        pos = np.where(self.visible_plots_canvas['canvas'] == plot_canvas)[0][0]
        self.visible_plots_canvas[pos] = (None, False, '', None)
        plot_canvas.destroy()




if __name__ == '__main__':
    Interface(None)

