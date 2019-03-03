STATES = ['init']
MAX_PLOTS_VISIBLES_IN_WINDOW = 4
MAX_VALUES_VISIBLES_IN_LISTBOX = 6
PLOTTABLE_ENTITIES = ['class', 'object', 'color']
COOCURRENCE_OPTIONS = ['1/PC', '1/2', 'local selectivity sum']
REPRESENTATION_OPTIONS = ['1/PC', '1', 'local selectivity']

import tkinter as tk# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3
import warnings
from os.path import relpath

try:
    from tkinter import *
    from tkinter import ttk
    from tkinter import filedialog
except ImportError:
    from Tkinter import *
    from Tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ..util.interface_plotting import get_one_layer_plot, get_plot_net_summary_figure
from .popup_windows.open_selected_neuron_plot import OpenSelectedNeuronPlot
from .popup_windows.special_value_popup_window import SpecialValuePopupWindow
from .popup_windows.combobox_popup_window import ComboboxPopupWindow
from .popup_windows.one_layer_popup_window import OneLayerPopupWindow
from .popup_windows.neuron_window import NeuronWindow
from .popup_windows.confirm_popup import ConfirmPopup
from .popup_windows.erase_calculated_index_popup import EraseCalculatedIndexPopup
from .popup_windows.select_index_window import SelectIndexWindow
from . import EventController as events
from ..util.general_functions import clean_widget, destroy_canvas_subplot_if_exist, addapt_widget_for_grid
from ..network_data import NetworkData
from ..util.plotting import plot_nf_of_entities_in_pc, plot_coocurrence_graph, plot_entity_representation,\
    plot_similarity_graph,plot_similarity_tsne



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
        self.general_info_label = tk.Label(self.general_info_frame)
        self.update_general_info_label()
        self.general_info_label.pack()
        self.plot_general_index(index=None)
        self.set_menu_bar()

        #from nefesi.util.plotting import neurons_by_object_vs_ocurrences_in_imagenet
        #neurons_by_object_vs_ocurrences_in_imagenet(self.network_data, entity='object',operation='1')

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
        addapt_widget_for_grid(self._plots_canvas)
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

    def update_general_info_label(self, file_name = None):
        if self.network_data is None:
            self.general_info_label.configure(text = "No model selected")
        else:
            if self.network_data.model is not None:
                network_name = self.network_data.model.name
            elif file_name is not None:
                network_name = file_name[file_name.rfind('/')+1:file_name.rfind('.')]
            else:
                network_name = self.network_data.save_path
            self.general_info_label.configure(text="Network: "+network_name+"   ---   "
                                            " Dataset: "+self.network_data.dataset.src_dataset)

    def raise_neuron_window(self, layer, neuron_idx):
        NeuronWindow(master=self.window, network_data=self.network_data, layer_to_evaluate=layer, neuron_idx=neuron_idx)

    def set_menu_bar(self):
        menubar = Menu(master=self.window)
        self.window.config(menu=menubar)
        file_name = Menu(menubar)
        file_name.add_command(label="Force to save", command=self.force_to_save)
        file_name.add_command(label='Change analysis object', command=self.set_model)
        menubar.add_cascade(label="File", menu=file_name)
        config_menu = Menu(menubar)
        config_menu.add_command(label="Set Label traduction", command=self.set_labels_dict)
        config_menu.add_command(label='Set Orientation Degrees', command=self.set_orientation_default_degrees)
        config_menu.add_command(label='Set Threshold Population Code', command=self.set_default_thr_pc)
        config_menu.add_command(label='Update Indexes Accepted', command=self.update_indexes_accepted)
        config_menu.add_command(label='Erase calculated index', command=self.erase_calculated_index)
        menubar.add_cascade(label="Configuration", menu=config_menu)
        plot_menu = Menu(menubar)
        plot_menu.add_command(label="Select Neuron plot", command=self.plot_specific_neuron)
        plot_menu.add_command(label="Entity Selective neurons", command=self.class_selective_plot)
        plot_menu.add_command(label="Entity Representation", command=self.entity_representation_plot)
        plot_menu.add_command(label="Entity Co-ocurrence Graph", command=self.coocurrence_plot)
        plot_menu.add_command(label="Neuron Similarity Graph", command=self.similarity_graph)
        plot_menu.add_command(label="Neuron Feature Similarity TSNE", command=self.similarity_TSNE)
        menubar.add_cascade(label="Plot", menu=plot_menu)



    def plot_specific_neuron(self):
        layer, neuron_idx = self.get_neuron_params_plot_from_popup()
        if layer is not None and neuron_idx is not None:
            neuron_window = NeuronWindow(master=self.window, network_data=self.network_data, layer_to_evaluate=layer, neuron_idx=neuron_idx)
            self.window.wait_window(neuron_window.window)

    def class_selective_plot(self):
        text = 'Select an entity for make the plot'
        entity = self.get_value_from_popup_combobox(values=PLOTTABLE_ENTITIES, text=text)
        if entity != -1:
            plot_nf_of_entities_in_pc(self.network_data, master = self.window, entity=entity)

    def coocurrence_plot(self):
        text = 'Select an entity for make the graph'
        entity = self.get_value_from_popup_combobox(values=PLOTTABLE_ENTITIES, text=text)
        if entity != -1:
            text = 'Contribution of each neuron to a pair'
            operation = self.get_value_from_popup_combobox(values=COOCURRENCE_OPTIONS, text=text)
            if operation != -1:
                plot_coocurrence_graph(self.network_data, layers=self.current_layers_in_view, interface=self,entity=entity,
                                   operation=operation)

    def similarity_graph(self):
        text = 'Select an entity for print on the node names'
        entity = self.get_value_from_popup_combobox(values=PLOTTABLE_ENTITIES, text=text)
        if entity != -1:
            layer = self.current_layers_in_view
            if type(layer) is not list:
                # Compile the Regular expresion
                regEx = re.compile(layer)
                # Select the layerNames that satisfies RegEx
                layer = list(filter(regEx.match, [layer for layer in self.network_data.get_layers_name()]))
            if len(layer)>1:
                text = 'Layer where plot similarity'
                layer = [self.get_value_from_popup_combobox(values=layer, text=text)]
                if layer[0] == -1:
                    return
            plot_similarity_graph(self.network_data, layer=layer, interface=self)

    def similarity_TSNE(self):
        layer = self.current_layers_in_view
        if type(layer) is not list:
            # Compile the Regular expresion
            regEx = re.compile(layer)
            # Select the layerNames that satisfies RegEx
            layer = list(filter(regEx.match, [layer for layer in self.network_data.get_layers_name()]))
        if len(layer)>1:
            text = 'Layer where plot similarity TSNE'
            layer = [self.get_value_from_popup_combobox(values=layer, text=text)]
            if layer[0] == -1:
                return
        plot_similarity_tsne(layer_data=self.network_data.get_layer_by_name(layer[0]))


    def entity_representation_plot(self):
        text = 'Select an entity for make the plot'
        entity = self.get_value_from_popup_combobox(values=PLOTTABLE_ENTITIES, text=text)
        if entity != -1:
            text = 'Contribution of each neuron'
            operation = self.get_value_from_popup_combobox(values=REPRESENTATION_OPTIONS, text=text)
            if operation != -1:
                plot_entity_representation(self.network_data, layers=self.current_layers_in_view, interface=self,
                                           entity=entity, operation=operation)

    def ask_for_file(self, title="Select file", type='obj'):
        filename = filedialog.askopenfilename(title=title,
                                   filetypes=((type, '*.'+type), ("all files", "*.*")))
        if filename != '':
            filename = relpath(filename)
        return filename

    def force_to_save(self):
        self.network_data.save_to_disk(file_name=None, save_model=False)
        print("Changes saved")

    def set_labels_dict(self):
        labels_dict_file = self.ask_for_file(title="Select labels traduction (Python dict object)")
        self.network_data.default_labels_dict = labels_dict_file

    def erase_calculated_index(self):
        popup = EraseCalculatedIndexPopup(master = self.window, network_data = self.network_data)
        self.window.wait_window(popup.top)

    def update_indexes_accepted(self):
        popup = SelectIndexWindow(master=self.window, actual_indexes=self.network_data.indexs_accepted)
        self.window.wait_window(popup.top)
        if popup.new_indexes != -1:
            self.network_data.indexs_accepted = popup.new_indexes


    def set_model(self):
        network_data_file = self.ask_for_file(title="Select NetworkData object (.obj file)")
        if network_data_file != '':
            model_file = self.ask_for_file(title="Select model (.h5 file)",type="h5")
            model_file = model_file if model_file != '' else None
            self.network_data = NetworkData.load_from_disk(file_name=network_data_file, model_file=model_file)
            self.update_general_info_label(file_name=network_data_file)
            self.update_layers_lstbox(lstbox=self.layers_listbox)


    def set_orientation_default_degrees(self):
        orientation_degrees = self.get_value_from_popup('orientation')
        if orientation_degrees != -1:
            self.network_data.default_degrees_orientation_idx = orientation_degrees

    def set_default_thr_pc(self):
        thr_pc = self.get_value_from_popup('population code')
        if thr_pc != -1:
            self.network_data.default_thr_pc = thr_pc

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
        self.update_layers_lstbox(lstbox=lstbox, list_values=list_values)
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

    def update_layers_lstbox(self, lstbox=None, list_values = None):
        if lstbox is None:
            lstbox = self.layers_listbox
        if list_values is None:
            list_values = [layer.layer_id for layer in self.network_data.layers_data]
        lstbox.delete(0,END)
        lstbox.insert(END, 'all')
        for item in list_values:
            lstbox.insert(END, item)

    def set_save_changes_check_box(self,master):
        checkbox_value = tk.BooleanVar(master=master,value=self.network_data.save_changes)
        checkbox = ttk.Checkbutton(master=master, text="Save all index updated", variable=checkbox_value,
                                    command= lambda: self._on_checkbox_clicked(checkbox_value))
        checkbox.pack()

    def _on_checkbox_clicked(self,checkbox_value):
        self.network_data.save_changes = checkbox_value.get()

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
            confirmation = True
            if type(layers) in [str, np.str_]:
                    layers = self.network_data.get_layers_analyzed_that_match_regEx(layers)
            if not self.network_data.is_index_in_layer(layers=layers,index=index,special_value=special_value):
                confirmation = self.user_confirmation(index=index)
            if type(layers) is list:
                if confirmation:
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
                    figure = hidden_annotations = None
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
            addapt_widget_for_grid(master_canvas)
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
        addapt_widget_for_grid(plot_canvas.get_tk_widget())
        plot_canvas.get_tk_widget().configure(width=800, height=450)
        # plot_canvas.draw()
        plot_canvas.get_tk_widget().grid(row=1, sticky=SW)


    def get_index_button_general(self, master, default_index = None):
        """
        Gets a general button to select wich graphic to plot
        :return: A select button with each index possible, and the event to plot it when called
        """
        combo = ttk.Combobox(master=master, values=self.network_data.indexs_accepted, state='readonly',justify=CENTER,width=15)
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


    def get_value_from_popup(self, index='', max=100., start=10, text=''):
        popup_window = SpecialValuePopupWindow(self.window, network_data=self.network_data, index=index, max=max, start=start, text=text)
        self.window.wait_window(popup_window.top)
        return popup_window.value

    def get_value_from_popup_combobox(self, values, text):
        popup_window = ComboboxPopupWindow(self.window, values=values, text=text)
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

    def get_neuron_params_plot_from_popup(self):
        popup_window = OpenSelectedNeuronPlot(master=self.window, network_data=self.network_data)
        self.window.wait_window(popup_window.top)
        return popup_window.layer, popup_window.neuron

    def destroy_plot_canvas(self, plot_canvas):
        clean_widget(plot_canvas)
        pos = np.where(self.visible_plots_canvas['canvas'] == plot_canvas)[0][0]
        self.visible_plots_canvas[pos] = (None, False, '', None)
        plot_canvas.destroy()
    def user_confirmation(self, index):
        text = index.title() +' is not calculated for one or more of selected neurons\n' \
                              'this operation may take '
        if index in ['symmetry', 'orientation']:
            text += 'several hours\n'
        elif index in 'color':
            text += 'several minutes\n'
        elif index == 'object':
            text += 'a few minutes\n'
        else:
            text += 'a few seconds\n'
        text+= '\n Are you sure to calculate it?'
        popup_window = ConfirmPopup(self.window, text=text)
        self.window.wait_window(popup_window.top)
        return popup_window.value

    """
    def raise_loading_popup(self):
        self.loading_popup = LoadingPopup(master = self.window)
    def destroy_loading_popup(self):
        if self.loading_popup is not None:
            self.loading_popup.top.destroy()
            self.loading_popup = None
    """

