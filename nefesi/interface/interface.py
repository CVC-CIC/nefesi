import tkinter as tk# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3
try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from Tkinter import ttk

import threading
import multiprocessing
import time
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from nefesi.layer_data import ALL_INDEX_NAMES
from nefesi.util.interface_plotting import get_plot_net_summary_figure
from nefesi.interface.popup_window import PopupWindow

STATES = ['init']
MAX_PLOTS_VISIBLES_IN_WINDOW = 4
MAX_VALUES_VISIBLES_IN_LISTBOX = 6
class Interface():
    def __init__(self, network_data, title = 'Nefesi'):
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
        self.state = 'init'
        self.plot_general_index(index='class')
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
            self.clean_widget(self.plots_canvas)


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
        lstbox.bind('<<ListboxSelect>>',lambda event: self._on_listbox_change_selection(event, lstbox))
        lstbox.selection_set(0)
        ok_button = ttk.Button(master=master, text="Proceed", style="TButton",
                               command=self._on_click_proceed_button)
        lstbox_frame.pack()
        lstbox_tittle.pack(side=TOP)
        lstbox.pack(side=LEFT)
        scrollbar.pack(side=RIGHT, fill="y")
        ok_button.pack()
        return lstbox

    def _on_click_proceed_button(self):
        current_plots = self.visible_plots_canvas[self.visible_plots_canvas['used']]
        for canvas, _,index, special_value in current_plots:
            if index in ALL_INDEX_NAMES:
                self.plot_general_index(index=index, master_canvas=canvas,
                                        special_value=special_value)

    def get_listbox_selection(self, lstbox):
        selection = lstbox.curselection()
        last_idx = selection[-1] if len(selection) > 1 else None
        layers_selected = lstbox.get(first=selection[0], last=last_idx)
        if type(layers_selected) is not str:
            layers_selected = list(layers_selected)
        elif layers_selected == 'all':
            layers_selected = '.*'
        return layers_selected
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
        self.current_layers_in_view = self.get_listbox_selection(lstbox)


    def set_save_changes_check_box(self,master):
        checkbox_value = tk.BooleanVar(master=master)
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
        combo.bind("<<ComboboxSelected>>", self._on_number_of_plots_to_show_changed)
        combo.set(options[0])
        combo.pack(expand=False)
        return combo

    def clean_widget(self, widget):
        for child in list(widget.children.values()):
            if list(child.children.values()) == []:
                child.destroy()
            else:
                self.clean_widget(child)
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
    def plot_general_index(self, index, master_canvas=None, special_value = 180):
        """
        Plots a general graphic of specified index in the general plots section
        :param index: String representing the index to plot. Needs to be one of prensents in
         nefesi.layer_data.ALL_INDEX_NAMES ('color', 'orientation', 'symmetry', 'class' or 'population code')
        :return:
        """
        figure, hidden_annotations = get_plot_net_summary_figure(index,
                                                                 layersToEvaluate=self.current_layers_in_view,
                                                                 degrees_orientation_idx=special_value,
                                                                 network_data=self.network_data)
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
            self.put_figure_plot(master=master_canvas, figure=figure, hidden_annotations=hidden_annotations)
            visible_plot_idx = np.where(self.visible_plots_canvas['canvas'] == master_canvas)[0][0]
            self.visible_plots_canvas[visible_plot_idx]['index'] = index
            self.visible_plots_canvas[visible_plot_idx]['special_value'] = special_value
        selector = self.get_index_button_general(master_canvas,default_index=index)
        selector.place(relx=0.25,rely=0)#grid(row=0,column=0, columnspan=2)
        erase_button = self.get_erase_plot_button(master=master_canvas)
        erase_button.place(relx=0.85,rely=0)#((row=0,column=3)

    def put_figure_plot(self, master, figure, hidden_annotations):
        self.destroy_canvas_subplot_if_exist(master_canvas=master)
        plot_canvas = FigureCanvasTkAgg(figure, master=master)
        if hidden_annotations is not None:
            plot_canvas.mpl_connect('motion_notify_event',
                                    lambda event: self._on_in_bar_hover(event, hidden_annotations))
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
        combo = ttk.Combobox(master=master, values=ALL_INDEX_NAMES, state='readonly')
        combo.bind("<<ComboboxSelected>>", self._on_general_plot_selector_changed)
        if default_index is not None:
            combo.set(default_index)
        return combo

    def get_erase_plot_button(self, master):
        button = ttk.Button(master=master, text="X", style="TButton")
        button.bind('<Button-1>', self._on_click_destroy_subplot)
        return button


    def get_value_from_popup(self,text='',index=''):
        popup_window = PopupWindow(self.window, text=text,index=index)
        self.window.wait_window(popup_window.top)
        return popup_window.value


    def destroy_plot_canvas(self, plot_canvas):
        self.clean_widget(plot_canvas)
        pos = np.where(self.visible_plots_canvas['canvas'] == plot_canvas)[0][0]
        self.visible_plots_canvas[pos] = (None, False, '', None)
        plot_canvas.destroy()


    def _on_in_bar_hover(self, event, hidden_annotations):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            for annotation, x0, x1, y0, y1 in hidden_annotations:
                if x0 < x < x1 and y0 < y < y1:
                    if not annotation.get_visible():
                        annotation.set_visible(True)
                        annotation.figure.canvas.draw()
                else:
                    if annotation.get_visible():
                        annotation.set_visible(False)
                        annotation.figure.canvas.draw()
        else:
            for annotation in hidden_annotations['annotation']:
                if annotation.get_visible():
                    annotation.set_visible(False)
                    annotation.figure.canvas.draw()


    def _on_click_destroy_subplot(self, event):
        #Is the master of the button (the canvas that have button,selector and plot)
        self.destroy_plot_canvas(plot_canvas=event.widget.master)
        self.graphics_to_show_combo.set(np.count_nonzero(self.visible_plots_canvas['used']))

    def _on_general_plot_selector_changed(self,event):
        """
        event called when user selects another chart to show in the combobox of the plot canvas (in general state (init)
        :param event: event with the widget of the combobox changed
        """
        master = event.widget.master
        selected = event.widget.get()
        rotation_degrees = None
        if selected == 'orientation':
            rotation_degrees = self.get_value_from_popup(index='Orientation', text='Set degrees of each rotation\n'
                                                                                '(only values in range [1,359] allowed).\n'
                                                                            'NOTE: Lower values will increment processing time')
        self.destroy_canvas_subplot_if_exist(master_canvas=master)
        self.plot_general_index(index=selected, master_canvas=master, special_value=rotation_degrees)

    def destroy_canvas_subplot_if_exist(self, master_canvas):
        if '!canvas' in master_canvas.children:
            oldplot = master_canvas.children['!canvas']
            self.clean_widget(widget=oldplot)
            oldplot.destroy()

    def _on_number_of_plots_to_show_changed(self, event):
        """
        Event called when user change the value of the combobox with... How many charts show? Add place to new charts
        or delete the charts that overflows
        :param event: event with the widget of combobox changed
        """
        # value selected by user
        selected = int(event.widget.get())
        # pos on array self.visible_plots_canvas of plots that are in use now
        plots_in_use_idx = np.where(self.visible_plots_canvas['used'] == True)[0]
        # erase the plots that overflows the new number of plots
        while len(plots_in_use_idx) > selected:
            # index of last plot that not have place for exist in new plots length
            element_to_erase = plots_in_use_idx[-1]
            self.destroy_plot_canvas(self.visible_plots_canvas['canvas'][element_to_erase])
            plots_in_use_idx = np.where(self.visible_plots_canvas['used'] == True)[0]
        # Readjust the plots in order to put put ordered. (example: if remains plots 1 and 3 and selected is 2, plots
        # 1 and 3 will be plots 0 and 1)
        if plots_in_use_idx[-1] >= selected:
            idx = [i for i in range(len(self.visible_plots_canvas))]
            valids_idx, non_valids = idx[:selected], idx[selected:]
            self.visible_plots_canvas[valids_idx] = self.visible_plots_canvas[plots_in_use_idx]
            self.visible_plots_canvas[non_valids] = (None, False,'', None)
            # Readjust in screen too
            for i, canvas in enumerate(self.visible_plots_canvas['canvas'][valids_idx]):
                canvas.grid(column=i % 2, row=(i // 2) + 1, sticky=SW)

        # Put new empty places for plot charts
        for i in range(len(plots_in_use_idx), selected):
            self.add_figure_to_frame(figure=None)


if __name__ == '__main__':
    Interface(None)

