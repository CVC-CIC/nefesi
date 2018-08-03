import tkinter as tk# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3
try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk

import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from nefesi.layer_data import ALL_INDEX_NAMES
from nefesi.util.InterfacePlotting import get_plot_net_summary_figure
from .popup_window import PopupWindow

STATES = ['init']
MAX_PLOTS_VISIBLES_IN_WINDOW = 4

class Interface():
    def __init__(self, network_data, title = 'Nefesi'):
        self.network_data = network_data
        self.visible_plots_canvas = np.zeros(MAX_PLOTS_VISIBLES_IN_WINDOW,
                                             dtype=np.dtype([('canvas', np.object), ('used',np.bool),('have_plot',np.bool)]))
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
        self.set_general_buttons_frame('init')
        #self.general_buttons_frame.bind("<Configure>", self._on_resize_plot_canvas)
        tk.Label(self.general_buttons_frame, text="Place to put texts").pack()
        ttk.Button(self.general_buttons_frame, text="button",style="TButton").pack()
        tk.Label(self.general_info_frame, text="Place to put texts").pack()
        self.state = 'init'
        self.plot_general_index(index='class')
        self.plot_general_index(index='class')
        self.plot_general_index(index='class')
        self.plot_general_index(index='class')
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
        self._plots_canvas.pack(side=RIGHT, expand=True,padx=(100,0))

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


    def set_general_buttons_frame(self,state='init'):
        #Title just in top of selector
        combo_title = ttk.Label(self.general_buttons_frame, text="Graphics to show")
        combo_title.pack(expand=False)
        # Options will be 1,2,3,4
        options = [i for i in range(1,MAX_PLOTS_VISIBLES_IN_WINDOW+1)]
        combo = ttk.Combobox(master=self.general_buttons_frame, values=options, state='readonly', width=4)
        #When selection is changed, calls the function _on_number_of_plots_to_show_changed
        combo.bind("<<ComboboxSelected>>", self._on_number_of_plots_to_show_changed)
        combo.set(options[0])
        combo.pack(expand=False)


    def clean_widget(self, widget):
        for child in list(widget.children.values()):
            if list(child.children.values()) == []:
                child.destroy()
            else:
                self.clean_widget(child)


    def plot_general_index(self, index):
        """
        Plots a general graphic of specified index in the general plots section
        :param index: String representing the index to plot. Needs to be one of prensents in
         nefesi.layer_data.ALL_INDEX_NAMES ('color', 'orientation', 'symmetry', 'class' or 'population code')
        :return:
        """
        self.add_figure_to_frame(figure=get_plot_net_summary_figure('class', self.network_data))

    def add_figure_to_frame(self, figure=None):
        first_util_place = np.where(self.visible_plots_canvas['used']==False)[0][0]
        canvas = Canvas(master=self.plots_canvas)
        self.addapt_widget_for_grid(canvas)
        canvas.grid(column=first_util_place%2, row=(first_util_place//2)+1, sticky=SW)
        if figure is not None:
            plot_canvas = FigureCanvasTkAgg(figure, master=canvas)
            self.addapt_widget_for_grid(plot_canvas.get_tk_widget())
            plot_canvas.get_tk_widget().configure(width=600, height=375)
            #plot_canvas.draw()
            plot_canvas.get_tk_widget().grid(row=1,sticky=SW)

        selector = self.get_index_button_general(canvas)
        selector.place(relx=0.25,rely=0)#grid(row=0,column=0, columnspan=2)
        erase_button = self.get_erase_plot_button(master=canvas)
        erase_button.place(relx=0.85,rely=0)#((row=0,column=3)

        if figure is None:
            self.visible_plots_canvas[first_util_place] = (canvas, True,False)
        else:
            self.visible_plots_canvas[first_util_place] = (canvas, True, True)

    """
    def _on_resize_plot_canvas(self,event):
        print("HOLIS")
        w,h = max(self.plots_frame.winfo_width()-100,50), max(self.plots_frame.winfo_height()-100,50)
        plots_widget = event.widget.children['!canvas'].children['!canvas']
        plots_widget.configure(width=w, height=h)
        print("HOLIS")
    """
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
        self.visible_plots_canvas[pos] = (None, False, False)
        plot_canvas.destroy()

    def _on_click_destroy_subplot(self, event):
        #Is the master of the button (the canvas that have button,selector and plot)
        self.destroy_plot_canvas(plot_canvas=event.widget.master)

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
        oldplot = master.children['!canvas']
        self.clean_widget(oldplot)
        oldplot.destroy()
        print ("Change selector event. Acaba esto ya")

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
            print("plot " + str(element_to_erase) + " erased")
            self.destroy_plot_canvas(self.visible_plots_canvas['canvas'][element_to_erase])
            plots_in_use_idx = np.where(self.visible_plots_canvas['used'] == True)[0]
        # Readjust the plots in order to put put ordered. (example: if remains plots 1 and 3 and selected is 2, plots
        # 1 and 3 will be plots 0 and 1)
        if plots_in_use_idx[-1] >= selected:
            idx = [i for i in range(len(self.visible_plots_canvas))]
            valids_idx, non_valids = idx[:selected], idx[selected:]
            self.visible_plots_canvas[valids_idx] = self.visible_plots_canvas[plots_in_use_idx]
            self.visible_plots_canvas[non_valids] = (None, False, False)
            # Readjust in screen too
            for i, canvas in enumerate(self.visible_plots_canvas['canvas'][valids_idx]):
                canvas.grid(column=i % 2, row=(i // 2) + 1, sticky=SW)
                print("plot readjusted")

        # Put new empty places for plot charts
        for i in range(len(plots_in_use_idx), selected):
            self.add_figure_to_frame(figure=None)


if __name__ == '__main__':
    Interface(None)

