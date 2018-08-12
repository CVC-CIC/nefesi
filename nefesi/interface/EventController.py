# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3
try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from Tkinter import ttk

import numpy as np
from nefesi.layer_data import ALL_INDEX_NAMES

STATES = ['init']
MAX_PLOTS_VISIBLES_IN_WINDOW = 4
MAX_VALUES_VISIBLES_IN_LISTBOX = 6
class EventController():
    def __init__(self, interface):
        self.interface = interface

    def _on_click_proceed_button(self):
        current_plots = self.interface.visible_plots_canvas[self.interface.visible_plots_canvas['used']]
        for canvas, _,index, special_value in current_plots:
            if index in ALL_INDEX_NAMES:
                self.interface.plot_general_index(index=index, master_canvas=canvas,
                                                  layers=self.interface.current_layers_in_view,
                                        special_value=special_value)

    def _on_listbox_change_selection(self,event,lstbox):
        selection = lstbox.curselection()
        if len(selection) <= 0:
            selection = self.interface.lstbox_last_selection
            for idx in self.interface.lstbox_last_selection:
                lstbox.select_set(idx)
        #'all' not have sense to be selected with more layer_names
        if 0 in selection and len(selection)>1:
            lstbox.select_clear(selection[1],END)
        self.interface.lstbox_last_selection = selection
        self.interface.current_layers_in_view = self.interface.get_listbox_selection(lstbox)


    def _on_checkbox_clicked(self,checkbox_value):
        self.interface.network_data.save_changes = checkbox_value.get()

    def _on_in_plot_element_double_click(self, event, hidden_annotations, master_canvas, index, special_value):
        """
        When user have double click on bar or neuron, plots the more specific level. A one layer plot if clicks on layer of
        general chart or opens the single neuron windows, if clicks on a neuron
        :param event: the event raised by mpl_connect
        :param hidden_annotations: the hidden annotetions numpy of the chart where user clicked
        :param master_canvas: the master canvas of the plot where user clicks
        :param index: the index calculated in the plot where user clicks
        :param special_value: the special_value of the plot where user clicks
        """
        if event.dblclick:
            if event.inaxes is not None:
                x, y = event.xdata, event.ydata
                for layer_name, neuron_idx, annotation, x0, x1, y0, y1 in hidden_annotations:
                    if x0 < x < x1 and y0 < y < y1:
                        print("click in " + str(layer_name)+". neuron "+str(neuron_idx))
                        if neuron_idx==-1:
                            self.interface.plot_general_index(index=index,master_canvas=master_canvas, layers=layer_name,
                                                          neuron=neuron_idx,special_value=special_value)
                        else:
                            print("MAKE THE F*CKING PLOT OF SINGLE NEURON")
                        break


    def _on_in_plot_element_hover(self, event, hidden_annotations):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            for layer_name,neuron_idx, annotation, x0, x1, y0, y1 in hidden_annotations:
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
        self.interface.destroy_plot_canvas(plot_canvas=event.widget.master)
        self.interface.graphics_to_show_combo.set(np.count_nonzero(self.interface.visible_plots_canvas['used']))

    def _on_general_plot_selector_changed(self,event):
        """
        event called when user selects another chart to show in the combobox of the plot canvas (in general state (init)
        :param event: event with the widget of the combobox changed
        """
        master = event.widget.master
        selected = event.widget.get()
        special_value = None
        if selected in ['orientation','population code'] :
            special_value = self.interface.get_value_from_popup(index=selected)

        self.interface.destroy_canvas_subplot_if_exist(master_canvas=master)
        self.interface.plot_general_index(index=selected, master_canvas=master, layers=self.interface.current_layers_in_view,
                                          special_value=special_value)


    def _on_number_of_plots_to_show_changed(self, event):
        """
        Event called when user change the value of the combobox with... How many charts show? Add place to new charts
        or delete the charts that overflows
        :param event: event with the widget of combobox changed
        """
        # value selected by user
        selected = int(event.widget.get())
        # pos on array self.interface.visible_plots_canvas of plots that are in use now
        plots_in_use_idx = np.where(self.interface.visible_plots_canvas['used'] == True)[0]
        # erase the plots that overflows the new number of plots
        while len(plots_in_use_idx) > selected:
            # index of last plot that not have place for exist in new plots length
            element_to_erase = plots_in_use_idx[-1]
            self.interface.destroy_plot_canvas(self.interface.visible_plots_canvas['canvas'][element_to_erase])
            plots_in_use_idx = np.where(self.interface.visible_plots_canvas['used'] == True)[0]
        # Readjust the plots in order to put put ordered. (example: if remains plots 1 and 3 and selected is 2, plots
        # 1 and 3 will be plots 0 and 1)
        if plots_in_use_idx[-1] >= selected:
            idx = [i for i in range(len(self.interface.visible_plots_canvas))]
            valids_idx, non_valids = idx[:selected], idx[selected:]
            self.interface.visible_plots_canvas[valids_idx] = self.interface.visible_plots_canvas[plots_in_use_idx]
            self.interface.visible_plots_canvas[non_valids] = (None, False,'', None)
            # Readjust in screen too
            for i, canvas in enumerate(self.interface.visible_plots_canvas['canvas'][valids_idx]):
                canvas.grid(column=i % 2, row=(i // 2) + 1, sticky=SW)

        # Put new empty places for plot charts
        for i in range(len(plots_in_use_idx), selected):
            self.interface.add_figure_to_frame(figure=None)