# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3
try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from Tkinter import ttk

from .popup_windows.neuron_window import IMAGE_BIG_DEFAULT_SIZE, IMAGE_SMALL_DEFAULT_SIZE
from ..util.general_functions import clean_widget, mosaic_n_images, destroy_canvas_subplot_if_exist, \
    get_listbox_selection
from .popup_windows.receptive_field_popup_window import ReceptiveFieldPopupWindow

import numpy as np
from PIL import ImageTk, Image

STATES = ['init']
MAX_PLOTS_VISIBLES_IN_WINDOW = 4
MAX_VALUES_VISIBLES_IN_LISTBOX = 6
class EventController():
    def __init__(self, interface):
        self.interface = interface

    def _on_click_proceed_button(self):
        current_plots = self.interface.visible_plots_canvas[self.interface.visible_plots_canvas['used']]
        for canvas, _,index, special_value in current_plots:
            if index in self.interface.network_data.indexs_accepted:
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
        self.interface.current_layers_in_view = get_listbox_selection(lstbox)



    def _on_in_plot_element_double_click(self, event, hidden_annotations, master_canvas, index=None, special_value=None):
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
            if len(hidden_annotations)>0:
                if event.inaxes is not None:
                    x, y = event.xdata, event.ydata
                    if type(hidden_annotations[0]) is np.void:
                        for layer_name, neuron_idx, annotation, x0, x1, y0, y1 in hidden_annotations:
                            if x0 < x < x1 and y0 < y < y1:
                                print("Going to " + str(layer_name)+". neuron "+str(neuron_idx))
                                if neuron_idx==-1:
                                    self.interface.plot_general_index(index=index,master_canvas=master_canvas, layers=layer_name,
                                                                  neuron=neuron_idx,special_value=special_value)
                                else:
                                    self.interface.raise_neuron_window(layer=layer_name,neuron_idx=neuron_idx)
                                break
                    elif type(hidden_annotations[0]) is np.ndarray:
                        for hidden_annotation in hidden_annotations:
                            layer_name, neuron_idx, annotation, x0, x1, y0, y1 = hidden_annotation[0]
                            if x0 < x < x1 and y0 < y < y1:
                                print("click in " + str(layer_name) + ". neuron " + str(neuron_idx))
                                if neuron_idx == -1:
                                    self.interface.plot_general_index(index=index, master_canvas=master_canvas,
                                                                      layers=layer_name,
                                                                      neuron=neuron_idx, special_value=special_value)
                                else:
                                    self.interface.raise_neuron_window(layer=layer_name, neuron_idx=neuron_idx)
                                break




    def _on_in_plot_element_hover(self, event, hidden_annotations):
        if len(hidden_annotations) > 0:
            if event.inaxes is not None:
                x, y = event.xdata, event.ydata
                if type(hidden_annotations[0]) is np.void:
                    for layer_name,neuron_idx, annotation, x0, x1, y0, y1 in hidden_annotations:
                        if x0 < x < x1 and y0 < y < y1:
                            if not annotation.get_visible():
                                annotation.set_visible(True)
                                annotation.figure.canvas.draw()
                        else:
                            if annotation.get_visible():
                                annotation.set_visible(False)
                                annotation.figure.canvas.draw()
                elif type(hidden_annotations[0]) is np.ndarray:
                    for hidden_annotation in hidden_annotations:
                        for layer_name, neuron_idx, annotation, x0, x1, y0, y1 in hidden_annotation:
                            if x0 < x < x1 and y0 < y < y1:
                                if not annotation.get_visible():
                                    annotation.set_visible(True)
                                    annotation.figure.canvas.draw()
                            else:
                                if annotation.get_visible():
                                    annotation.set_visible(False)
                                    annotation.figure.canvas.draw()
            else:
                if type(hidden_annotations[0]) is np.void:
                    for annotation in hidden_annotations['annotation']:
                        if annotation.get_visible():
                            annotation.set_visible(False)
                            annotation.figure.canvas.draw()
                elif type(hidden_annotations[0]) is np.ndarray:
                    for hidden_annotation in hidden_annotations:
                        for annotation in hidden_annotation['annotation']:
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

        destroy_canvas_subplot_if_exist(master_canvas=master)
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
        if plots_in_use_idx != [] and plots_in_use_idx[-1] >= selected:
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





#---------------NEURON WINDOW EVENTS-------------------------------

    def _on_decrease_click(self, panel, image_num_label, activation_label, norm_activation_label, class_label):
        self.interface.actual_img_index = (self.interface.actual_img_index - 1) % len(self.interface.neuron.images_id)
        self.interface.update_decomposition_label(activation_label, class_label, image_num_label, norm_activation_label)
        self.interface.update_decomposition_panel(panel=panel)


    def _on_increase_click(self, panel, image_num_label, activation_label, norm_activation_label, class_label):
        self.interface.actual_img_index = (self.interface.actual_img_index + 1) % len(self.interface.neuron.images_id)
        self.interface.update_decomposition_label(activation_label, class_label, image_num_label, norm_activation_label)
        self.interface.update_decomposition_panel(panel=panel)

    def _on_link_decrease_click(self, panel, neuron_label, relevance_label):
        self.interface.actual_link = (self.interface.actual_link - 1) % len(self.interface.relevance)
        self.interface.update_decomposition_label(neuron_label, relevance_label)
        self.interface.update_decomposition_panel(panel=panel)


    def _on_link_increase_click(self, panel, neuron_label, relevance_label):
        self.interface.actual_link = (self.interface.actual_link + 1) % len(self.interface.relevance)
        self.interface.update_decomposition_label(neuron_label, relevance_label)
        self.interface.update_decomposition_panel(panel=panel)


    def _on_image_click(self, event,layer_name, idx_neuron):
        # layer_data = self.interface.network_data.get_layer_by_name(layer=self.interface.layer_to_evaluate)
        layer_data = self.interface.network_data.get_layer_by_name(layer=layer_name)
        actual_idx = self.interface.actual_img_index
        receptive_field = layer_data.receptive_field_map
        y0, y1, x0, x1 = receptive_field[self.interface.neuron.xy_locations[actual_idx, 0],
                                         self.interface.neuron.xy_locations[actual_idx, 1]]
        x_len, y_len = x1 - x0, y1 - y0


        mosaic_n_images(self.interface.neuron.get_patches(network_data=self.interface.network_data,
                                                layer_data=self.interface.network_data.get_layer_by_name(
                                                    self.interface.layer_to_evaluate)))
        image_name = self.interface.neuron.images_id[actual_idx]

        cropped_image = self.interface.neuron.get_patch_by_idx(self.interface.network_data,
                                                     self.interface.network_data.get_layer_by_name(self.interface.layer_to_evaluate),
                                                     actual_idx)
        cropped_image = Image.fromarray(cropped_image).resize(self.interface.image_actual_size, Image.ANTIALIAS)  # resize mantaining aspect ratio
        np_cropped = np.array(cropped_image)
        np_cropped = self.interface.draw_rectangle_on_image(np_cropped, 2, np_cropped.shape[0]-3, 2, np_cropped.shape[1]-3,
                                                  margin=2,
                                                  draw_lines=False)

        cropped_image = Image.fromarray(np_cropped.astype('uint8'), 'RGB')
        cropped_image = ImageTk.PhotoImage(cropped_image)
        ReceptiveFieldPopupWindow(master=self.interface.window, image_cropped=cropped_image,
                                  x_len=x_len, y_len=y_len,
                                  image_name=image_name, layer_name=layer_name,neuron_idx=idx_neuron,
                                interface = self.interface,x0=x0,x1=x1,y0=y0,y1=y1, actual_idx = actual_idx)
    def _on_nf_changed(self, event, combo):
        selection = combo.get()
        self.interface.set_nf_panel(option=selection)

    def _on_checkbox_clicked(self, checkbox_value, checkbox_img_value = None):
        if checkbox_img_value is not None and checkbox_img_value.get():
            checkbox_img_value.set(False)
            self._on_expand_images_checkbox_clicked(checkbox_value=checkbox_img_value)
        if checkbox_value.get():
            self.interface.advanced_plots_frame = Frame(master=self.interface.window)
            self.interface.advanced_plots_frame.pack(side=BOTTOM)
            self.interface.add_figure_to_frame(master_canvas=self.interface.advanced_plots_frame, figure=None)
        else:
            clean_widget(self.interface.advanced_plots_frame)
            self.interface.advanced_plots_frame.destroy()
            self.interface.advanced_plots_frame = None
            self.interface.advanced_plots_canvas = None

    def _on_expand_images_checkbox_clicked(self, checkbox_value, checkbox_advanced_charts = None):
        if checkbox_advanced_charts is not None and checkbox_advanced_charts.get():
            checkbox_advanced_charts.set(False)
            self._on_checkbox_clicked(checkbox_value=checkbox_advanced_charts)

        self.interface.image_actual_size = IMAGE_BIG_DEFAULT_SIZE if checkbox_value.get() else IMAGE_SMALL_DEFAULT_SIZE
        self.interface.update_images_size()
