from nefesi.interface.popup_windows.one_layer_popup_window import OneLayerPopupWindow

IMAGE_DEFAULT_SIZE = (350,350)
ADVANCED_CHARTS = ['Activation Curve', 'Similar Neurons']
#That have with images of A are the column of A

import tkinter as tk# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.draw import line_aa
import math
from nefesi.class_index import get_path_sep
from nefesi.util.interface_plotting import get_one_neuron_plot, plot_similar_neurons

try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk
import numpy as np

from nefesi.util.general_functions import clean_widget, mosaic_n_images, add_red_separations, \
    destroy_canvas_subplot_if_exist, addapt_widget_for_grid
from PIL import ImageTk, Image
from nefesi.interface.EventController import EventController
import nefesi.util.plotting as plotting

class NeuronWindow(object):
    def __init__(self, master, network_data, layer_to_evaluate, neuron_idx):
        self.event_controller = EventController(self)
        self.network_data = network_data
        self.layer_to_evaluate = layer_to_evaluate
        self.neuron_idx = neuron_idx
        self.actual_img_index = 0
        self.mosaic = None
        self.advanced_plots_canvas = None
        self.selector = None
        self.advanced_plots_frame = None
        self.neuron = self.network_data.get_neuron_of_layer(layer=layer_to_evaluate, neuron_idx=neuron_idx)
        self.window=Toplevel(master)
        self.basic_frame = Frame(master=self.window)
        self.window.title(str(layer_to_evaluate) + ' Neuron: '+str(neuron_idx))
        self.index_info = Frame(master=self.basic_frame)
        self.set_index_info(master=self.index_info)
        self.images_frame = Frame(master=self.basic_frame)
        self.neuron_feature_frame = Frame(master=self.images_frame)
        self.decomposition_frame = Frame(master=self.images_frame)

        self.set_neuron_feature_frame(master=self.neuron_feature_frame)
        self.neuron_feature_frame.pack(side=LEFT, fill=BOTH, expand=True)
        self.decomposition_frame = Frame(self.basic_frame)
        self.set_decomposition_frame(self.decomposition_frame)
        self.images_frame.pack(side=LEFT,padx=5)
        self.neuron_feature_frame.pack(side=LEFT,padx=5)
        self.index_info.pack(side=RIGHT)
        self.decomposition_frame.pack(side=RIGHT)
        self.basic_frame.pack(side=TOP)


    def set_decomposition_frame(self, master):
        image_frame = Frame(master=master)
        image_num_label, activation_label, norm_activation_label, class_label = self.set_decomposition_label(master)
        neuron_feature = self.neuron.get_patch_by_idx(self.network_data,
                                                      self.network_data.get_layer_by_name(self.layer_to_evaluate),
                                                      self.actual_img_index)
        neuron_feature = neuron_feature.resize(IMAGE_DEFAULT_SIZE, Image.ANTIALIAS)  # resize mantaining aspect ratio
        img = ImageTk.PhotoImage(neuron_feature)

        self.panel_image = Label(master=image_frame, image=img)
        self.panel_image.image = img
        self.panel_image.bind("<Double-Button-1>", lambda event:self.event_controller._on_image_click(event,self.panel_image.image))
        decrease_button = Button(master=image_frame, text='◀', command=lambda: self.event_controller._on_decrease_click(self.panel_image,
                                        image_num_label,activation_label,norm_activation_label,class_label))
        increase_button = Button(master=image_frame, text='▶', command=lambda: self.event_controller._on_increase_click(self.panel_image,
                                        image_num_label,activation_label,norm_activation_label,class_label))
        increase_button.pack(side=RIGHT, fill='y')
        decrease_button.pack(side=LEFT,fill='y')
        self.panel_image.pack(side=RIGHT)
        image_frame.pack(side=BOTTOM)

    def set_decomposition_label(self, master):
        text_frame = Frame(master=master)
        activation = self.neuron.activations[self.actual_img_index]
        norm_activation = self.neuron.norm_activations[self.actual_img_index]
        label = self.get_current_image_class()
        Label(master=text_frame, text="Image", font='Helvetica 10').pack(side=LEFT)
        self.image_num_label = Label(master=text_frame, text=str(self.actual_img_index), font='Helvetica 10 bold')
        self.image_num_label.pack(side=LEFT)
        Label(master=text_frame, text="Act.:", font='Helvetica 10').pack(side=LEFT)
        self.activation_label = Label(master=text_frame, text=str(round(activation, ndigits=2)), font='Helvetica 10 bold')
        self.activation_label.pack(side=LEFT)
        Label(master=text_frame, text="Norm. Act.:", font='Helvetica 10').pack(side=LEFT)
        self.norm_activation_label = Label(master=text_frame, text=str(round(norm_activation, ndigits=2)),
                                 font='Helvetica 10 bold')
        self.norm_activation_label.pack(side=LEFT)
        Label(master=text_frame, text="Class:", font='Helvetica 10').pack(side=LEFT)
        self.class_label = Label(master=text_frame, text=label,
                                      font='Helvetica 10 bold')
        self.class_label.pack(side=LEFT)
        text_frame.pack(side=TOP)
        return self.image_num_label, self.activation_label, self.norm_activation_label, self.class_label

    def get_current_image_class(self):
        image_name = self.neuron.images_id[self.actual_img_index]
        path_sep = get_path_sep(image_name)
        label = image_name[:image_name.index(path_sep)]
        return label



    def update_decomposition_label(self, activation_label, class_label, image_num_label, norm_activation_label):
        image_num_label.configure(text=self.actual_img_index)
        activation_label.configure(text=str(round(self.neuron.activations[self.actual_img_index], ndigits=2)))
        norm_activation_label.configure(text=str(round(self.neuron.norm_activations[self.actual_img_index], ndigits=2)))
        class_label.configure(text=self.get_current_image_class())

    def update_decomposition_panel(self, panel):
        new_image = self.neuron.get_patch_by_idx(self.network_data,
                                                      self.network_data.get_layer_by_name(self.layer_to_evaluate),
                                                      self.actual_img_index)
        new_image = new_image.resize(IMAGE_DEFAULT_SIZE,Image.ANTIALIAS)
        img = ImageTk.PhotoImage(new_image)
        panel.configure(image=img)
        panel.image = img

    def draw_rectangle_on_image(self, np_image, x0, x1,y0,y1,margin=3, draw_lines=True):
        red = [255,0,0]
        x0 = max(x0-1,margin)
        x1 = min(x1 + 1, np_image.shape[1]-margin)
        y0 = max(y0 - 1, margin)
        y1 = min(y1 + 1, np_image.shape[0]-margin)
        for i in range(margin):
            np_image[x0:x1+1,y0-i,:],np_image[x0:x1+1,y1+i,:],np_image[x0-i,y0:y1+1,:],np_image[x1+i,y0:y1+1,:] = red,red,red,red
        if draw_lines:
            rr, cc, _ = line_aa(x0, y1, 0, np_image.shape[0]-2)
            np_image[rr, cc,:] = red

            rr, cc, _ = line_aa(x1, y1, np_image.shape[1]-2, np_image.shape[0]-2)
            np_image[rr, cc,:] = red
        return np_image

    #TODO: Tracts to resize dinamically
    def set_neuron_feature_frame(self, master):
        options = ['Neuron Feature', 'Images Mosaic']
        combo = ttk.Combobox(master=master, values=options, state='readonly', width=15, justify=CENTER)
        combo.set(options[0])
        # When selection is changed, calls the function _on_number_of_plots_to_show_changed
        combo.bind("<<ComboboxSelected>>", lambda event: self.event_controller._on_nf_changed(event, combo))
        combo.pack(side=TOP)
        self.panel_nf = Label(master=master)
        self.set_nf_panel()

    def _on_mosaic_click(self, event):
        images_per_axis = math.ceil(math.sqrt(len(self.neuron.activations)))
        x = int((((event.x-1)/IMAGE_DEFAULT_SIZE[0])*images_per_axis))
        y = int((((event.y-1)/IMAGE_DEFAULT_SIZE[-1])*images_per_axis))
        #to avoid margins
        if -1<x<images_per_axis or -1<y<images_per_axis:
            num_image = y*images_per_axis+x
            self.actual_img_index = num_image
            self.update_decomposition_label(self.image_num_label, self.activation_label,
                                                      self.norm_activation_label, self.class_label)
            self.update_decomposition_panel(panel=self.panel_image)


    def set_nf_panel(self,option='neuron feature'):
        option= option.lower()
        if option=='neuron feature':
            img = self.neuron._neuron_feature
            img = img.resize(IMAGE_DEFAULT_SIZE, Image.ANTIALIAS)
            self.panel_nf.unbind('<Double-Button-1>')
        elif option=='images mosaic':
            if self.mosaic != None:
                img = self.mosaic
            else:
                img = mosaic_n_images(self.neuron.get_patches(network_data=self.network_data,
                                            layer_data=self.network_data.get_layer_by_name(self.layer_to_evaluate)))
                img = Image.fromarray(img.astype('uint8'),'RGB')
                img = img.resize(IMAGE_DEFAULT_SIZE, Image.ANTIALIAS)
                img = np.array(img)
                img = add_red_separations(img, math.ceil(math.sqrt(len(self.neuron.activations))))
                self.mosaic = img = Image.fromarray(img.astype('uint8'), 'RGB')
            self.panel_nf.bind('<Double-Button-1>', self._on_mosaic_click)

        img = ImageTk.PhotoImage(img)
        self.panel_nf.configure(image=img)
        self.panel_nf.image= img
        self.panel_nf.pack(side=BOTTOM, fill=BOTH, expand=True)


    def _on_resize_image(self, event, panel,neuron_feature,top_widget):
        panel.update()
        h, w = panel.winfo_height()-top_widget.winfo_height(),\
               panel.winfo_width()
        if h>0:
            neuron_feature = neuron_feature.resize((min(h,w), min(h,w)), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(neuron_feature)
            panel.configure(image=img)
            panel.image = img
            panel.pack(side=BOTTOM, fill=BOTH, expand=True)

    def set_index_info(self, master, orientation_degrees=90,thr_class_idx=1., thr_pc=0.1):
        indexs = self.get_index_info()
        Label(master=master, text='Selectivity Indexs: ').grid(column=0, row=0)
        for i, (label, idx) in enumerate(indexs.items()):
            if label == 'color':
                text = ' Color: '+ str(round(idx,ndigits=3))
            elif label == 'orientation':
                text = ' Orientation ('+str(orientation_degrees)+'º): ' \
                                        'μ='+ str(round(idx[-1],ndigits=3))+' σ='+str(round(np.std(idx[:-1]),ndigits=3))
            elif label == 'symmetry':
                text = ' Symmetry: μ=' + str(round(idx[-1], ndigits=3)) + ' σ=' + str(round(np.std(idx[:-1]), ndigits=3))
            elif label == 'class':
                text = ' Class: '+ str(round(idx[-1], ndigits=3))+' ('+str(idx[0])+')'
            elif label == 'population code':
                text = ' Population code (thr='+str(thr_pc)+'): '+str(idx)
            Label(master=master, text=text, justify=LEFT).grid(column=0, row=i+1)
        checkbox_value = tk.BooleanVar(master=master)
        checkbox = ttk.Checkbutton(master=master, text="Show advanced charts", variable=checkbox_value,
                                    command= lambda: self.event_controller._on_checkbox_clicked(checkbox_value))
        checkbox.grid(column=0, row=len(indexs.items())+1)

    def get_index_info(self,orientation_degrees=90, thr_class_idx=1., thr_pc=0.1):
        indexs = self.network_data.get_all_index_of_neuron(layer=self.layer_to_evaluate, neuron_idx=self.neuron_idx,
                                                           orientation_degrees=orientation_degrees,
                                                           thr_class_idx=thr_class_idx,thr_pc=thr_pc)
        return indexs

    def add_figure_to_frame(self, master_canvas=None, figure=None, default_value=None):
        self.combo_frame = Frame(master=master_canvas)
        self.figure_frame = Frame(master=master_canvas)
        if figure is not None:
            self.put_figure_plot(master=self.figure_frame, figure=figure)
        self.selector = self.get_index_button_general(self.combo_frame, default_value=default_value)
        self.selector.pack()  # grid(row=0,column=0, columnspan=2)
        self.combo_frame.pack(side=TOP)
        self.figure_frame.pack(side=BOTTOM)

    def put_figure_plot(self, master, figure,  hidden_annotations=None):
        destroy_canvas_subplot_if_exist(master_canvas=master)
        plot_canvas = FigureCanvasTkAgg(figure, master=master)
        addapt_widget_for_grid(plot_canvas.get_tk_widget())
        plot_canvas.get_tk_widget().configure(width=800, height=450)
        plot_canvas.get_tk_widget().grid(row=1, sticky=SW)
        if hidden_annotations is not None:
            plot_canvas.mpl_connect('motion_notify_event',
                                    lambda event: self.event_controller._on_in_plot_element_hover(event, hidden_annotations))
            plot_canvas.mpl_connect('button_press_event',
                                lambda event: self.event_controller._on_in_plot_element_double_click(event, hidden_annotations,
                                                                                                     master))
    def get_index_button_general(self, master, default_value = None):
        """
        Gets a general button to select wich graphic to plot
        :return: A select button with each index possible, and the event to plot it when called
        """

        combo = ttk.Combobox(master=master, values=ADVANCED_CHARTS, state='readonly',justify=CENTER,width=15)
        combo.bind("<<ComboboxSelected>>",lambda event: self._on_general_plot_selector_changed(event, combo))
        if default_value is not None:
            combo.set(default_value)
        else:
            combo.set('Select Chart')
        return combo

    def _on_general_plot_selector_changed(self, event,combo):
        selected = combo.get()
        if selected.lower() == 'similar neurons':
            min, condition1, max, condition2, order, max_neurons = \
                self.get_similarity_params_from_popup(layer_to_evaluate=self.layer_to_evaluate)
            figure, hidden_annotations = plot_similar_neurons(network_data=self.network_data, layer=self.layer_to_evaluate,
                                          neuron_idx=self.neuron_idx,min=min, max=max, condition1=condition1,
                                          condition2=condition2, order=order, max_neurons=max_neurons)
        else:
            figure = get_one_neuron_plot(network_data=self.network_data,layer=self.layer_to_evaluate,
                                     neuron_idx=self.neuron_idx, chart=selected)
            hidden_annotations = None
        self.put_figure_plot(master=self.figure_frame,figure=figure, hidden_annotations=hidden_annotations)


    def get_similarity_params_from_popup(self,index='similarity',layer_to_evaluate='unknow'):
        if type(layer_to_evaluate) is list:
            layer_to_evaluate = layer_to_evaluate[0]
        popup_window = OneLayerPopupWindow(self.window, layer_to_evaluate=layer_to_evaluate, index=index)
        self.window.wait_window(popup_window.top)
        return popup_window.value1,popup_window.condition1,popup_window.value2,popup_window.condition2,\
               popup_window.order, popup_window.neurons_to_show

    def raise_neuron_window(self, layer, neuron_idx):
        NeuronWindow(master=self.window, network_data=self.network_data, layer_to_evaluate=layer, neuron_idx=neuron_idx)