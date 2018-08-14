IMAGE_DEFAULT_SIZE = (350,350)

import tkinter as tk# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.draw import line_aa
import math
from nefesi.class_index import get_path_sep

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

class NeuronWindow(object):
    def __init__(self, master, network_data, layer_to_evaluate, neuron_idx):
        self.event_controller = EventController(self)
        self.network_data = network_data
        self.layer_to_evaluate = layer_to_evaluate
        self.neuron_idx = neuron_idx
        self.actual_img_index = 0
        self.mosaic = None
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

        panel = Label(master=image_frame, image=img)
        panel.image = img
        panel.bind("<Double-Button-1>", lambda event:self.event_controller._on_image_click(event,panel.image))
        decrease_button = Button(master=image_frame, text='<-', command=lambda: self.event_controller._on_decrease_click(panel,
                                        image_num_label,activation_label,norm_activation_label,class_label))
        increase_button = Button(master=image_frame, text='->', command=lambda: self.event_controller._on_increase_click(panel,
                                        image_num_label,activation_label,norm_activation_label,class_label))
        increase_button.pack(side=RIGHT, fill='y')
        decrease_button.pack(side=LEFT,fill='y')
        panel.pack(side=RIGHT)
        image_frame.pack(side=BOTTOM)

    def set_decomposition_label(self, master):
        text_frame = Frame(master=master)
        activation = self.neuron.activations[self.actual_img_index]
        norm_activation = self.neuron.norm_activations[self.actual_img_index]
        label = self.get_current_image_class()
        Label(master=text_frame, text="Image", font='Helvetica 10').pack(side=LEFT)
        image_num_label = Label(master=text_frame, text=str(self.actual_img_index), font='Helvetica 10 bold')
        image_num_label.pack(side=LEFT)
        Label(master=text_frame, text="Act.:", font='Helvetica 10').pack(side=LEFT)
        activation_label = Label(master=text_frame, text=str(round(activation, ndigits=2)), font='Helvetica 10 bold')
        activation_label.pack(side=LEFT)
        Label(master=text_frame, text="Norm. Act.:", font='Helvetica 10').pack(side=LEFT)
        norm_activation_label = Label(master=text_frame, text=str(round(norm_activation, ndigits=2)),
                                 font='Helvetica 10 bold')
        norm_activation_label.pack(side=LEFT)
        Label(master=text_frame, text="Class:", font='Helvetica 10').pack(side=LEFT)
        class_label = Label(master=text_frame, text=label,
                                      font='Helvetica 10 bold')
        class_label.pack(side=LEFT)
        text_frame.pack(side=TOP)
        return image_num_label, activation_label, norm_activation_label,class_label

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
        #self.window.bind('<Configure>', lambda event: self._on_resize_image(event, panel, neuron_feature, label))


    def set_nf_panel(self,option='neuron feature'):
        option= option.lower()
        if option=='neuron feature':
            img = self.neuron._neuron_feature
            img = img.resize(IMAGE_DEFAULT_SIZE, Image.ANTIALIAS)
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
        if figure is not None:
            self.put_figure_plot(master=master_canvas, figure=figure)
        selector = self.get_index_button_general(master_canvas, default_value=default_value)
        selector.place(relx=0.4, rely=0)  # grid(row=0,column=0, columnspan=2)

    def put_figure_plot(self, master, figure):
        destroy_canvas_subplot_if_exist(master_canvas=master)
        plot_canvas = FigureCanvasTkAgg(figure, master=master)
        addapt_widget_for_grid(plot_canvas.get_tk_widget())
        plot_canvas.get_tk_widget().configure(width=800, height=450)
        plot_canvas.get_tk_widget().grid(row=1, sticky=SW)

    def get_index_button_general(self, master, default_value = None):
        """
        Gets a general button to select wich graphic to plot
        :return: A select button with each index possible, and the event to plot it when called
        """
        combo = ttk.Combobox(master=master, values=['a','b','c'], state='readonly',justify=CENTER,width=15)
        #combo.bind("<<ComboboxSelected>>", self.event_controller._on_general_plot_selector_changed)
        if default_value is not None:
            combo.set(default_value)
        else:
            combo.set('Select Index')
        return combo