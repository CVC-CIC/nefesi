from anytree import RenderTree

from .one_layer_popup_window import OneLayerPopupWindow

MAX_CONCEPTS_TO_SHOW = 4

IMAGE_BIG_DEFAULT_SIZE = (800,800)
IMAGE_SMALL_DEFAULT_SIZE = (450,450)
ADVANCED_CHARTS = ['Activation Curve', 'Similar Neurons', 'Relevant Neurons']
TREE_THRESHOLD = 50
#That have with images of A are the column of A

import tkinter as tk# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.draw import line_aa
import math

from ...class_index import get_path_sep,get_hierarchical_population_code_idx
from ...util.interface_plotting import get_one_neuron_plot, plot_similar_neurons, plot_relevant_neurons
from .combobox_popup_window import ComboboxPopupWindow

try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk
import numpy as np

from ...util.general_functions import mosaic_n_images, add_red_separations, destroy_canvas_subplot_if_exist,\
    addapt_widget_for_grid
from PIL import ImageTk, Image
from ..EventController import EventController

class NeuronWindow(object):
    def __init__(self, master, network_data, layer_to_evaluate, neuron_idx, image_actual_size=IMAGE_SMALL_DEFAULT_SIZE):
        self.event_controller = EventController(self)
        self.network_data = network_data
        self.layer_to_evaluate = layer_to_evaluate
        self.layer_data = self.network_data.get_layer_by_name(layer=layer_to_evaluate)
        self.neuron_idx = neuron_idx
        self.image_actual_size = image_actual_size
        self.with_activations = True
        self.actual_img_index = 0
        self.mosaic = None
        self.advanced_plots_canvas = None
        self.selector = None
        self.advanced_plots_frame = None
        self.neuron = self.network_data.get_neuron_of_layer(layer=layer_to_evaluate, neuron_idx=neuron_idx)
        self.window = Toplevel(master)
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
        self.set_nf_frame(self.decomposition_frame)
        self.images_frame.pack(side=LEFT,padx=5)
        self.neuron_feature_frame.pack(side=LEFT,padx=5)
        self.index_info.pack(side=RIGHT)
        self.decomposition_frame.pack(side=RIGHT)
        self.basic_frame.pack(side=TOP)


    def update_images_size(self):
        self.set_nf_panel(option=self.combo_nf_option.get())
        self.update_decomposition_panel(panel=self.panel_image)

    def set_nf_frame(self, master):
        image_frame2 = Frame(master=master)

        current_image2 = self.neuron._neuron_feature
        current_image2 = current_image2.resize((int(self.image_actual_size[0] / 2), int(self.image_actual_size[1] / 2)),
                                               Image.ANTIALIAS)  # resize mantaining aspect ratio
        img = ImageTk.PhotoImage(current_image2)

        self.panel_image2 = Label(master=image_frame2, image=img)
        self.panel_image2.image = img
        self.panel_image2.pack(side=BOTTOM)
        image_frame2.pack(side=BOTTOM)




    def set_decomposition_frame(self, master):
        image_frame = Frame(master=master)
        image_num_label, activation_label, norm_activation_label, class_label = self.set_decomposition_label(master)
        current_image = self.neuron.get_patch_by_idx(self.network_data,
                                                      self.network_data.get_layer_by_name(self.layer_to_evaluate),
                                                      self.actual_img_index)
        current_image = current_image.resize((int(self.image_actual_size[0]/2),int(self.image_actual_size[1]/2)), Image.ANTIALIAS)  # resize mantaining aspect ratio
        img = ImageTk.PhotoImage(current_image)

        self.panel_image = Label(master=image_frame, image=img)
        self.panel_image.image = img
        self.panel_image.bind("<Double-Button-1>",lambda event: self.event_controller._on_image_click(event, self.layer_to_evaluate, self.neuron_idx))
        decrease_button = Button(master=image_frame, text='<', command=lambda: self.event_controller._on_decrease_click(self.panel_image,
                                        image_num_label,activation_label,norm_activation_label,class_label))
        increase_button = Button(master=image_frame, text='>', command=lambda: self.event_controller._on_increase_click(self.panel_image,
                                        image_num_label,activation_label,norm_activation_label,class_label))
        increase_button.pack(side=RIGHT, fill='y')
        decrease_button.pack(side=LEFT,fill='y')
        self.panel_image.pack(side=TOP)
        image_frame.pack(side=TOP)

    def set_decomposition_label(self, master):
        text_frame = Frame(master=master)
        activation = self.neuron.activations[self.actual_img_index]
        norm_activation = self.neuron.norm_activations[self.actual_img_index]
        label = self.get_current_image_class()
        Label(master=text_frame, text="Image", font='Helvetica 10').pack(side=LEFT)
        self.image_num_label = Label(master=text_frame, text=str(self.actual_img_index), font='Helvetica 10 bold')
        self.image_num_label.pack(side=LEFT)
        Label(master=text_frame, text="Class:", font='Helvetica 10').pack(side=LEFT)
        self.class_label = Label(master=text_frame, text=label,
                                      font='Helvetica 10 bold')
        self.class_label.pack(side=LEFT)
        Label(master=text_frame, text="Act.:", font='Helvetica 10').pack(side=LEFT)
        self.activation_label = Label(master=text_frame, text=str(round(activation, ndigits=2)), font='Helvetica 10 bold')
        self.activation_label.pack(side=LEFT)
        Label(master=text_frame, text="Norm. Act.:", font='Helvetica 10').pack(side=LEFT)
        self.norm_activation_label = Label(master=text_frame, text=str(round(norm_activation, ndigits=2)),
                                 font='Helvetica 10 bold')
        self.norm_activation_label.pack(side=LEFT)
        text_frame.pack(side=TOP)
        return self.image_num_label, self.activation_label, self.norm_activation_label, self.class_label

    def get_current_image_class(self):
        image_name = self.neuron.images_id[self.actual_img_index]
        path_sep = get_path_sep(image_name)
        label = image_name[:image_name.index(path_sep)]
        if self.network_data.default_labels_dict is not None:
            label = self.network_data.default_labels_dict[label]
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
        new_image = new_image.resize((int(self.image_actual_size[0]/2),int(self.image_actual_size[1]/2)),Image.ANTIALIAS)
        img = ImageTk.PhotoImage(new_image)
        panel.configure(image=img)
        panel.image = img

    def draw_rectangle_on_image(self, np_image, x0, x1, y0, y1,margin=2, draw_lines=True):
        red = [255,0,0]
        x0 = max(x0-1,0)
        x1 = min(x1 + 1, np_image.shape[1]-1)
        y0 = max(y0 - 1, 0)
        y1 = min(y1 + 1, np_image.shape[0]-1)
        for i in range(margin):
            np_image[y0:y1+1,max(x0-i,0),:],\
            np_image[y0:y1+1,min(x1+i, np_image.shape[1]-1),:],\
            np_image[max(y0-i,0),x0:x1+1,:],\
            np_image[min(y1+i, np_image.shape[0]-1),x0:x1+1,:] = red,red,red,red
        if draw_lines:
            rr, cc, _ = line_aa(y0, x1, 0, np_image.shape[1]-2)
            np_image[rr, cc,:] = red

            rr, cc, _ = line_aa(y1, x1, np_image.shape[0]-2, np_image.shape[1]-2)
            np_image[rr, cc,:] = red
        return np_image

    #TODO: Tracts to resize dinamically
    def set_neuron_feature_frame(self, master):
        options = ['Neuron Feature', 'Images Mosaic']
        self.combo_nf_option = ttk.Combobox(master=master, values=options, state='readonly', width=15, justify=CENTER)
        self.combo_nf_option.set(options[0])
        # When selection is changed, calls the function _on_number_of_plots_to_show_changed
        self.combo_nf_option.bind("<<ComboboxSelected>>", lambda event: self.event_controller._on_nf_changed(event, self.combo_nf_option))
        self.combo_nf_option.pack(side=TOP)
        self.panel_nf = Label(master=master)
        self.set_nf_panel()

    def _on_mosaic_click(self, event):
        images_per_axis = math.ceil(math.sqrt(len(self.neuron.activations)))
        x = int((((event.x-1)/self.image_actual_size[0])*images_per_axis))
        y = int((((event.y-1)/self.image_actual_size[-1])*images_per_axis))
        #to avoid margins
        if -1<x<images_per_axis or -1<y<images_per_axis:
            num_image = y*images_per_axis+x
            self.actual_img_index = num_image
            self.update_decomposition_label(activation_label=self.activation_label, class_label=self.class_label,
                                            image_num_label=self.image_num_label,
                                            norm_activation_label=self.norm_activation_label)
            self.update_decomposition_panel(panel=self.panel_image)


    def set_nf_panel(self,option='images mosaic'):
        option= option.lower()
        if option=='neuron feature':
            img = self.neuron._neuron_feature
            img = img.resize(self.image_actual_size, Image.ANTIALIAS)
            self.panel_nf.unbind('<Double-Button-1>')
        elif option=='images mosaic':
            if self.mosaic != None:
                img = self.mosaic.resize(self.image_actual_size, Image.ANTIALIAS)
            else:
                img = mosaic_n_images(self.neuron.get_patches(network_data=self.network_data,
                                            layer_data=self.network_data.get_layer_by_name(self.layer_to_evaluate)))
                img = Image.fromarray(img.astype('uint8'),'RGB')
                img = img.resize(self.image_actual_size, Image.ANTIALIAS)
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

    def set_index_info(self, master, orientation_degrees=None, thr_pc=None, concept='object', plot_wordnet_tree=True):
        """
        plot in window the next indexes up to ['symmetry', 'orientation', 'color', 'class', 'population code', 'object']
        :param master:
        :param orientation_degrees:
        :param thr_pc:
        :param concept:
        :return:
        """
        if orientation_degrees is None:
            orientation_degrees = self.network_data.default_degrees_orientation_idx
        if thr_pc is None:
            thr_pc = self.network_data.default_thr_pc

        indexes = self.get_index_info(orientation_degrees=orientation_degrees, thr_pc=thr_pc)

        Label(master=master, text='Selectivity Indexes: ').grid(column=0, row=0)
        rows = 0
        for i, (label, idx) in enumerate(indexes.items()):
            if label == 'color':
                text = self.get_text_for_composed_index(label,idx)
                text += '[Ivet Color idx: '+ str(round(indexes['ivet_color'],ndigits=3))+']\n'
            elif label in ['class', 'object', 'part']:
                text = self.get_text_for_composed_index(label,idx)
            elif label == 'orientation':
                text = ' Orientation ('+str(orientation_degrees)+'º): ' \
                                        'μ='+ str(round(idx[-1],ndigits=3))+' σ='+str(round(np.std(idx[:-1]),ndigits=3))+'\n'
            elif label == 'symmetry':
                text = ' Symmetry: μ=' + str(round(idx[-1], ndigits=3)) + ' σ=' + str(round(np.std(idx[:-1]), ndigits=3))+'\n'
            else:
                continue
            rows+=1
            Label(master=master, text=text, justify=LEFT).grid(column=0, row=rows)

        if indexes['class']['label'][0] != 'None' and plot_wordnet_tree:
            try:
                tree = get_hierarchical_population_code_idx(
                    self.network_data.get_neuron_of_layer(layer=self.layer_to_evaluate,
                                                          neuron_idx=self.neuron_idx),
                    threshold_pc=thr_pc,
                    population_code=len(indexes['class']),
                    class_sel=round(np.sum(indexes['class']['value'])))
                text = 'Semantical Hierarchy: \n'
                for pre, _, node in RenderTree(tree):
                    name = node.name if type(node.name) is str else node.name[0]
                    treestr = u"%s%s" % (pre, name)
                    if len(treestr) > TREE_THRESHOLD:
                        treestr = treestr[:TREE_THRESHOLD - 3] + '...'
                    text += treestr.ljust(TREE_THRESHOLD) + ' ' + str(node.rep) + ' (' + str(
                        round(node.freq, 2)) + ')\n'
                text += '\n'
                Label(master=master, text=text, justify=LEFT).grid(column=0, row=rows+1)
            except:
                pass


        checkbox_img_value = tk.BooleanVar(master=master)
        checkbox_advanced_charts_value = tk.BooleanVar(master=master)
        checkbox = ttk.Checkbutton(master=master, text="Expand Images", variable=checkbox_img_value,
                                   command=lambda: self.event_controller._on_expand_images_checkbox_clicked
                                   (checkbox_img_value, checkbox_advanced_charts_value))
        checkbox.grid(column=0, row=len(indexes.items()) + 1)
        checkbox = ttk.Checkbutton(master=master, text="Show advanced charts", variable=checkbox_advanced_charts_value,
                                    command= lambda: self.event_controller._on_checkbox_clicked
                                    (checkbox_advanced_charts_value,checkbox_img_value))
        checkbox.grid(column=0, row=len(indexes.items())+2)

    def get_text_for_composed_index(self, index_name, index):
        pc = 0 if index[0]['label'] == 'None' else len(index)
        text = index_name.capitalize()+': idx - '+ str(round(np.sum(index['value']),ndigits=3)) +', pc - '+str(pc)+'\n'
        if pc > 0:
            text += '('
            for i, (label, value) in enumerate(index):
                if i > 0:
                    text += ', '
                text += label + '(' + str(round(value, ndigits=3)) + ')'
            text += ')\n'
        return text
    def get_index_info(self,orientation_degrees=None, thr_pc=None):
        if orientation_degrees is None:
            orientation_degrees = self.network_data.default_degrees_orientation_idx
        if thr_pc is None:
            thr_pc = self.network_data.default_thr_pc
        indexes = self.network_data.get_all_index_of_neuron(layer=self.layer_data, neuron_idx=self.neuron_idx,
                                                           orientation_degrees=orientation_degrees,
                                                           thr_pc=thr_pc)
        return indexes

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
        plot_canvas.get_tk_widget().configure(width=IMAGE_SMALL_DEFAULT_SIZE[1]*2, height=450)
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
        charts_to_show = ADVANCED_CHARTS
        if self.layer_to_evaluate == self.network_data.layers_data[0].layer_id:
            charts_to_show.remove('Relevant Neurons')
        combo = ttk.Combobox(master=master, values=charts_to_show, state='readonly',justify=CENTER,width=15)
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
                self.get_params_from_popup(layer_to_evaluate=self.layer_to_evaluate)
            if min is None or condition1 is None or order is None or max_neurons is None:
                hidden_annotations, figure = None, None
            else:
                figure, hidden_annotations = plot_similar_neurons(network_data=self.network_data,
                                                                    layer=self.layer_to_evaluate,
                                          neuron_idx=self.neuron_idx,min=min, max=max, condition1=condition1,
                                          condition2=condition2, order=order, max_neurons=max_neurons)
        elif selected.lower() == 'activation curve':
            figure = get_one_neuron_plot(network_data=self.network_data,layer=self.layer_to_evaluate,
                                     neuron_idx=self.neuron_idx, chart=selected)
            hidden_annotations = None
        elif selected.lower() == 'relevant neurons':
            min, condition1, max, condition2, order, max_neurons = \
                self.get_params_from_popup(layer_to_evaluate=self.layer_to_evaluate,index='relevance')

            if min is None or condition1 is None or order is None or max_neurons is None:
                hidden_annotations, figure = None, None
            else:
                ablatable_layers= self.network_data.get_ablatable_layers(actual_layer=self.layer_to_evaluate)

                layer_to_ablate = self.get_value_from_popup_combobox(values=ablatable_layers,
                                                                     text='Select objective layer',
                                                                     default = ablatable_layers[-1])

                figure, hidden_annotations = plot_relevant_neurons(network_data=self.network_data,
                                                                  layer=self.layer_to_evaluate,
                                                                  layer_to_ablate=layer_to_ablate,
                                                                  neuron_idx=self.neuron_idx, min=min, max=max,
                                                                  condition1=condition1,
                                                                  condition2=condition2, order=order,
                                                                  max_neurons=max_neurons)

        self.put_figure_plot(master=self.figure_frame,figure=figure, hidden_annotations=hidden_annotations)


    def get_params_from_popup(self, index='similarity', layer_to_evaluate='unknow'):
        if type(layer_to_evaluate) is list:
            layer_to_evaluate = layer_to_evaluate[0]
        popup_window = OneLayerPopupWindow(self.window, layer_to_evaluate=layer_to_evaluate, index=index)
        self.window.wait_window(popup_window.top)
        return popup_window.value1,popup_window.condition1,popup_window.value2,popup_window.condition2,\
               popup_window.order, popup_window.neurons_to_show

    def get_value_from_popup_combobox(self, values, text, default=None):
        popup_window = ComboboxPopupWindow(self.window, values=values, text=text, default = default)
        self.window.wait_window(popup_window.top)
        return popup_window.value

    def raise_neuron_window(self, layer, neuron_idx):
        NeuronWindow(master=self.window, network_data=self.network_data, layer_to_evaluate=layer, neuron_idx=neuron_idx)