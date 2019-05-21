
MAX_CONCEPTS_TO_SHOW = 4

IMAGE_BIG_DEFAULT_SIZE = (800,800)
IMAGE_SMALL_DEFAULT_SIZE = (450,450)
ADVANCED_CHARTS = ['Activation Curve', 'Similar Neurons', 'Relevant Neurons']
TREE_THRESHOLD = 50
#That have with images of A are the column of A

from PIL import ImageDraw
from ...util.general_functions import ordinal

try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk
import numpy as np

from PIL import ImageTk, Image
from ..EventController import EventController

class LinkWindow(object):
    def __init__(self, master, network_data, original_layer, ablated_layer, neuron_idx, image_actual_size=IMAGE_SMALL_DEFAULT_SIZE):
        self.event_controller = EventController(self)
        self.network_data = network_data
        self.original_layer = original_layer
        self.ablated_layer = ablated_layer
        self.neuron_idx = neuron_idx
        self.master = master
        if self.network_data.model is not None:
            network_name = self.network_data.model.name.capitalize()
        else:
            file_name = self.network_data.save_path
            network_name = self.network_data.save_path[file_name.rfind('/')+1:file_name.rfind('.')].capitalize()

        self.window = Toplevel(master=master)
        self.window.title('Network: '+network_name+' - Layer: '+original_layer + ' - Neuron: '+str(neuron_idx)+
                          '. Links on layer: '+ablated_layer)
        self.image_actual_size = image_actual_size
        #self.font = ImageFont.truetype(font='sans-serif.ttf', size=16)
        self.original_neuron = self.network_data.get_neuron_of_layer(layer=original_layer, neuron_idx=neuron_idx)
        self.actual_link = 0
        self.relevance = self.original_neuron.relevance_idx[self.ablated_layer]
        self.relevance_order = np.argsort(self.relevance)[::-1]
        self.basic_frame = ttk.Frame(master=self.window)
        self.decomposition_frame = ttk.Frame(self.basic_frame)
        self.set_decomposition_frame(self.decomposition_frame)
        self.decomposition_frame.pack(side=RIGHT)
        self.basic_frame.pack(side=TOP)


    def set_decomposition_frame(self, master):
        image_frame = ttk.Frame(master=master)
        neuron_label, relevance_label = self.set_decomposition_label(master)
        neuron_of_link_idx = self.relevance_order[self.actual_link]
        neuron_of_link = self.network_data.get_neuron_of_layer(layer=self.ablated_layer, neuron_idx=neuron_of_link_idx)
        current_image = neuron_of_link._neuron_feature
        current_image = current_image.resize((int(self.image_actual_size[0]/2),int(self.image_actual_size[1]/2)), Image.ANTIALIAS)  # resize mantaining aspect ratio
        self.write_decreasing_on_image(img = current_image)
        img = ImageTk.PhotoImage(current_image)

        self.panel_image = ttk.Label(master=image_frame, image=img)
        self.panel_image.image = img
        self.panel_image.bind("<Double-Button-1>",lambda event: self._on_link_image_click(event,
                                                                self.master, self.network_data,self.ablated_layer,
                                                                neuron_of_link_idx))
        decrease_button = ttk.Button(master=image_frame, text='<', width=1, command=lambda: self.event_controller._on_link_decrease_click(self.panel_image,
                                        neuron_label, relevance_label))
        increase_button = ttk.Button(master=image_frame, text='>', width=1, command=lambda: self.event_controller._on_link_increase_click(self.panel_image,
                                        neuron_label, relevance_label))
        increase_button.pack(side=RIGHT, fill='y')
        decrease_button.pack(side=LEFT,fill='y')
        self.panel_image.pack(side=TOP)
        image_frame.pack(side=TOP)


    def _on_link_image_click(self, event,master, network_data, layer_name, neuron_idx):
        from .neuron_window import NeuronWindow
        NeuronWindow(master=master,network_data=network_data,layer_to_evaluate=layer_name, neuron_idx=neuron_idx)


    def set_decomposition_label(self, master):
        ord = ordinal(self.actual_link+1)
        text_frame = ttk.Frame(master=master)
        neuron_of_link = self.relevance_order[self.actual_link]
        relevance = str(round(self.relevance[neuron_of_link],ndigits=3))
        ttk.Label(master=text_frame, text="Neuron: ", font='Helvetica 10').pack(side=LEFT)
        self.neuron_label = ttk.Label(master=text_frame, text=str(neuron_of_link)+' ('+ord+')', font='Helvetica 10 bold')
        self.neuron_label.pack(side=LEFT)
        ttk.Label(master=text_frame, text="Relevance:", font='Helvetica 10').pack(side=LEFT)
        self.relevance_label = ttk.Label(master=text_frame, text=relevance,
                                         font='Helvetica 10 bold')
        self.relevance_label.pack(side=LEFT)
        text_frame.pack(side=TOP)
        return self.neuron_label, self.relevance_label


    def update_decomposition_label(self, neuron_label, relevance_label):
        neuron_idx = self.relevance_order[self.actual_link]
        neuron_label.configure(text=str(neuron_idx)+' ('+ordinal(self.actual_link+1)+')')
        relevance_label.configure(text=str(round(self.relevance[neuron_idx], ndigits=3)))

    def update_decomposition_panel(self, panel):
        neuron = self.network_data.get_neuron_of_layer(layer=self.ablated_layer,
                                              neuron_idx=self.relevance_order[self.actual_link])
        new_image = neuron._neuron_feature
        new_image = new_image.resize((int(self.image_actual_size[0]/2),int(self.image_actual_size[1]/2)),Image.ANTIALIAS)
        self.write_decreasing_on_image(img=new_image)
        img = ImageTk.PhotoImage(new_image)
        panel.configure(image=img)
        panel.image = img

    def write_decreasing_on_image(self, img):
        neuron_idx = self.relevance_order[self.actual_link]
        a = self.original_neuron.get_relevance_idx(network_data=self.network_data, layer_name=self.original_layer,
                                               neuron_idx = self.neuron_idx, layer_to_ablate=self.ablated_layer,
                                               for_neuron=neuron_idx, return_decreasing=True)
        concept =  ('Not Ready', 0.0)#self.original_neuron.most_relevant_concept[self.ablated_layer][neuron_idx]
        type = ('Not Ready', 0.0)#self.original_neuron.most_relevant_type[self.ablated_layer][neuron_idx]
        text_image = ImageDraw.Draw(img)
        text_image.text((10, 0), "Decreased Type: "+type[0]+ ' -> '+str(type[1]), fill=0)
        text_image.text((10, 15), "Decreased Concept: " + concept[0] + ' -> ' + str(concept[1]), fill=0)

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