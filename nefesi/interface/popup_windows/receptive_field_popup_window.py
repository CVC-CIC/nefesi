
try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk

from ...util.general_functions import get_image_masked
from ...util.segmentation.utils import maskrcnn_colorencode
from ...class_index import concept_selectivity_of_image, translate_concept_hist
from PIL import ImageTk,Image
import os
import numpy as np



class ReceptiveFieldPopupWindow(object):
    def __init__(self, master, image_cropped,x_len,y_len, image_name, layer_name,
                 neuron_idx,interface,x0,x1,y0,y1, actual_idx):
        self.window=Toplevel(master)
        self.network_data = interface.network_data
        self.image_name = image_name
        self.layer_name = layer_name
        self.neuron_idx = neuron_idx
        self.actual_size=interface.image_actual_size
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.interface = interface
        self.actual_idx = actual_idx
        self.layer_data = self.network_data.get_layer_by_name(self.layer_name)
        self.receptive_field = self.layer_data.receptive_field_map
        self.neuron = self.layer_data.neurons_data[self.neuron_idx]
        self.labels_printed = False
        self.color_list = None
        self.window.title('Receptive Field of '+str(x_len)+'X'+str(y_len)+' (Neuron '+str(neuron_idx)+' '+layer_name+').')
        self.complete_image_frame = Frame(master=self.window)
        Label(master=self.complete_image_frame, text="Original image").pack(side=TOP)
        self.cropped_image_frame = Frame(master=self.window)
        Label(master=self.cropped_image_frame, text="Receptive field").pack(side=TOP)
        self.put_image(master=self.complete_image_frame, img=None, isComplete=True)
        self.put_image(master=self.cropped_image_frame, img=image_cropped)
        self.receptive_camp_frame = Frame(master=self.window)
        self.set_receptive_camp_frame(master=self.receptive_camp_frame)
        self.set_threshold_frame(master=self.receptive_camp_frame)
        self.set_false_labels_frame(master=self.receptive_camp_frame)
        self.complete_image_frame.pack(side=LEFT)
        self.receptive_camp_frame.pack(side=RIGHT)
        self.cropped_image_frame.pack(side=RIGHT)
        self._on_checkbox_clicked()

    def put_image(self, master, img, isComplete = False):
        panel = Label(master=master, image=img)
        panel.image = img
        panel.pack(side=BOTTOM, fill=BOTH, expand=True)
        if isComplete:
            self.panel_complete_img = panel

    def set_receptive_camp_frame(self, master):
        self.method_value = DoubleVar(master=master, value=1)
        master = LabelFrame(master, text="Overlap", padx=2, pady=2)
        master.pack(side=TOP, fill=X, expand="yes")
        ttk.Radiobutton(master, text="Original", variable=self.method_value, value=0,
                        command=lambda: self._on_checkbox_clicked()).pack(side=TOP, anchor="w", padx=10)
        ttk.Radiobutton(master, text="Torralba", variable=self.method_value, value=1,
                        command=lambda: self._on_checkbox_clicked()).pack(side=TOP, anchor="w", padx=10)
        ttk.Radiobutton(master, text="Vedaldi", variable=self.method_value, value=2,
                        command=lambda: self._on_checkbox_clicked()).pack(side=TOP, anchor="w", padx=10)
        ttk.Radiobutton(master, text="Act Torr", variable=self.method_value, value=3,
                        command=lambda: self._on_checkbox_clicked()).pack(side=TOP, anchor="w", padx=10)
        ttk.Radiobutton(master, text="Act Ved", variable=self.method_value, value=4,
                        command=lambda: self._on_checkbox_clicked()).pack(side=TOP, anchor="w", padx=10)
        ttk.Radiobutton(master, text="Segmentation", variable=self.method_value, value=5,
                        command=lambda: self._on_checkbox_clicked()).pack(side=TOP, anchor="w", padx=10)
        # checkbox_value = BooleanVar(master=master, value=True)
        # ttk.Checkbutton(master=master, text="Receptive camp",
        #                            variable=checkbox_value,
        #                            command=lambda: self._on_checkbox_clicked(checkbox_value)).pack(side=RIGHT)

    def _on_checkbox_clicked(self):
        value = self.method_value.get()
        if 0<value<5:
            img = get_image_masked(network_data=self.network_data, image_name=self.image_name,
                                   layer_name=self.layer_name, neuron_idx=self.neuron_idx,
                                   type=self.method_value.get(), thr_mth=self.thr_mth.get(), thr=self.thr.get()/100)
        else:
            img = self.network_data.dataset._load_image(self.image_name)
            if value == 5:
                if self.network_data.dataset.src_segmentation_dataset is not None:
                    image_path = os.path.join(self.network_data.dataset.src_segmentation_dataset, self.image_name)+'.npz'
                    segmentation = np.load(image_path)['object']
                else:
                    from ...util.segmentation.Broden_analize import Segment_images
                    image_path = os.path.join(self.network_data.dataset.src_dataset, self.image_name)
                    segmentation = Segment_images([image_path])[0]['object']
                if self.color_list is None:
                    self.color_list = np.random.rand(1000, 3) * .7 + .3
                img = maskrcnn_colorencode(np.asarray(img), segmentation, self.color_list)
                if not self.labels_printed:
                    ri, rf, ci, cf = self.receptive_field[
                        self.neuron.xy_locations[self.actual_idx, 0], self.neuron.xy_locations[self.actual_idx, 1]]
                    ri, rf, ci, cf = abs(ri),abs(rf),abs(ci),abs(cf)
                    segmentation = segmentation[ri:rf, ci:cf]
                    self.set_labels_frame(master=self.receptive_camp_frame, segment=segmentation, color_list=self.color_list)
                    self.labels_printed = True

        img = self.interface.draw_rectangle_on_image(np.array(img), self.x0, self.x1, self.y0, self.y1)
        img = Image.fromarray(img.astype('uint8'))
        img = img.resize(self.actual_size, Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.panel_complete_img.configure(image=img)
        self.panel_complete_img.image = img

    def set_threshold_frame(self, master):
        master = LabelFrame(master, text="Threshold", padx=2, pady=2)
        master.pack(side=TOP, fill=X, expand="yes")
        self.thr_mth = DoubleVar(master=master, value=1.)
        ttk.Radiobutton(master, text="Max over Top", variable=self.thr_mth, value=0,
                        command=lambda: self._on_checkbox_clicked()).pack(side=TOP, anchor="w")
        ttk.Radiobutton(master, text="Max over Image", variable=self.thr_mth, value=1,
                        command=lambda: self._on_checkbox_clicked()).pack(side=TOP, anchor="w")
        #This needs to be changed for use decimal numbers, but sames to have a bug with resolutions decimals
        #Temporary is using a % scale
        self.thr = Scale(master, from_=0, to=100, orient=HORIZONTAL, digits=3, resolution=1)
        self.thr.set(5)
        self.thr.pack(side=TOP)

    def set_labels_frame(self, master, segment, color_list):
        ids, freqs = concept_selectivity_of_image(activations_mask=None, segmented_image=segment, type='percent')
        arr = [(id, freq, tuple(color_list[id])) for id, freq in zip(ids, freqs)]
        arr = np.array(arr, dtype=[('label', np.int), ('value', np.float), ('color', np.object)])
        arr_transl = translate_concept_hist(arr, 'object')
        arr_comp = np.zeros(len(arr), dtype=[('label', np.object), ('value', np.float), ('color', np.object)])
        arr_comp['label'], arr_comp['value'], arr_comp['color'] = arr_transl['label'], arr_transl['value'], arr['color']
        arr_comp = np.sort(arr_comp, order = 'value')[::-1]
        for label, freq, color in arr_comp:
            text = label + '('+str(np.round(freq*100,2))+'%)'
            ttk.Label(self.labels_frame, text=text, background=rgb_to_code(color)).pack(side=TOP, anchor="w")

    def set_false_labels_frame(self, master,):
        self.labels_frame = LabelFrame(master, text="Segments (on rec. field)", padx=2, pady=2)
        self.labels_frame.pack(side=TOP, fill=X, expand="yes")
def rgb_to_code(rgb):
    if type(rgb[0]) is not int:
        rgb = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    return "#%02x%02x%02x" % rgb