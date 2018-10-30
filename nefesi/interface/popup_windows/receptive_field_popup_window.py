
try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk

from ...util.general_functions import get_image_masked
from PIL import ImageTk,Image
import numpy as np



class ReceptiveFieldPopupWindow(object):
    def __init__(self, master, image_cropped,x_len,y_len, image_name, layer_name,
                 neuron_idx,interface,x0,x1,y0,y1):
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
        self.method_value = IntVar(master=master, value=2)
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
        # checkbox_value = BooleanVar(master=master, value=True)
        # ttk.Checkbutton(master=master, text="Receptive camp",
        #                            variable=checkbox_value,
        #                            command=lambda: self._on_checkbox_clicked(checkbox_value)).pack(side=RIGHT)

    def _on_checkbox_clicked(self):
        if self.method_value.get():
            img = get_image_masked(network_data=self.network_data, image_name=self.image_name,
                                   layer_name=self.layer_name, neuron_idx=self.neuron_idx,
                                   type=self.method_value.get(), thr_mth=self.thr_mth.get(), thr=self.thr.get())
        else:
            img = self.network_data.dataset._load_image(self.image_name)
        img = self.interface.draw_rectangle_on_image(np.array(img), self.x0, self.x1, self.y0, self.y1)
        img = Image.fromarray(img.astype('uint8'))
        img = img.resize(self.actual_size, Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.panel_complete_img.configure(image=img)
        self.panel_complete_img.image = img

    def set_threshold_frame(self, master):
        master = LabelFrame(master, text="Threshold", padx=2, pady=2)
        master.pack(side=TOP, fill=X, expand="yes")
        self.thr_mth = IntVar(master=master, value=1)
        ttk.Radiobutton(master, text="Max over Top", variable=self.thr_mth, value=0,
                        command=lambda: self._on_checkbox_clicked()).pack(side=TOP, anchor="w")
        ttk.Radiobutton(master, text="Max over Image", variable=self.thr_mth, value=1,
                        command=lambda: self._on_checkbox_clicked()).pack(side=TOP, anchor="w")
        self.thr = Scale(master, from_=0, to=1, orient=HORIZONTAL, digits=4, resolution=0.001)
        self.thr.set(0.005)
        self.thr.pack(side=TOP)