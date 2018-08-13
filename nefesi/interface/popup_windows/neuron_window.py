import tkinter as tk# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3

from skimage.draw import line_aa

try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk
import numpy as np
from nefesi.util.general_functions import clean_widget
from nefesi.util.interface_plotting import ORDER
from PIL import ImageTk, Image
from nefesi.interface.popup_windows.receptive_field_popup_window import ReceptiveFieldPopupWindow

IMAGE_DEFAULT_SIZE = (250,250)
class NeuronWindow(object):
    def __init__(self, master, network_data, layer_to_evaluate, neuron_idx):
        self.network_data = network_data
        self.layer_to_evaluate = layer_to_evaluate
        self.neuron_idx = neuron_idx
        self.actual_img_index = 0
        self.neuron = self.network_data.get_neuron_of_layer(layer=layer_to_evaluate, neuron_idx=neuron_idx)
        self.window=Toplevel(master)
        self.window.title(str(layer_to_evaluate) + ' Neuron: '+str(neuron_idx))
        self.index_info = Frame(master=self.window)
        self.set_index_info(master=self.index_info)
        self.images_frame = Frame(master=self.window)
        self.neuron_feature_frame = Frame(master=self.images_frame)
        self.decomposition_frame = Frame(master=self.images_frame)

        self.set_neuron_feature_frame(master=self.neuron_feature_frame)
        self.neuron_feature_frame.pack(side=LEFT, fill=BOTH, expand=True)
        self.decomposition_frame = Frame(self.window)
        self.set_decomposition_frame(self.decomposition_frame)
        self.images_frame.pack(side=LEFT,padx=5)
        self.neuron_feature_frame.pack(side=LEFT,padx=5)
        self.index_info.pack(side=RIGHT)
        self.decomposition_frame.pack(side=RIGHT)

        self.neuron.get_patch_by_idx(self.network_data, self.network_data.get_layer_by_name(layer_to_evaluate), 0)

    def set_decomposition_frame(self, master):
        label = Label(master=master, text="Decomposition")
        label.pack(side=TOP)
        neuron_feature = self.neuron.get_patch_by_idx(self.network_data,
                                                      self.network_data.get_layer_by_name(self.layer_to_evaluate),
                                                      self.actual_img_index)

        neuron_feature = neuron_feature.resize(IMAGE_DEFAULT_SIZE, Image.ANTIALIAS)  # resize mantaining aspect ratio
        img = ImageTk.PhotoImage(neuron_feature)
        panel = Label(master=master, image=img)
        panel.image = img
        panel.bind("<Double-Button-1>", lambda event:self._on_image_click(event,panel.image))
        decrease_button = Button(master=master, text='<-', command=lambda: self._on_decrease_click(panel))
        increase_button = Button(master=master, text='->', command=lambda: self._on_increase_click(panel))
        increase_button.pack(side=RIGHT, fill='y')
        decrease_button.pack(side=LEFT,fill='y')
        panel.pack(side=RIGHT)

    def _on_decrease_click(self,panel):
        self.actual_img_index = (self.actual_img_index-1)%len(self.neuron.images_id)
        self.update_decomposition_panel(panel=panel)

    def _on_increase_click(self,panel):
        self.actual_img_index = (self.actual_img_index+1)%len(self.neuron.images_id)
        self.update_decomposition_panel(panel=panel)

    def update_decomposition_panel(self, panel):
        new_image = self.neuron.get_patch_by_idx(self.network_data,
                                                      self.network_data.get_layer_by_name(self.layer_to_evaluate),
                                                      self.actual_img_index)
        new_image = new_image.resize(IMAGE_DEFAULT_SIZE,Image.ANTIALIAS)
        img = ImageTk.PhotoImage(new_image)
        panel.configure(image=img)
        panel.image = img
    def _on_image_click(self,event, cropped_image):
        actual_idx = self.actual_img_index
        complete_image = self.network_data.dataset._load_image(self.neuron.images_id[actual_idx])
        np_image = np.array(complete_image)
        layer_data = self.network_data.get_layer_by_name(layer=self.layer_to_evaluate)
        receptive_field = layer_data.receptive_field_map
        x0,x1,y0,y1 = receptive_field[self.neuron.xy_locations[self.actual_img_index, 0],
                                         self.neuron.xy_locations[actual_idx, 1]]
        x_len,y_len = x1-x0, y1-y0

        np_image = self.draw_rectangle_on_image(np_image,x0,x1,y0,y1)
        complete_image = Image.fromarray(np_image.astype('uint8'), 'RGB')
        complete_image = complete_image.resize(IMAGE_DEFAULT_SIZE, Image.ANTIALIAS)
        complete_image = ImageTk.PhotoImage(complete_image)

        cropped_image = self.neuron.get_patch_by_idx(self.network_data,
                                                      self.network_data.get_layer_by_name(self.layer_to_evaluate),
                                                      self.actual_img_index)

        cropped_image = cropped_image.resize(IMAGE_DEFAULT_SIZE, Image.ANTIALIAS)  # resize mantaining aspect ratio
        np_cropped = np.array(cropped_image)
        np_cropped = self.draw_rectangle_on_image(np_cropped, 0, np_cropped.shape[0], 0, np_cropped.shape[1],margin=2,
                                                  draw_lines=False)
        cropped_image = Image.fromarray(np_cropped.astype('uint8'), 'RGB')
        cropped_image = ImageTk.PhotoImage(cropped_image)
        ReceptiveFieldPopupWindow(master=self.window, image_complete=complete_image, image_cropped=cropped_image,
                                  x_len=x_len, y_len=y_len)


    def draw_rectangle_on_image(self, np_image, x0, x1,y0,y1,margin=3, draw_lines=True):
        red = [255,0,0]
        x0 = max(x0-1,margin)
        x1 = min(x1 + 1, np_image.shape[1]-1-margin)
        y0 = max(y0 - 1, margin)
        y1 = min(y1 + 1, np_image.shape[0]-1-margin)
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
        label = Label(master=master, text="Neuron Mean Feature")
        label.pack(side=TOP)
        neuron_feature = self.neuron._neuron_feature
        neuron_feature = neuron_feature.resize(IMAGE_DEFAULT_SIZE,Image.ANTIALIAS) #resize mantaining aspect ratio
        img = ImageTk.PhotoImage(neuron_feature)
        panel = Label(master=master, image=img)
        panel.image = img
        panel.pack(side=BOTTOM, fill=BOTH, expand=True)
        #self.window.bind('<Configure>', lambda event: self._on_resize_image(event, panel, neuron_feature, label))

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

    def get_index_info(self,orientation_degrees=90, thr_class_idx=1., thr_pc=0.1):
        indexs = self.network_data.get_all_index_of_neuron(layer=self.layer_to_evaluate, neuron_idx=self.neuron_idx,
                                                           orientation_degrees=orientation_degrees,
                                                           thr_class_idx=thr_class_idx,thr_pc=thr_pc)
        return indexs








    def set_footers(self, master):
        frame = Frame(master=master)
        label = Label(master=frame, text=self.footer1,font=("Times New Roman", 8))
        label.grid(row=0,padx=(75,0))
        label = Label(master=frame, text="** Accepted range [" + str(NEURONS_TO_SHOW_RANGE[0]) + ", " + \
                                          str(NEURONS_TO_SHOW_RANGE[-1]) + "]", font=("Times New Roman", 8))
        label.grid(row=1,padx=(75,0))
        frame.pack(side=BOTTOM)

    def cleanup(self):
        if self.combo1 is not None:
            self.condition1=self.combo1.get()
            if self.entry1 is not None:
                self.value1 = float(self.entry1.get())
        if self.combo2 is not None:
            self.condition2=self.combo2.get()
            if self.entry2 is not None:
                self.value2 = float(self.entry2.get())
        self.order = self.order_combo.get()
        self.neurons_to_show = int(self.neurons_to_show_entry.get())
        self.top.destroy()

    def set_max_neurons_to_show_entry(self, master):
        label = Label(master=master, text=NEURONS_TO_SHOW_TEXT)
        validate_command = (master.register(self._on_entry_updated_check_max_neurons_validity),
                             '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        self.neurons_to_show_entry = Entry(master, validate='key', validatecommand=validate_command,
                           textvariable=StringVar(master=self.top, value=15),justify=CENTER,width = 5)
        label.grid(column=0,row=0)
        self.neurons_to_show_entry.grid(column=1,row=0)
        label = Label(master=master, text="**")
        label.grid(column=2,row=0)

    def set_order_selector(self, master):
        label = Label(master=master, text=ORDER_FRAME_TEXT)
        self.order_combo = ttk.Combobox(master=master, values=ORDER, state='readonly', width=8, justify=CENTER)
        self.order_combo.set(ORDER[0])
        self.order_combo.bind("<<ComboboxSelected>>", self._on_order_or_condition_selector_changed)
        label.pack(side=LEFT,padx=(2,25))
        self.order_combo.pack(side=RIGHT)


    def set_conditions_selector(self, master, default1='>=', default2=None):
        assert((default1 in CONDITIONS) or (default2 in CONDITIONS))
        values_frame = Frame(master=master)
        if default1 == None or default2 == None:
            if default1 == None:
                default1 = default2
                if default1 == None:
                    raise ValueError("Two default values of type selector can't be None")
            label_column, combo1_column, entry1_column = 0,1,2
            if self.combo2 is not None:
                self.combo2.destroy()
                self.combo2 = None
            if self.entry2 is not None:
                self.entry2.destroy()
                self.entry2 = None
            #Only one
        else:
            #Is range
            self.combo2 = ttk.Combobox(master=values_frame, values=CONDITIONS, width=3, state='readonly',justify=CENTER)
            self.combo2.set(default2)
            self.entry2 = Entry(values_frame, validate='key', validatecommand=self.validate_command_entry_2,
                           textvariable=StringVar(master=self.top, value=self.entry2_default),justify=CENTER,width = 5)
            self.combo2.grid(column=3,row=0,padx=2)
            self.entry2.grid(column=4,row=0,padx=2)
            label_column, combo1_column, entry1_column = 2, 1, 0

        label = Label(master=values_frame, text = self.range_label_text)
        if self.combo1 is not None:
            self.combo1.destroy()
            self.combo1 = None
        self.combo1 = ttk.Combobox(master=values_frame, values=CONDITIONS, width=3, state='readonly',justify=CENTER)
        self.combo1.set(default1)
        self.combo1.bind("<<ComboboxSelected>>", self._on_order_or_condition_selector_changed)
        if self.entry1 is not None:
            self.entry1.destroy()
            self.entry1 = None
        self.entry1 = Entry(values_frame, validate='key', validatecommand=self.validate_command_entry_1,
                            textvariable=StringVar(master=self.top, value=self.entry1_default),justify=CENTER, width = 5)
        label.grid(column=label_column,row=0,padx=2)
        self.combo1.grid(column=combo1_column,row=0,padx=2)
        self.entry1.grid(column=entry1_column,row=0,padx=2)

        label_explication= Label(master=master, text=CONDITIONS_TEXT)
        label_explication.pack(side=TOP,padx=(0,75))
        values_frame.pack(side=BOTTOM,padx=5)

    def set_selection_type_selector(self, master):
        label = Label(master=master,text="Select constraints: ")
        self.type_combo = ttk.Combobox(master=master, values=SELECTOR_OPTIONS, state='readonly',width=9, justify=CENTER)
        self.type_combo.bind("<<ComboboxSelected>>", self._on_type_selector_changed)
        self.type_combo.set(SELECTOR_OPTIONS[0])
        label.pack(side=LEFT)
        self.type_combo.pack(side=RIGHT)

    def _on_type_selector_changed(self,event):
        """
        event called when user selects another chart to show in the combobox of the plot canvas (in general state (init)
        :param event: event with the widget of the combobox changed
        """
        selector = event.widget
        selected = selector.get()
        if selected != self._type_selector_last_value:
            #if comes from not range to range or from range to non range
            if selected == 'in range':
                clean_widget(self.conditions_frame)
                self.set_conditions_selector(master=self.conditions_frame, default1='<=', default2='<=')
            else:
                if self._type_selector_last_value == 'in range':
                    clean_widget(self.conditions_frame)
                    self.set_conditions_selector(master=self.conditions_frame, default1='>=')
                if selected == SELECTOR_OPTIONS[0]:  # highers
                    self.order_combo.set(ORDER[0]) #descend
                elif selected == SELECTOR_OPTIONS[1]: #lowers
                    self.order_combo.set(ORDER[1])
                self.combo1.set('>=')
                self.entry1.delete(0,END)
                self.entry1.insert(0,'0')
            self._type_selector_last_value = selected


    def _on_order_or_condition_selector_changed(self, event):
        if self.entry1 is not None:
            value = self.entry1.get()
            try:
                value = float(value)
                self._redefine_type_selector(value)
            except:
                pass

    def _on_entry_updated_check_range_1_0_entry_1(self, action, index, value_if_allowed,
                                                        prior_value, text, validation_type, trigger_type, widget_name):
        is_valid = False
        # action=1 -> insert
        if(action=='1'):
            if text in '+0123456789.':
                try:
                    #if new value is valid float
                    value = float(value_if_allowed)
                    if value<0.0 or value >1.0:
                        self.ok_button['state'] = 'disabled'
                        if self.entry1 is not None:
                            self.entry1.config({"background": RED_LIGHTED_COLOR})
                    else:
                        #if is not in correct range
                        self.ok_button['state'] = 'normal'
                        if self.entry1 is not None:
                            self.entry1.config({"background": 'white'})
                            self._redefine_type_selector(value)
                    return True
                except ValueError:
                    return False
            else:

                if len(value_if_allowed) == 1:
                    self.ok_button['state'] = 'disabled'
                    if self.entry1 is not None:
                        self.entry1.config({"background": RED_LIGHTED_COLOR})
                else:
                    try:
                        value = float(prior_value)
                        if value<0.0 or value >1.0:
                            self.ok_button['state'] = 'disabled'
                            if self.entry1 is not None:
                                self.entry1.config({"background": RED_LIGHTED_COLOR})
                        else:
                            self.ok_button['state'] = 'normal'
                            if self.entry1 is not None:
                                self.entry1.config({"background": 'white'})
                                self._redefine_type_selector(value)
                    except:
                        self.ok_button['state'] = 'normal'
                        if self.entry1 is not None:
                            self.entry1.config({"background": 'white'})
                return False
        #action 0 -> delete
        elif(action == '0'):
            #if will be empty
            if value_if_allowed =='':
                self.ok_button['state'] = 'disabled'
                if self.entry1 is not None:
                    self.entry1.config({"background": RED_LIGHTED_COLOR})
            else:

                try:
                    #if last value (the value that will remain) is a valid value in range
                    value = float(value_if_allowed)
                    if value > 1.0 or value < 0.0:
                        self.ok_button['state'] = 'disabled'
                        if self.entry1 is not None:
                            self.entry1.config({"background": RED_LIGHTED_COLOR})
                    else:
                        self.ok_button['state'] = 'normal'
                        if self.entry1 is not None:
                            self.entry1.config({"background": 'white'})
                except:
                    pass
            return True
        else:
            self.ok_button['state'] = 'normal'
            if self.entry1 is not None:
                self.entry1.config({"background": 'white'})
            return True

    def _redefine_type_selector(self, value):
        actual_combo = self.type_combo.get()
        if actual_combo != 'in range':
            if np.isclose(value, self.entry1_default) and self.combo1.get() == '>=':
                if self.order_combo.get() == ORDER[0] and actual_combo != SELECTOR_OPTIONS[0]:  # descend
                    self.type_combo.set(SELECTOR_OPTIONS[0])
                elif self.order_combo.get() == ORDER[1] and actual_combo != SELECTOR_OPTIONS[1]:
                    self.type_combo.set(SELECTOR_OPTIONS[1])
            elif np.isclose(value, self.entry2_default) and self.combo1.get() == '<=':
                if self.order_combo.get() == ORDER[0] and actual_combo != SELECTOR_OPTIONS[
                    0]:  # descend
                    self.type_combo.set(SELECTOR_OPTIONS[0])
                elif self.order_combo.get() == ORDER[1] and actual_combo != SELECTOR_OPTIONS[1]:
                    self.type_combo.set(SELECTOR_OPTIONS[1])
            else:
                if actual_combo != SELECTOR_OPTIONS[-1]:
                    self.type_combo.set(SELECTOR_OPTIONS[-1])

    def _on_entry_updated_check_range_1_0_entry_2(self, action, index, value_if_allowed,
                                                        prior_value, text, validation_type, trigger_type, widget_name):
        is_valid = False
        # action=1 -> insert
        if(action=='1'):
            if text in '+0123456789.':
                try:
                    #if new value is valid float
                    value = float(value_if_allowed)
                    if value<0.0 or value >1.0:
                        self.ok_button['state'] = 'disabled'
                        if self.entry2 is not None:
                            self.entry2.config({"background": RED_LIGHTED_COLOR})
                    else:
                        #if is not in correct range
                        self.ok_button['state'] = 'normal'
                        if self.entry2 is not None:
                            self.entry2.config({"background": 'white'})
                    return True
                except ValueError:
                    return False
            else:

                if len(value_if_allowed) == 1:
                    self.ok_button['state'] = 'disabled'
                    if self.entry2 is not None:
                        self.entry2.config({"background": RED_LIGHTED_COLOR})
                else:
                    try:
                        value = float(prior_value)
                        if value<0.0 or value >1.0:
                            self.ok_button['state'] = 'disabled'
                            if self.entry2 is not None:
                                self.entry2.config({"background": RED_LIGHTED_COLOR})
                        else:
                            self.ok_button['state'] = 'normal'
                            if self.entry2 is not None:
                                self.entry2.config({"background": 'white'})
                    except:
                        self.ok_button['state'] = 'normal'
                        if self.entry2 is not None:
                            self.entry2.config({"background": 'white'})
                return False
        #action 0 -> delete
        elif(action == '0'):
            #if will be empty
            if value_if_allowed =='':
                self.ok_button['state'] = 'disabled'
                if self.entry2 is not None:
                    self.entry2.config({"background": RED_LIGHTED_COLOR})
            else:

                try:
                    #if last value (the value that will remain) is a valid value in range
                    value = float(value_if_allowed)
                    if value > 1.0 or value < 0.0:
                        self.ok_button['state'] = 'disabled'
                        if self.entry2 is not None:
                            self.entry2.config({"background": RED_LIGHTED_COLOR})
                    else:
                        self.ok_button['state'] = 'normal'
                        if self.entry2 is not None:
                            self.entry2.config({"background": 'white'})
                except:
                    pass
            return True
        else:
            self.ok_button['state'] = 'normal'
            if self.entry2 is not None:
                self.entry2.config({"background": 'white'})
            return True

    def _on_entry_updated_check_non_negative_int_entry_2(self, action, index, value_if_allowed,
                                                        prior_value, text, validation_type, trigger_type, widget_name):
        is_valid = False
        # action=1 -> insert
        if(action=='1'):
            if text in '+0123456789':
                try:
                    #if new value is valid float
                    value = int(value_if_allowed)
                    if value<0:
                        self.ok_button['state'] = 'disabled'
                        if self.entry2 is not None:
                            self.entry2.config({"background": RED_LIGHTED_COLOR})
                    else:
                        #if is not in correct range
                        self.ok_button['state'] = 'normal'
                        if self.entry2 is not None:
                            self.entry2.config({"background": 'white'})
                    return True
                except ValueError:
                    return False
            else:

                if len(value_if_allowed) == 1:
                    self.ok_button['state'] = 'disabled'
                    if self.entry2 is not None:
                        self.entry2.config({"background": RED_LIGHTED_COLOR})
                else:
                    try:
                        value = int(prior_value)
                        if value<0:
                            self.ok_button['state'] = 'disabled'
                            if self.entry2 is not None:
                                self.entry2.config({"background": RED_LIGHTED_COLOR})
                        else:
                            self.ok_button['state'] = 'normal'
                            if self.entry2 is not None:
                                self.entry2.config({"background": 'white'})
                    except:
                        self.ok_button['state'] = 'normal'
                        if self.entry2 is not None:
                            self.entry2.config({"background": 'white'})
                return False
        #action 0 -> delete
        elif(action == '0'):
            #if will be empty
            if value_if_allowed =='':
                self.ok_button['state'] = 'disabled'
                if self.entry2 is not None:
                    self.entry2.config({"background": RED_LIGHTED_COLOR})
            else:

                try:
                    #if last value (the value that will remain) is a valid value in range
                    value = int(value_if_allowed)
                    if value<0:
                        self.ok_button['state'] = 'disabled'
                        if self.entry2 is not None:
                            self.entry2.config({"background": RED_LIGHTED_COLOR})
                    else:
                        self.ok_button['state'] = 'normal'
                        if self.entry2 is not None:
                            self.entry2.config({"background": 'white'})
                except:
                    pass
            return True
        else:
            self.ok_button['state'] = 'normal'
            if self.entry2 is not None:
                self.entry2.config({"background": 'white'})
            return True

    def _on_entry_updated_check_non_negative_int_entry_1(self, action, index, value_if_allowed,
                                                        prior_value, text, validation_type, trigger_type, widget_name):
        is_valid = False
        # action=1 -> insert
        if(action=='1'):
            if text in '+0123456789':
                try:
                    #if new value is valid float
                    value = int(value_if_allowed)
                    if value<0:
                        self.ok_button['state'] = 'disabled'
                        if self.entry1 is not None:
                            self.entry1.config({"background": RED_LIGHTED_COLOR})
                    else:
                        #if is not in correct range
                        self.ok_button['state'] = 'normal'
                        if self.entry1 is not None:
                            self.entry1.config({"background": 'white'})
                            self._redefine_type_selector(value)
                    return True
                except ValueError:
                    return False
            else:

                if len(value_if_allowed) == 1:
                    self.ok_button['state'] = 'disabled'
                    if self.entry1 is not None:
                        self.entry1.config({"background": RED_LIGHTED_COLOR})
                else:
                    try:
                        value = int(prior_value)
                        if value<0:
                            self.ok_button['state'] = 'disabled'
                            if self.entry1 is not None:
                                self.entry1.config({"background": RED_LIGHTED_COLOR})
                        else:
                            self.ok_button['state'] = 'normal'
                            if self.entry1 is not None:
                                self.entry1.config({"background": 'white'})
                                self._redefine_type_selector(value)
                    except:
                        self.ok_button['state'] = 'normal'
                        if self.entry1 is not None:
                            self.entry1.config({"background": 'white'})
                return False
        #action 0 -> delete
        elif(action == '0'):
            #if will be empty
            if value_if_allowed =='':
                self.ok_button['state'] = 'disabled'
                if self.entry1 is not None:
                    self.entry1.config({"background": RED_LIGHTED_COLOR})
            else:

                try:
                    #if last value (the value that will remain) is a valid value in range
                    value = int(value_if_allowed)
                    if value < 0:
                        self.ok_button['state'] = 'disabled'
                        if self.entry1 is not None:
                            self.entry1.config({"background": RED_LIGHTED_COLOR})
                    else:
                        self.ok_button['state'] = 'normal'
                        if self.entry1 is not None:
                            self.entry1.config({"background": 'white'})
                except:
                    pass
            return True
        else:
            self.ok_button['state'] = 'normal'
            if self.entry1 is not None:
                self.entry1.config({"background": 'white'})
            return True

    def _on_entry_updated_check_max_neurons_validity(self, action, index, value_if_allowed,
                                          prior_value, text, validation_type, trigger_type, widget_name):
        is_valid = False
        # action=1 -> insert
        if (action == '1'):
            if text in '0123456789':
                try:
                    # if new value is valid float
                    value = int(value_if_allowed)
                    if value < NEURONS_TO_SHOW_RANGE[0] or value > NEURONS_TO_SHOW_RANGE[-1]:
                        self.ok_button['state'] = 'disabled'
                        if self.neurons_to_show_entry is not None:
                            self.neurons_to_show_entry.config({"background":RED_LIGHTED_COLOR})
                    else:
                        # if is not in correct range
                        self.ok_button['state'] = 'normal'
                        if self.neurons_to_show_entry is not None:
                            self.neurons_to_show_entry.config({"background": 'white'})
                    return True
                except ValueError:
                    return False
            else:

                if len(value_if_allowed) == 1:
                    self.ok_button['state'] = 'disabled'
                    if self.neurons_to_show_entry is not None:
                        self.neurons_to_show_entry.config({"background": RED_LIGHTED_COLOR})
                else:
                    try:
                        value = int(prior_value)
                        if value < NEURONS_TO_SHOW_RANGE[0] or value > NEURONS_TO_SHOW_RANGE[-1]:
                            self.ok_button['state'] = 'disabled'
                            if self.neurons_to_show_entry is not None:
                                self.neurons_to_show_entry.config({"background": RED_LIGHTED_COLOR})
                        else:
                            self.ok_button['state'] = 'normal'
                            if self.neurons_to_show_entry is not None:
                                self.neurons_to_show_entry.config({"background": 'white'})
                    except:
                        self.ok_button['state'] = 'normal'
                        if self.neurons_to_show_entry is not None:
                            self.neurons_to_show_entry.config({"background": 'white'})
                return False
        # action 0 -> delete
        elif (action == '0'):
            # if will be empty
            if value_if_allowed == '':
                self.ok_button['state'] = 'disabled'
                if self.neurons_to_show_entry is not None:
                    self.neurons_to_show_entry.config({"background": RED_LIGHTED_COLOR})
            else:

                try:
                    # if last value (the value that will remain) is a valid value in range
                    value = int(value_if_allowed)
                    if value < NEURONS_TO_SHOW_RANGE[0] or value > NEURONS_TO_SHOW_RANGE[-1]:
                        self.ok_button['state'] = 'disabled'
                        if self.neurons_to_show_entry is not None:
                            self.neurons_to_show_entry.config({"background": RED_LIGHTED_COLOR})
                    else:
                        self.ok_button['state'] = 'normal'
                        if self.neurons_to_show_entry is not None:
                            self.neurons_to_show_entry.config({"background": 'white'})
                except:
                    pass
            return True
        else:
            self.ok_button['state'] = 'normal'
            if self.neurons_to_show_entry is not None:
                self.neurons_to_show_entry.config({"background": 'white'})
            return True
