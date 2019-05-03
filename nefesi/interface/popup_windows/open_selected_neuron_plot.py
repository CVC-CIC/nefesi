import tkinter as tk# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3

try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk


from ..interface import MAX_VALUES_VISIBLES_IN_LISTBOX
import numpy as np
RED_LIGHTED_COLOR = '#ffcccc'

class OpenSelectedNeuronPlot(object):
    def __init__(self, master, network_data):
        self.layer = None #Returned value if user clicks on X
        self.network_data = network_data
        self.current_layer = None
        self.neuron_selected = None
        self.neuron = None
        self.top=Toplevel(master)
        self.top.title('Select layer and Neuron')
        lstbox_layer_frame = ttk.Frame(master=self.top)
        lstbox_neuron_frame = ttk.Frame(master=self.top)
        entry_neuron_frame = ttk.Frame(master=self.top)
        self.set_layers_listbox(master=lstbox_layer_frame)
        self.set_neurons_lstbox(master=lstbox_neuron_frame)
        self.ok_button = ttk.Button(entry_neuron_frame, text='Ok', command=self.cleanup)
        self.ok_button['state'] = 'disabled'
        validate_command = (master.register(self._on_entry_updated_check_validity),
                            '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        self.actual_info_label = ttk.Label(master = entry_neuron_frame, text = "No layer selected: ")
        entry_and_label_frame = ttk.Frame(master=entry_neuron_frame)
        neurons_to_show_label = ttk.Label(master=entry_and_label_frame, text = "Neuron: ")
        self.neurons_to_show_entry = ttk.Entry(master=entry_and_label_frame, validate='key', validatecommand=validate_command,
                                           textvariable=StringVar(master=self.top), justify=CENTER, width=7)
        self.neurons_to_show_entry['state'] = 'disabled'
        entry_neuron_frame.pack(side=RIGHT)
        lstbox_neuron_frame.pack(side=RIGHT)
        lstbox_layer_frame.pack(side=LEFT)
        self.actual_info_label.pack(side=TOP)
        entry_and_label_frame.pack(side=TOP)
        neurons_to_show_label.pack(side=LEFT)
        self.neurons_to_show_entry.pack(side=RIGHT)
        self.ok_button.pack(side=BOTTOM)

    def cleanup(self):
        self.layer = self.current_layer
        self.neuron = self.neuron_selected
        self.top.destroy()

    def entry_update(self, new_value):
        self.neurons_to_show_entry.delete(0,END)
        for i in reversed(range(len(new_value))):
            self.neurons_to_show_entry.insert(0,new_value[i])

    def set_layers_listbox(self, master):
        # Title just in top of selector
        lstbox_tittle = ttk.Label(master=master, text="Select Layer")
        list_values = [layer.layer_id for layer in self.network_data.layers_data]
        scrollbar = ttk.Scrollbar(master=master, orient="vertical")
        lstbox = Listbox(master=master, selectmode=SINGLE, yscrollcommand=scrollbar.set,
                         height=min(len(list_values)+2,MAX_VALUES_VISIBLES_IN_LISTBOX))
        scrollbar.config(command=lstbox.yview)
        for item in list_values:
            lstbox.insert(END, item)
        lstbox_tittle.pack(side=TOP)
        lstbox.pack(side=LEFT)
        scrollbar.pack(side=RIGHT, fill="y")
        lstbox.bind('<<ListboxSelect>>', lambda event: self._on_change_layer_selection(event, lstbox, list_values))
        return lstbox

    def set_neurons_lstbox(self,master):
        title_frame = ttk.Frame(master=master)
        combo_title = ttk.Label(title_frame, text="Order by:")
        combo_title.pack(side=LEFT, expand=False)
        # Options will be 1,2,3,4
        self.order_combo = ttk.Combobox(master=title_frame, values=self.network_data.indexs_accepted,
                                        state='readonly', width=15, justify=CENTER)
        self.order_combo.set('Select Index')
        # When selection is changed, calls the function _on_number_of_plots_to_show_changed
        self.order_combo.bind("<<ComboboxSelected>>", lambda event: self._on_order_by_selection_changed(event, self.order_combo))
        self.order_combo.pack(side=LEFT, expand=False)
        title_frame.pack(side=TOP)
        lstbox_frame = ttk.Frame(master=master)
        lstbox_frame.pack(side=BOTTOM)
        scrollbar = ttk.Scrollbar(master=lstbox_frame, orient="vertical")
        self.neuron_lstbox = Listbox(master=lstbox_frame, selectmode=SINGLE, yscrollcommand=scrollbar.set,
                         height=MAX_VALUES_VISIBLES_IN_LISTBOX, width=25)
        scrollbar.config(command=self.neuron_lstbox.yview)
        self.neuron_lstbox.bind('<<ListboxSelect>>', self._on_change_neuron_lstbox)
        self.neuron_lstbox.pack(side=LEFT)
        scrollbar.pack(side=RIGHT, fill="y")

    def _on_change_neuron_lstbox(self,event):
        selection = self.neuron_lstbox.curselection()
        if selection is not ():
            selection = selection[0]
            selection = self.neuron_lstbox.get(selection)
            self.neuron_selected = selection[len('Neuron: '):selection.index('.')]
            self.entry_update(self.neuron_selected)
            self.ok_button['state'] = 'normal'

    def _on_change_layer_selection(self, event, lstbox, lstbox_values):
        selection = lstbox.curselection()
        if selection is not ():
            selection = lstbox_values[selection[0]]
            if selection != self.current_layer:
                self.current_layer = selection
                if self.neurons_to_show_entry['state'] != 'normal':
                    self.neurons_to_show_entry['state'] = 'normal'
                self.update_info_label()
                self.entry_update(self.neurons_to_show_entry.get())
                if self.order_combo.get() != 'Select Index':
                    self.update_neurons_lstbox(layer_selected=selection)

    def update_info_label(self):
        text = self.current_layer+' ('+\
               str(self.network_data.get_len_neurons_of_layer(self.current_layer))+' Neur.)'
        self.actual_info_label.configure(text=text)
    def _on_order_by_selection_changed(self,event, combo):
        if self.current_layer is not None:
            self.update_neurons_lstbox(self.current_layer)

    def update_neurons_lstbox(self, layer_selected):
        selection = self.order_combo.get().lower()
        sel_idx = self.network_data.get_selectivity_idx(sel_index=selection,layer_name=layer_selected)[selection][0]
        if selection in ['symmetry', 'orientation']:
            sel_idx = sel_idx[:,-1]
        elif selection == 'class':
            sel_idx = sel_idx['value']
        elif selection == 'concept':
            sel_idx = np.array([neuron_concept[0]['count'][0] for neuron_concept in sel_idx])
        self.neuron_lstbox.delete(0,END)
        args_sorted = np.argsort(sel_idx)
        sel_idx = sel_idx[args_sorted]
        for i in reversed(range(len(sel_idx))):
            text = 'Neuron: '+str(args_sorted[i])+'. Idx: '+str(round(sel_idx[i],2))
            self.neuron_lstbox.insert(END, text)





    def _on_entry_updated_check_validity(self, action, index, value_if_allowed,
                                          prior_value, text, validation_type, trigger_type, widget_name):
        is_valid = False
        neurons_of_actual_layer = self.network_data.get_len_neurons_of_layer(layer=self.current_layer)
        # action=1 -> insert
        if (action == '1'):
            if text in '0123456789':
                try:
                    # if new value is valid float
                    value = int(value_if_allowed)
                    if value < 0 or value >= neurons_of_actual_layer:
                        self.ok_button['state'] = 'disabled'
                        if self.neurons_to_show_entry is not None:
                            self.neurons_to_show_entry.config({"background":RED_LIGHTED_COLOR})
                    else:
                        # if is not in correct range
                        self.ok_button['state'] = 'normal'
                        if self.neurons_to_show_entry is not None:
                            self.neurons_to_show_entry.config({"background": 'white'})
                        self.neuron_selected = value
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
                        if value < 0 or value >= neurons_of_actual_layer:
                            self.ok_button['state'] = 'disabled'
                            if self.neurons_to_show_entry is not None:
                                self.neurons_to_show_entry.config({"background": RED_LIGHTED_COLOR})
                        else:
                            self.ok_button['state'] = 'normal'
                            if self.neurons_to_show_entry is not None:
                                self.neurons_to_show_entry.config({"background": 'white'})
                            self.neuron_selected = value
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
                    if value < 0 or value >= neurons_of_actual_layer:
                        self.ok_button['state'] = 'disabled'
                        if self.neurons_to_show_entry is not None:
                            self.neurons_to_show_entry.config({"background": RED_LIGHTED_COLOR})
                    else:
                        self.ok_button['state'] = 'normal'
                        if self.neurons_to_show_entry is not None:
                            self.neurons_to_show_entry.config({"background": 'white'})
                        self.neuron_selected = value
                except:
                    pass
            return True
        else:
            self.ok_button['state'] = 'normal'
            if self.neurons_to_show_entry is not None:
                self.neurons_to_show_entry.config({"background": 'white'})
            try:
                self.neuron_selected = int(value_if_allowed)
            except:
                pass
            return True
