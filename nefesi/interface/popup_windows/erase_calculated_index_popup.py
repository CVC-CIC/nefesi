import tkinter as tk# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3

from nefesi.interface.EventController import EventController
from nefesi.util.general_functions import get_listbox_selection, clean_widget

try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk



from ..interface import MAX_VALUES_VISIBLES_IN_LISTBOX
import numpy as np
RED_LIGHTED_COLOR = '#ffcccc'

class EraseCalculatedIndexPopup(object):
    def __init__(self, master, network_data):
        self.network_data = network_data
        self.lstbox_last_selection = (0,)
        self.top=Toplevel(master)
        self.top.title('Select index and layers for erase')
        self.ok_button = Button(master = self.top, text='Erase', command=self.cleanup)
        self.ok_button['state'] = 'disabled'
        self.layers_last_selection = (0,)
        indexes_frame = Frame(master=self.top)
        self.set_indexes_lstbox(master=indexes_frame)
        indexes_frame.pack(side=LEFT)
        lstbox_layer_frame = Frame(master=self.top)
        self.set_layers_listbox(master=lstbox_layer_frame)
        self.ok_button.pack(side=RIGHT)
        lstbox_layer_frame.pack(side=RIGHT)

    def cleanup(self):
        index_to_erase = self.lstbox_last_selection  # indexes
        layers = get_listbox_selection(self.layers_lstbox, selection=self.layers_last_selection)
        print('Erased index: '+index_to_erase+'. From: '+str(layers))
        self.network_data.erase_index_from_layers(layers=layers,index_to_erase=index_to_erase)
        self.update_indexes_lstbox()


    def set_indexes_lstbox(self, master):
        combo_title = ttk.Label(master, text="Indexes")
        combo_title.pack(side=TOP, expand=False)
        lstbox_frame = Frame(master=master)
        lstbox_frame.pack(side=BOTTOM)
        scrollbar = tk.Scrollbar(master=lstbox_frame, orient="vertical")
        self.indexes_lstbox = Listbox(master=lstbox_frame, selectmode=SINGLE, yscrollcommand=scrollbar.set,
                                      height=MAX_VALUES_VISIBLES_IN_LISTBOX, width=25)
        values = np.array(list(self.network_data.get_calculated_indexes_keys()))
        values = np.sort(values)
        for item in values:
            self.indexes_lstbox.insert(END, item)
        scrollbar.config(command=self.indexes_lstbox.yview)
        self.indexes_lstbox.bind('<<ListboxSelect>>', self._on_change_index_lstbox)
        self.indexes_lstbox.pack(side=LEFT)
        scrollbar.pack(side=RIGHT, fill="y")

    def update_indexes_lstbox(self):
        self.indexes_lstbox.delete(0, END)
        values = np.array(list(self.network_data.get_calculated_indexes_keys()))
        values = np.sort(values)
        for item in values:
            self.indexes_lstbox.insert(END, item)
        self.layers_lstbox.delete(0,END)
        self.ok_button['state'] = 'disabled'

    def set_layers_listbox(self, master):
        # Title just in top of selector
        lstbox_tittle = ttk.Label(master=master, text="Select Layer")
        scrollbar = tk.Scrollbar(master=master, orient="vertical")
        self.layers_lstbox = Listbox(master=master, selectmode=EXTENDED, yscrollcommand=scrollbar.set,
                         height=MAX_VALUES_VISIBLES_IN_LISTBOX)
        scrollbar.config(command=self.layers_lstbox.yview)
        lstbox_tittle.pack(side=TOP)
        self.layers_lstbox.pack(side=LEFT)
        scrollbar.pack(side=RIGHT, fill="y")
        self.layers_lstbox.bind('<<ListboxSelect>>',lambda event: self._on_change_layer_lstbox(event, self.layers_lstbox))
        self.layers_lstbox.selection_set(0)
        return self.layers_lstbox

    def _on_change_layer_lstbox(self, event,lstbox):
        selection = lstbox.curselection()
        if not len(selection) <= 0:
            self.layers_last_selection = selection
        #If activate it, enters on no-ending bucle...
        """
        else:
            for idx in self.layers_last_selection:
                lstbox.select_set(idx)
        """

    def _on_change_index_lstbox(self, event):
        selection = self.indexes_lstbox.curselection()
        if selection is not ():
            selection = selection[0]
            selection = self.indexes_lstbox.get(selection)
            self.update_layers_lstbox(selection)
            self.ok_button['state'] = 'normal'
            self.lstbox_last_selection = selection

    def update_layers_lstbox(self, index_selected):
        self.layers_lstbox.delete(0, END)
        values = self.network_data.get_layers_with_index(index_selected)
        self.layers_lstbox.insert(END, 'all')
        for item in values:
            self.layers_lstbox.insert(END, item)

