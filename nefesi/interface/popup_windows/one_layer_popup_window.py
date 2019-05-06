
try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk
import numpy as np
from ...util.general_functions import clean_widget
from ...util.interface_plotting import ORDER
from math import ceil

orientation_idx_range = (1,359)
RED_LIGHTED_COLOR = '#ffcccc'
ORDER_FRAME_TEXT = "If crop show: "
CONDITIONS_TEXT = 'Define constraints: *'
SELECTOR_OPTIONS = ['highers', 'lowers', 'in range', 'value sel.']
CONDITIONS = ['<','<=','==', '>=', '>']
NEURONS_TO_SHOW_RANGE = (1,20)
NEURONS_TO_SHOW_TEXT = "Max. neurons to show: "
POPULATION_CODE_RANGE = (0., 1.)
ORIENTATION_TEXT = 'Set degrees of each rotation\n'\
            '(only values in range [1,359] allowed).\n'\
        'NOTE: Lower values will increment processing time'
POPULATION_CODE_TEXT = 'Set the threshold to consider selective\n'\
                        '(only values in range [0.,1.] allowed).\n'\
                'NOTE: lower threshold values can introduce classes\n' \
                    'that neurons are not really being selectives'

class OneLayerPopupWindow(object):
    def __init__(self, master, layer_to_evaluate,index='Sel. idx', special_value = 0.1):
        index = index.lower()
        if index != 'population code':
            self.validate_command_entry_1 = (master.register(self._on_entry_updated_check_range_1_0_entry_1),
                                         '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
            self.validate_command_entry_2 = (master.register(self._on_entry_updated_check_range_1_0_entry_2),
                                     '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
            self.footer1="* Accepted range [0., 1.]"
            self.entry1_default = 0.0
            self.entry2_default = 1.0
            self.range_label_text = index.capitalize()
        else:
            self.validate_command_entry_1 = (master.register(self._on_entry_updated_check_non_negative_int_entry_1),
                                             '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
            self.validate_command_entry_2 = (master.register(self._on_entry_updated_check_non_negative_int_entry_2),
                                             '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
            self.footer1 = "* Accepted range [0, ∞)"
            self.entry1_default = 0
            self.entry2_default = ceil(1/special_value)
            self.range_label_text = 'Popul. code'

        self.condition1 = self.combo1 = self.condition2 = self.combo2 = self.value1 = self.entry1 = self.value2 = \
            self.entry2 = self.order_combo = self.order = self.neurons_to_show_entry = self.neurons_to_show = None

        self.top=Toplevel(master)
        self.top.title(index.title() + ' Selectivity')
        header_text = index.title()+" selectivity of layer "+layer_to_evaluate+":\n" \
                    "Define the chart constraints, to filter\n" \
                        " the neurons that will be showed"
        self.text_label= Label(self.top, text=header_text,height=5)
        self.ok_button = ttk.Button(self.top, text='Ok', command=self.cleanup)
        self.selection_type_selector_frame = ttk.Frame(self.top)
        self.set_selection_type_selector(master=self.selection_type_selector_frame)
        self._type_selector_last_value = SELECTOR_OPTIONS[0]
        self.conditions_frame = ttk.Frame(self.top)
        self.set_conditions_selector(master=self.conditions_frame)
        self.order_selector_frame = ttk.Frame(master=self.top)
        self.set_order_selector(self.order_selector_frame)
        self.max_naurons_to_show_frame = ttk.Frame(master=self.top)
        self.set_max_neurons_to_show_entry(master=self.max_naurons_to_show_frame)
        self.text_label.pack()
        self.selection_type_selector_frame.pack()
        self.conditions_frame.pack()
        self.order_selector_frame.pack()
        self.max_naurons_to_show_frame.pack()
        self.ok_button.pack(pady=(8,5), ipadx=10)
        self.set_footers(master=self.top)

    def set_footers(self, master):
        frame = ttk.Frame(master=master)
        label = ttk.Label(master=frame, text=self.footer1,font=("Times New Roman", 8))
        label.grid(row=0,padx=(75,0))
        label = ttk.Label(master=frame, text="** Accepted range [" + str(NEURONS_TO_SHOW_RANGE[0]) + ", " + \
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
        label = ttk.Label(master=master, text=NEURONS_TO_SHOW_TEXT)
        validate_command = (master.register(self._on_entry_updated_check_max_neurons_validity),
                             '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        self.neurons_to_show_entry = Entry(master, validate='key', validatecommand=validate_command,
                           textvariable=StringVar(master=self.top, value=15),justify=CENTER,width = 5)
        label.grid(column=0,row=0)
        self.neurons_to_show_entry.grid(column=1,row=0)
        label = ttk.Label(master=master, text="**")
        label.grid(column=2,row=0)

    def set_order_selector(self, master):
        label = ttk.Label(master=master, text=ORDER_FRAME_TEXT)
        self.order_combo = ttk.Combobox(master=master, values=ORDER, state='readonly', width=8, justify=CENTER)
        self.order_combo.set(ORDER[0])
        self.order_combo.bind("<<ComboboxSelected>>", self._on_order_or_condition_selector_changed)
        label.pack(side=LEFT,padx=(2,25))
        self.order_combo.pack(side=RIGHT)


    def set_conditions_selector(self, master, default1='>=', default2=None):
        assert((default1 in CONDITIONS) or (default2 in CONDITIONS))
        values_frame = ttk.Frame(master=master)
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

        label = ttk.Label(master=values_frame, text = self.range_label_text)
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

        label_explication= ttk.Label(master=master, text=CONDITIONS_TEXT)
        label_explication.pack(side=TOP,padx=(0,75))
        values_frame.pack(side=BOTTOM,padx=5)

    def set_selection_type_selector(self, master):
        label = ttk.Label(master=master,text="Select constraints: ")
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
