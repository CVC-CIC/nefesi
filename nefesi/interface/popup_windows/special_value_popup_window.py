import tkinter as tk# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3

try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk

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


class SpecialValuePopupWindow(object):
    def __init__(self, master,index=''):
        self.value = -1 #Returned value if user clicks on X
        if index.lower() == 'orientation':
            self.validate_command = (master.register(self._on_entry_updated_check_orientation_index_float),
                                     '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
            default_entry = 180
            text = ORIENTATION_TEXT
        elif index.lower() == 'population code':
            self.validate_command = (master.register(self._on_entry_updated_check_orientation_index_in_range_0_1),
                                     '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
            default_entry = 0.1
            text = POPULATION_CODE_TEXT
        self.top=Toplevel(master)
        self.top.title(index.title() + ' Selectivity')
        self.text_label=Label(self.top, text=text)
        self.text_label.pack()
        self.ok_button = Button(self.top, text='Ok', command=self.cleanup)
        self.value_entry = None
        self.value_entry=Entry(self.top, validate ='key', validatecommand=self.validate_command,
                               textvariable=StringVar(master=self.top,value=default_entry),width=20, justify=CENTER)
        self.value_entry.pack()
        self.ok_button.pack()
    def cleanup(self):
        self.value=float(self.value_entry.get())
        self.top.destroy()
    def _on_entry_updated_check_orientation_index_float(self, action, index, value_if_allowed,
                                                        prior_value, text, validation_type, trigger_type, widget_name):

        is_valid = False
        # action=1 -> insert
        if(action=='1'):
            if text in '+0123456789.':
                try:
                    #if new value is valid float
                    value = float(value_if_allowed)
                    if value>orientation_idx_range[-1] or value <orientation_idx_range[0]:
                        self.ok_button['state'] = 'disabled'
                        if self.value_entry is not None:
                            self.value_entry.config({"background": RED_LIGHTED_COLOR})
                    else:
                        #if is not in correct range
                        self.ok_button['state'] = 'normal'
                        if self.value_entry is not None:
                            self.value_entry.config({"background": 'white'})
                    return True
                except ValueError:
                    return False
            else:
                if len(value_if_allowed) == 1:
                    self.ok_button['state'] = 'disabled'
                    if self.value_entry is not None:
                        self.value_entry.config({"background": RED_LIGHTED_COLOR})
                else:
                    try:
                        value = float(prior_value)
                        if value>orientation_idx_range[-1] or value <orientation_idx_range[0]:
                            self.ok_button['state'] = 'disabled'
                            if self.value_entry is not None:
                                self.value_entry.config({"background": RED_LIGHTED_COLOR})
                        else:
                            self.ok_button['state'] = 'normal'
                            if self.value_entry is not None:
                                self.value_entry.config({"background": 'white'})
                    except:
                        self.ok_button['state'] = 'normal'
                        if self.value_entry is not None:
                            self.value_entry.config({"background": 'white'})
                return False
        #action 0 -> delete
        elif(action == '0'):
            #if will be empty
            if value_if_allowed =='':
                self.ok_button['state'] = 'disabled'
                if self.value_entry is not None:
                    self.value_entry.config({"background": RED_LIGHTED_COLOR})
            else:
                try:
                    #if last value (the value that will remain) is a valid value in range
                    value = float(value_if_allowed)
                    if value > 359 or value < 1:
                        self.ok_button['state'] = 'disabled'
                        if self.value_entry is not None:
                            self.value_entry.config({"background": RED_LIGHTED_COLOR})
                    else:
                        self.ok_button['state'] = 'normal'
                        if self.value_entry is not None:
                            self.value_entry.config({"background": 'white'})
                except:
                    pass
            return True
        else:
            self.ok_button['state'] = 'normal'
            if self.value_entry is not None:
                self.value_entry.config({"background": 'white'})
            return True


    def _on_entry_updated_check_orientation_index_in_range_0_1(self, action, index, value_if_allowed,
                                                        prior_value, text, validation_type, trigger_type, widget_name):

        is_valid = False
        # action=1 -> insert
        if(action=='1'):
            if text in '+0123456789.':
                try:
                    #if new value is valid float
                    value = float(value_if_allowed)
                    if value>POPULATION_CODE_RANGE[-1] or value <POPULATION_CODE_RANGE[0]:
                        self.ok_button['state'] = 'disabled'
                        if self.value_entry is not None:
                            self.value_entry.config({"background": RED_LIGHTED_COLOR})
                    else:
                        #if is not in correct range
                        self.ok_button['state'] = 'normal'
                        if self.value_entry is not None:
                            self.value_entry.config({"background": 'white'})
                    return True
                except ValueError:
                    return False
            else:
                if len(value_if_allowed) == 1:
                    self.ok_button['state'] = 'disabled'
                    if self.value_entry is not None:
                        self.value_entry.config({"background": RED_LIGHTED_COLOR})
                else:
                    try:
                        value = float(prior_value)
                        if value>POPULATION_CODE_RANGE[-1] or value <POPULATION_CODE_RANGE[0]:
                            self.ok_button['state'] = 'disabled'
                            if self.value_entry is not None:
                                self.value_entry.config({"background": RED_LIGHTED_COLOR})
                        else:
                            self.ok_button['state'] = 'normal'
                            if self.value_entry is not None:
                                self.value_entry.config({"background": 'white'})
                    except:
                        self.ok_button['state'] = 'normal'
                        if self.value_entry is not None:
                            self.value_entry.config({"background": 'white'})
                return False
        #action 0 -> delete
        elif(action == '0'):
            #if will be empty
            if value_if_allowed =='':
                self.ok_button['state'] = 'disabled'
                if self.value_entry is not None:
                    self.value_entry.config({"background": RED_LIGHTED_COLOR})
            else:
                try:
                    #if last value (the value that will remain) is a valid value in range
                    value = float(value_if_allowed)
                    if value > POPULATION_CODE_RANGE[-1] or value < POPULATION_CODE_RANGE[0]:
                        self.ok_button['state'] = 'disabled'
                        if self.value_entry is not None:
                            self.value_entry.config({"background": RED_LIGHTED_COLOR})
                    else:
                        self.ok_button['state'] = 'normal'
                        if self.value_entry is not None:
                            self.value_entry.config({"background": 'white'})
                except:
                    pass
            return True
        else:
            self.ok_button['state'] = 'normal'
            if self.value_entry is not None:
                self.value_entry.config({"background": 'white'})
            return True
