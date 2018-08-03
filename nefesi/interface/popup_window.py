import tkinter as tk# note that module name has changed from Tkinter in Python 2 to tkinter in Python 3
try:
    from tkinter import *
    from tkinter import ttk
except ImportError:
    from Tkinter import *
    from tkinter import ttk

orientation_idx_range = (1,359)

class PopupWindow(object):
    def __init__(self, master, text='Set the value',index='' ):
        self.value = -1 #Returned value if user clicks on X
        self.top=Toplevel(master)
        self.top.title(index + ' Selectivity')
        self.text_label=Label(self.top, text=text)
        self.text_label.pack()
        self.ok_button = Button(self.top, text='Ok', command=self.cleanup)
        if index.lower() == 'orientation':
            self.validate_command = (master.register(self._on_entry_updated_check_orientation_index_float),
                                     '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        self.value_entry=Entry(self.top, validate ='key', validatecommand=self.validate_command,
                               textvariable=StringVar(master=self.top,value=180))
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
            if text in '-+0123456789.':
                try:
                    #if new value is valid float
                    value = float(value_if_allowed)
                    if value>orientation_idx_range[-1] or value <orientation_idx_range[0]:
                        self.ok_button['state'] = 'disabled'
                    else:
                        #if is not in correct range
                        self.ok_button['state'] = 'normal'
                    return True
                except ValueError:
                    return False
            else:
                if len(value_if_allowed) == 1:
                    self.ok_button['state'] = 'disabled'
                else:
                    try:
                        value = float(prior_value)
                        if value>orientation_idx_range[-1] or value <orientation_idx_range[0]:
                            self.ok_button['state'] = 'disabled'
                        else:
                            self.ok_button['state'] = 'normal'
                    except:
                        self.ok_button['state'] = 'normal'
                return False
        #action 0 -> delete
        elif(action == '0'):
            #if will be empty
            if value_if_allowed =='':
                self.ok_button['state'] = 'disabled'
            else:
                try:
                    #if last value (the value that will remain) is a valid value in range
                    value = float(value_if_allowed)
                    if value > 359 or value < 1:
                        self.ok_button['state'] = 'disabled'
                    else:
                        self.ok_button['state'] = 'normal'
                except:
                    pass
            return True
        else:
            self.ok_button['state'] = 'normal'
            return True
