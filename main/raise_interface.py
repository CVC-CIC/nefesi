#!/usr/bin/env python

import sys
sys.path.append('..')
import nefesi.util.GPUtil as gpu
gpu.assignGPU()

if __name__ == '__main__':
    from nefesi.interface.selection_interface import SelectionInterface
    #from nefesi.util.general_functions import save_dataset_segmentation
    #save_dataset_segmentation('/datatmp/datasets/Broden+Flatted/')
    SelectionInterface()
