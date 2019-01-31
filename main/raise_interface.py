#!/usr/bin/env python

import sys
sys.path.append('..')
import nefesi.util.GPUtil as gpu
gpu.assignGPU()

if __name__ == '__main__':
    from nefesi.interface.selection_interface import SelectionInterface
    #from nefesi.util.general_functions import have_all_imagenet_segmentation
    #have_all_imagenet_segmentation('/datatmp/datasets/ImageNetFused/')
    SelectionInterface()
