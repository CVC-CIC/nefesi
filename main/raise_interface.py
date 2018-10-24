import nefesi.util.GPUtil as gpu
print(gpu.assignGPU())
import os
print(os.environ['CUDA_VISIBLE_DEVICES'])

if __name__ == '__main__':
    from nefesi.interface.selection_interface import SelectionInterface
    SelectionInterface()
