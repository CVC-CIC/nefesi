import nefesi.util.GPUtil as gpu
gpu.assignGPU()

if __name__ == '__main__':
    from nefesi.interface.selection_interface import SelectionInterface
    SelectionInterface()
