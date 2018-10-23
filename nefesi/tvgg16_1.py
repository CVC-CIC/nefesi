
import pickle

import numpy as np

from network_data import NetworkData
from util.image import ImageDataset
from keras.applications.vgg16 import VGG16
import nefesi.util.GPUtil as gpu
gpu.assignGPU()

def main():

    dataset = '/home/oprades/ImageNet/train/' # dataset path
    save_path = '/home/eric/Nefesi/'
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    num_max_activations = 100


    model = VGG16()

    my_net = pickle.load(open(save_path + 'vgg16.obj', 'rb'))
    my_net.model = model
    img_dataset = ImageDataset(dataset, (224, 224), norm_input)
    my_net.dataset = img_dataset
    my_net.save_path = save_path

    # first neuron of the first layer from the model
    my_net.get_layers()[0].get_filters()[0].print_params()



if __name__ == '__main__':
    avg_img = np.load('external/averageImage.npy')

    def norm_input(x):
        return x-avg_img

    main()
