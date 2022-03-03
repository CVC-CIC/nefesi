"""
This file contains a toy examples to have a first contact with Nefesi, and Keras.
This file has been created with tensorflow (and tensorflow-gpu) 1.8.0, keras 2.2.0, and python 3.6 (with anaconda3 interpreter)
"""
import cv2
import torch
from torchvision import transforms
import numpy as np
import functools
from functions.network_data2 import NetworkData
import types
import functions.GPUtil as gpu
BATCH_SIZE = 100
from  functions.image import ImageDataset
from functions.read_activations import get_activations
import interface_DeepFramework.DeepFramework as DeepF
import matplotlib.pyplot as plt


def preproces_imagenet_img( imgs_hr):

    img=np.array(imgs_hr)



    tnsr = [transforms.ToTensor()(img)]


    return tnsr




def main():

    #Here You can find all the funcitons that allow us to visualize and quantify a trained Neural Network.
    # To perform the calculation taking into acount the negative activations instead of positives, uncoment line 53 in the file read_activations


    # Load the Model with your weigths first

    # folder_dir ="C:/Users/arias/Desktop/Nefesi2022/"
    folder_dir = "/home/guillem/Nefesi2022/"


    # device = torch.device("cuda" if torch.cuda.is_available()
    #                       else "cpu")
    # model = torch.load( folder_dir+'Nefesi/Model_generation/Savedmodel/vgg16_class_positive')



    Nefesimodel= NetworkData.load_from_disk(folder_dir+'Nefesi/Nefesi_models/VGG16/Normal_class')
    Nefesimodel2= NetworkData.load_from_disk(folder_dir+'Nefesi/Nefesi_models/VGG16/Positive_class')
    Nefesimodel3= NetworkData.load_from_disk(folder_dir+'Nefesi/Nefesi_models/VGG16/Negative_class')


    for i in range(10):

        plt.subplot(3,1,1)
        neurona = Nefesimodel.get_neuron_of_layer('features.27', i)
        plt.imshow(neurona._neuron_feature/256)
        plt.subplot(3, 1, 2)
        neurona2 = Nefesimodel2.get_neuron_of_layer('features.27', i)
        plt.imshow(neurona2._neuron_feature / 256)
        plt.subplot(3, 1, 3)
        neurona3 = Nefesimodel3.get_neuron_of_layer('features.27', i)
        plt.imshow(neurona3._neuron_feature / 256)
        plt.show()






if __name__ == '__main__':
    main()