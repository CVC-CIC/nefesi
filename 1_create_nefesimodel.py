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


def preproces_hpeu_img( imgs_hr,small_size=(64,64)):

    imgs_hr=np.array(imgs_hr)/256
    imgs_lr=cv2.resize(np.array(imgs_hr),small_size)


    gray = 1 - (0.299 * (imgs_lr[:, :, 0] + 1) + 0.587 * (imgs_lr[:, :, 1] + 1) + 0.114 * (imgs_lr[:, :, 2] + 1)) / 2
    gray = [np.stack((gray,) * 3, axis=-1)]

    imgs_lr = [imgs_lr]
    imgs = imgs_lr + gray
    tnsr = [transforms.ToTensor()(img) for img in imgs]


    return tnsr


def preproces_imagenet_img( imgs_hr):

    img=np.array(imgs_hr)



    tnsr = [transforms.ToTensor()(img)]


    return tnsr




def main():

    #Here You can find all the funcitons that allow us to visualize and quantify a trained Neural Network.
    # To perform the calculation taking into acount the negative activations instead of positives, uncoment line 53 in the file read_activations


    # Load the Model with your weigths first
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = torch.load('Model_generation/Savedmodel/vgg16_normal')


    deepmodel = DeepF.deep_model(model)


    # Create a list with the layers that you want to analyze and 0 if they are encoding or 1 if they are decoding
    layers_interest = [['features.1', 0], ['features.3', 0], ['features.6', 0], ['features.8', 0]]
    # layers_interest = [['features.1', 0], ['features.3', 0], ['features.6', 0], ['features.8', 0], ['features.11', 0],
    #                    ['features.13', 0], ['features.15', 0], ['features.18', 0], ['features.20', 0],
    #                    ['features.22', 0], ['features.25', 0], ['features.27', 0], ['features.29', 0]]

    # Create the DatasetLoader: select your imagepath and your preprocessing functon (in case you have one)
    Path_images='C:/Users/arias/Desktop/Dataset/tiny-imagenet-200/val/images'
    preproces_function=preproces_imagenet_img
    dataset = ImageDataset(src_dataset=Path_images,target_size=(64,64),preprocessing_function=preproces_function,color_mode='rgb')

    # Path where you will save your results
    save_path= "Nefesi_models/VGG16"



    Nefesimodel= NetworkData(model=deepmodel,layer_data=layers_interest,save_path = save_path, dataset=dataset,default_file_name = 'VGG16_imagenet',input_shape=[(1,3,64,64)])
    Nefesimodel.generate_neuron_data()

    # calculate the top scoring images
    Nefesimodel.eval_network()

    print('Activation Calculus done!')
    Nefesimodel.save_to_disk('activations')

    # Nefesimodel=NetworkData.load_from_disk("Model_generation\Nefesi_models\VGG16\VGGPartialSave100WithoutNF.obj")

    # calculate the Neuron feature of each neuron (weighted average of top scoring images)
    Nefesimodel.calculateNF()
    print('NF done!')


    # calculate the Color selectivity of each neuron
    dataset = Nefesimodel.dataset
    for layer in Nefesimodel.get_layers_name():
        layer_data = Nefesimodel.get_layer_by_name(layer)
        print(layer)
        for n in range(Nefesimodel.get_len_neurons_of_layer(layer)):
            neurona = Nefesimodel.get_neuron_of_layer(layer, n)
            neurona.color_selectivity_idx_new(Nefesimodel, layer_data, dataset)
    Nefesimodel.save_to_disk('color_indx')
    #
    # # calculate the Similarity Index of each neuron in the same layer
    # for layer in Nefesimodel.get_layers_name():
    #     print(layer)
    #     Nefesimodel.get_layer_by_name(layer).similarity_index = None
    #     x=Nefesimodel.similarity_idx(layer)
    #     print(x)
    #
    # Nefesimodel.save_to_disk('similarity')







if __name__ == '__main__':
    main()