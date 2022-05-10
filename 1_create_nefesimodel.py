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
from Model_generation.Unet import UNet



def preproces_imagenet_img( imgs_hr):

    img=np.array(imgs_hr)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    tnsr = [transform(img)]


    return tnsr




def main():

    #Here You can find all the funcitons that allow us to visualize and quantify a trained Neural Network.
    # To perform the calculation taking into acount the negative activations instead of positives, uncoment line 53 in the file read_activations


    # Load the Model with your weigths first

    # folder_dir ="C:/Users/arias/Desktop/Nefesi2022/"
    folder_dir = "/home/guillem/Nefesi2022/"


    device = torch.device(0)

    model= UNet()
    model.load_state_dict(torch.load( folder_dir+'nefesi/Model_generation/_final.pt'))


    for n, m in model.named_modules():
        m.auto_name = n
        print(n)

    deepmodel = DeepF.deep_model(model)


    # Create a list with the layers that you want to analyze and 0 if they are encoding or 1 if they are decoding
    # layers_interest = [['features.1', 0], ['features.3', 0], ['features.6', 0], ['features.8', 0]]
    layers_interest = [['down1', 0], ['down2', 0], ['down3', 0], ['down4', 0], ['up1', 1], ['up2', 1],['up3', 1],['up4', 1] ]

    # Create the DatasetLoader: select your imagepath and your preprocessing functon (in case you have one)
    Path_images=folder_dir+'Dataset/tiny-imagenet-200/train'
    preproces_function=preproces_imagenet_img
    dataset = ImageDataset(src_dataset=Path_images,target_size=(64,64),preprocessing_function=preproces_function,color_mode='rgb')

    # Path where you will save your results
    save_path= "Nefesi_models/VGG16"



    Nefesimodel= NetworkData(model=deepmodel,layer_data=layers_interest,save_path = save_path, dataset=dataset,default_file_name = 'Unet',input_shape=[(1,3,64,64)])
    Nefesimodel.generate_neuron_data()

    # calculate the top scoring images
    Nefesimodel.eval_network()

    print('Activation Calculus done!')
    Nefesimodel.save_to_disk('activations')

    # Nefesimodel=NetworkData.load_from_disk("Model_generation\Nefesi_models\VGG16\VGGPartialSave100WithoutNF.obj")

    # calculate the Neuron feature of each neuron (weighted average of top scoring images)
    Nefesimodel.calculateNF()
    print('NF done!')


    # # calculate the Color selectivity of each neuron
    # dataset = Nefesimodel.dataset
    # for layer in Nefesimodel.get_layers_name():
    #     layer_data = Nefesimodel.get_layer_by_name(layer)
    #     print(layer)
    #     for n in range(Nefesimodel.get_len_neurons_of_layer(layer)):
    #         neurona = Nefesimodel.get_neuron_of_layer(layer, n)
    #         neurona.color_selectivity_idx_new(Nefesimodel, layer_data, dataset)
    # Nefesimodel.save_to_disk('Normal_class')
    # #
    # # # calculate the Similarity Index of each neuron in the same layer
    # # for layer in Nefesimodel.get_layers_name():
    # #     print(layer)
    # #     Nefesimodel.get_layer_by_name(layer).similarity_index = None
    # #     x=Nefesimodel.similarity_idx(layer)
    # #     print(x)
    # #
    # # Nefesimodel.save_to_disk('similarity')
    #






if __name__ == '__main__':
    main()