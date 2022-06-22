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

BATCH_SIZE = 100

import interface_DeepFramework.DeepFramework as DeepF
import matplotlib.pyplot as plt
import io
import os
from PIL import Image
from sklearn.manifold import TSNE
import scipy.spatial.distance as ssd


def preproces_hpeu_img(imgs_hr, small_size=(300, 300)):
    imgs_lr = cv2.resize(np.array(imgs_hr), small_size)

    gray = 1 - (0.299 * (imgs_lr[:, :, 0] + 1) + 0.587 * (imgs_lr[:, :, 1] + 1) + 0.114 * (imgs_lr[:, :, 2] + 1)) / 2
    gray = [np.stack((gray,) * 3, axis=-1)]

    imgs_lr = [imgs_lr]
    imgs = imgs_lr + gray
    tnsr = [transforms.ToTensor()(img) for img in imgs]

    return tnsr


dict_delete = {}


def preproces_hpeu_single_img(imgs_hr, small_size=(1000, 800)):
    imgs_lr = cv2.resize(np.array(imgs_hr), small_size)

    gray = 1 - (0.299 * (imgs_lr[:, :, 0] + 1) + 0.587 * (imgs_lr[:, :, 1] + 1) + 0.114 * (imgs_lr[:, :, 2] + 1)) / 2
    gray = [np.stack((gray,) * 3, axis=-1)]

    imgs_lr = [imgs_lr]
    imgs = imgs_lr + gray
    tnsr = [torch.unsqueeze(transforms.ToTensor()(img).float().cuda(), 0) for img in imgs]

    return tnsr


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))

stored_outputs=[]
def ablation_hook(idx, inputs, output):
    # print(dict_delete[idx.name])
    stored_outputs.append(output)
    output[:, dict_delete[idx.name], :, :] = 0

    return output


def main():
    file_name = '/home/guillem/Nefesi2022/nefesi/Nefesi_models/UNet/Unet_pos_final.obj'
    Nefesimodel = NetworkData.load_from_disk(file_name, model_file=None)

    for layer in Nefesimodel.layers_data:
        similarity=layer.similarity_index
        np.fill_diagonal(similarity,0)
        neurons=np.unravel_index(np.argmax(similarity),similarity.shape)
        print(sum(sum(similarity > 1)))
        print(similarity.shape[0]*similarity.shape[1])
        # neuron1=layer.neurons_data[neurons[0]]
        # neuron2=layer.neurons_data[neurons[1]]
        #
        # plt.subplot(2,1,1)
        # plt.imshow(neuron1._neuron_feature/256)
        # plt.subplot(2, 1, 2)
        # plt.imshow(neuron2._neuron_feature/256)
        # plt.show()






if __name__ == '__main__':
    main()

