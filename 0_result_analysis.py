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
    file_name = 'C:/Users/arias/Desktop/Nefesi2022/Nefesi_models/VGG16/color_indx.obj'

    # file_name= 'C:/Users/g84194584/Desktop/Code/nefesi/nefesi/hpeu_imagenet_good/Pos_Neg.obj'
    # model_path = "C:/Users/g84194584/Desktop/Code/nefesi/HPEU/mse_net_latest.pth"

    Nefesimodel = NetworkData.load_from_disk(file_name, model_file=None)
    device = torch.device('cuda:{}'.format(0))

    model = torch.load('Model_generation/Savedmodel/vgg16_normal')

    Nefesimodel.model = model

    layers = Nefesimodel.get_layer_names_to_analyze()

    for layer in Nefesimodel.layers_data:
        for neuron in layer.neurons_data:
            nf=neuron._neuron_feature
            plt.imshow(nf/265)
            plt.show()

    #
    # for layer in layers:
    #
    #     activations[layer]= [n.activations[0] for n in Nefesimodel.get_layer_by_name(layer).neurons_data]
    #     # negatives = [n.negative_activations[0] for n in Nefesimodel.get_layer_by_name(layer).neurons_data]
    #     # activations[layer] = [max(activations[layer][x],negatives[x]) for x in range(len(negatives))]
    #
    #     activations[layer]= activations[layer]/max(activations[layer])
    #     activationorder[layer]= np.argsort(activations[layer])[::-1]
    #     activations[layer]=activations[layer][activationorder[layer]]
    #     colorindex=np.array([n.selectivity_idx['color'] for n in Nefesimodel.get_layer_by_name(layer).neurons_data])
    #     colorindex = colorindex[activationorder[layer]]
    #     dict_delete[layer] = []
    #     # dict_delete[layer] = [x for i,x in enumerate(activationorder[layer]) if activations[layer][i]<threshold and colorindex[i] > color_threshold ]
    #     # dict_delete[layer] = [x for i, x in enumerate(activationorder[layer]) if   activations[layer][i] > threshold and colorindex[i] > color_threshold]
    #     # neurons_removed+=len(dict_delete[layer])
    #
    # # print(activations)
    #
    #
    #
    #
    #
    #
    #
    # # dict_delete['down1']=[8]
    # # dict_delete['leu1'] = [17]
    #
    # # neuronum=2
    # #
    # # # dict_delete['srm_up5'] = [13]
    # # # dict_delete['down1'] = [8]
    # # # dict_delete['conv_in'] = [9,13]
    # # dict_delete['srm_up5'] = [neuronum]
    # #
    # #
    # py_model = model.pytorchmodel
    #
    # # image = np.array(Image.open('D:/Dataset/Imagenet64/1/1_batch1_1630.jpg')) / 255
    # # image=np.array(Image.open('D:/Dataset/checker.png'))/255
    # image = np.array(Image.open('C:/Users/g84194584/Desktop/Code/hpeu_gf/dataset/special_light/jpg/119.jpg')) / 255
    # # image=np.array(Image.open('C:/Users/g84194584/Desktop/Imagenet64/12/12_batch1_82007.jpg'))/255
    #
    # tensor_im = preproces_hpeu_single_img(image)
    # prediction = py_model(*tensor_im)
    # image_out = np.einsum('kli->lik', np.squeeze(prediction.cpu().detach().numpy()))
    # image_out[image_out < 0] = 0
    # image_out[image_out > 1] = 1
    # imatge = Image.fromarray((image_out * 255).astype(np.uint8))
    #
    # for layer in layers:
    #     handle = rgetattr(py_model, layer).register_forward_hook(ablation_hook)
    #
    #
    # prediction2 = py_model(*tensor_im)
    # image_out2 = np.einsum('kli->lik', np.squeeze(prediction2.cpu().detach().numpy()))
    # image_out2[image_out2 < 0] = 0
    # # image_out2[image_out2 >1] = 1
    # imatge2 = Image.fromarray((image_out2 * 255).astype(np.uint8))
    #
    # print(np.mean(abs(image_out-image_out2))*100)
    #
    #
    # # print(layers)
    # # plt.subplot(2, 2, 1)
    # # plt.imshow(image)
    # # plt.title('original image')
    # # plt.subplot(2,2,2)
    # # plt.imshow(imatge)
    # # plt.title('original output')
    # # plt.subplot(2, 2, 3)
    # plt.imshow(imatge2)
    # plt.title('without clipping values')
    # print(np.mean(abs(image_out2 - image_out)))
    # print(neurons_removed)
    # plt.show()
    #
    #
    # # for j in range(len(layers)):
    # #     max = np.max(stored_outputs[j][0, :, :, :].cpu().detach().numpy())
    # #     min = -max
    # #     all_maps = stored_outputs[j][0, :, :, :].cpu().detach().numpy()
    # #     max_activation_values = np.sum(np.sum(abs(all_maps), axis=1), axis=1)
    # #     order=np.argsort(max_activation_values)
    # #     for index,i in enumerate(order[::-1][:64]):
    # #
    # #
    # #         featuremaps = all_maps[i,:,:]
    # #
    # #
    # #
    # #         plt.subplot(8,8,index+1)
    # #         # plt.title(layers[j]+ ' '+ str(i))
    # #         plt.title(int(max_activation_values[i]))
    # #
    # #         plt.imshow(featuremaps, vmin=min, vmax=max,cmap='seismic')
    # #
    # #     plt.show()
    #







if __name__ == '__main__':
    main()

