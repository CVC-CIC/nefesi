"""
Based on a trained model in Keras, this file is to transform its weight into a Pytorch model.
"""
import numpy as np

import keras
from keras.models import load_model

import torch


def keras_to_pyt(km, pm):
    keras_weight_list = []
    for layer in km.layers:
        if type(layer) is keras.layers.convolutional.Conv2D:
            # name = layer.get_config()['name']
            layer_weight = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
            layer_bias = layer.get_weights()[1]
            keras_weight_list.append(layer_weight)
            keras_weight_list.append(layer_bias)
        elif type(layer) is keras.layers.Dense:
            layer_weight = np.transpose(layer.get_weights()[0], (1, 0))
            layer_bias = layer.get_weights()[1]
            keras_weight_list.append(layer_weight)
            keras_weight_list.append(layer_bias)
    pyt_state_dict = pm.state_dict()
    for index, key in enumerate(pyt_state_dict.keys()):
        pyt_state_dict[key] = torch.from_numpy(keras_weight_list[index])
    pm.load_state_dict(pyt_state_dict)
    return pm

if __name__ == "__main__":
    keras_model = load_model("./vgg16.h5")
    pytorch_model = torch.load("./vgg16_pytorch.pkl")
    pytorch_model = keras_to_pyt(keras_model, pytorch_model)
    torch.save(pytorch_model, 'vgg16_pytorch_transform_from_keras.pkl')


