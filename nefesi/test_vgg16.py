
import pickle

import numpy as np

from nefesi.network_data import NetworkData
from keras.applications.vgg16 import VGG16
import nefesi.util.GPUtil as gpu
gpu.assignGPU()

def main():

    dataset = '/data/local/datasets/ImageNet/train/'
    save_path = '/data/115-1/users/oscar/'
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    avg_img = np.load('../nefesi/external/averageImage.npy')
    num_max_activations = 100

    def norm_input(x):
        return x-avg_img

    model = VGG16()

    my_net = NetworkData(model, layer_names, dataset, num_max_activations=num_max_activations, save_path=save_path)
    my_net.eval_network(1280000, (224, 224), batch_size=100, preprocessing_function=norm_input)
    my_net.save()
    #
    #
    # my_net = pickle.load(open('../external/sequential_1.obj', 'rb'))
    # my_net.model = model
    # my_net.dataset_path = dataset  # delete when model will be evaluated again!!!
    # my_net.dataset = ImageDataset(dataset, (224, 224), norm_input)


    # sel_idx = my_net.selectivity_idx_summary(['color'], layer_names)
    # print sel_idx['color'][0]
    # print sel_idx['color'][1]

    # sel_idx = my_net.selectivity_idx_summary(['orientation'], layer_names)
    # print sel_idx['orientation'][0]
    # print sel_idx['orientation'][1]

    # sel_idx = my_net.selectivity_idx_summary(['symmetry'], layer_names)
    # print sel_idx['symmetry'][0]
    # print sel_idx['symmetry'][1]

    # labels = pickle.load(open('external/labels_imagenet.obj', 'rb'))
    # sel_idx = my_net.selectivity_idx_summary(['class'], layer_names, labels=labels)
    # print sel_idx['class'][0]
    # print sel_idx['class'][1]

    # for l in my_net.get_layers():
    #     l.similarity_index = None  # delete when model will be evaluated again!!!!
    #
    # sim_idx = my_net.similarity_index(layer_names)
    # print sim_idx[0]
    # print sim_idx[1]


    # l = my_net.get_layers()[0]
    # l._decomposition_image(model, 0)


if __name__ == '__main__':
    main()
