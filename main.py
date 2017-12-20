
import numpy as np

from vgg_matconvnet import VGG
from network_data import NetworkData


import time
import pickle




def main():

    dataset = '/home/oprades/ImageNet/train/'
    save_path = '/data/users/oscar/'
    layer_names = ['activation_1', 'activation_2']
    avg_img = np.load('averageImage.npy')
    num_max_activations = 100

    def norm_input(x):
        return x-avg_img

    model = VGG()

    # my_net = NetworkData(model, layer_names, dataset, num_max_activations=num_max_activations, save_path=save_path)
    # my_net.eval_network(1280000, (224, 224), batch_size=100, preprocessing_function=norm_input)
    # my_net.save()




    my_net = pickle.load(open('sequential_1.obj', 'rb'))
    my_net.model = model
    my_net.dataset_path = dataset

    # sel_idx = my_net.selectivity_idx_summary(['color'], layer_names)
    # print sel_idx['color'][0]
    # print sel_idx['color'][1]

    # sel_idx = my_net.selectivity_idx_summary(['orientation'], layer_names)
    # print sel_idx['orientation'][0]
    # print sel_idx['orientation'][1]

    sel_idx = my_net.selectivity_idx_summary(['symmetry'], layer_names)
    print sel_idx['symmetry'][0]
    print sel_idx['symmetry'][1]




if __name__=='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
