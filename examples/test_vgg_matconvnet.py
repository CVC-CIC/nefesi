
import pickle

import numpy as np

from external.vgg_matconvnet import VGG


def main():

    dataset = '/home/oprades/ImageNet/train/'
    save_path = '/data/users/oscar/'
    layer_names = ['activation_1', 'activation_2']
    avg_img = np.load('external/averageImage.npy')
    num_max_activations = 100

    def norm_input(x):
        return x-avg_img

    model = VGG()

    # my_net = NetworkData(model, layer_names, dataset, num_max_activations=num_max_activations, save_path=save_path)
    # my_net.eval_network(1280000, (224, 224), batch_size=100, preprocessing_function=norm_input)
    # my_net.save()




    my_net = pickle.load(open('external/sequential_1.obj', 'rb'))
    my_net.model = model
    my_net.dataset_path = dataset

    # sel_idx = my_net.selectivity_idx_summary(['color'], layer_names)
    # print sel_idx['color'][0]
    # print sel_idx['color'][1]

    # sel_idx = my_net.selectivity_idx_summary(['orientation'], layer_names)
    # print sel_idx['orientation'][0]
    # print sel_idx['orientation'][1]

    # sel_idx = my_net.selectivity_idx_summary(['symmetry'], layer_names)
    # print sel_idx['symmetry'][0]
    # print sel_idx['symmetry'][1]

    labels = pickle.load(open('external/labels_imagenet.obj', 'rb'))
    sel_idx = my_net.selectivity_idx_summary(['class'], layer_names, labels=labels)
    print sel_idx['class'][0]
    print sel_idx['class'][1]


if __name__=='__main__':
    import osgit
    import sys
    from nefesi import network_data, layer_data, neuron_data

    # workaround. In this way, pickle can find the modules as if they were installed in the system
    sys.modules['network_data'] = network_data
    sys.modules['layer_data'] = layer_data
    sys.modules['neuron_data'] = neuron_data


    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
