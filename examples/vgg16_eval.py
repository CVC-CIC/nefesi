
from keras.applications.vgg16 import VGG16, preprocess_input
from nefesi.network_data import NetworkData

import os
import pickle

def main():
    # dataset = '/home/oprades/ImageNet/train/'  # dataset path
    #
    # model = VGG16()
    # my_net = NetworkData(model, dataset_path=dataset, save_path='/home/oprades/')
    # my_net.eval_network(['block1_conv1'],
    #                     target_size=(224, 224),
    #                     preprocessing_function=preprocess_input)


    t = NetworkData.load_from_disk(path='/home/oprades/', file_name='block1_conv1.obj')

    print(t.model, t.layers)
    print(t.dataset.preprocessing_function, t.dataset.src_dataset)

    for f in t.layers[0].get_filters():
        f.print_params()







if __name__ == '__main__':
    # print os.path.dirname(os.path.abspath(__file__))
    # print os.getcwd()

    main()
