import sys
from nefesi.network_data import NetworkData
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')


def main():
    nefesimodel = NetworkData.load_from_disk(file_name='/home/guillem/nefesi/Data/WithImageNet/vgg16Copy.obj', model_file='/home/guillem/nefesi/Data/WithImageNet/vgg16.h5')
    numero_neurona=453
    my_neuron=nefesimodel.get_neuron_of_layer('block5_conv3',numero_neurona)
    layer_names=nefesimodel.get_layer_names_to_analyze()
    layer_num=layer_names.index('block5_conv3')
    num_branches=2
    num_rows=5
    num_colums=num_branches**(num_rows-1)


    fig, ax = plt.subplots(nrows=num_rows, ncols=num_colums)

    for ax1 in ax:
        for axi in ax1:
            axi.axis('off')
    ax[0,int(num_colums/num_branches-1)].imshow(my_neuron.neuron_feature)
    ax[0,int(num_colums/num_branches-1)].set_title(numero_neurona)
    list_neurons=[my_neuron]
    for i in range(1,5):
        layer_num-=1
        son_layer=layer_names[layer_num]
        son_neurons=[]
        for neuron in list_neurons:
            ablations=neuron.relevance_idx[son_layer]
            top_relevant_neurons=np.argsort(ablations)
            top_relevant_neurons=top_relevant_neurons[::-1]
            top_relevant_neurons=top_relevant_neurons[:2]
            for x in top_relevant_neurons:
                son_neurons.append(x)

        list_neurons=[nefesimodel.get_neuron_of_layer(son_layer,x) for x in son_neurons]
        for j,neuron in enumerate(list_neurons):
            ax[i,int((j+1)*num_colums/(num_branches**i+1))].imshow(neuron.neuron_feature)
            ax[i,int((j+1)*num_colums/(num_branches**i+1))].set_title(son_neurons[j])



    plt.show()


if __name__ == '__main__':
    main()


