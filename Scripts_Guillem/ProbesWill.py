import sys
from nefesi.network_data import NetworkData
import numpy as np
from nefesi.util import plotting
sys.path.append('..')
import pickle
import matplotlib.pyplot as plt
from nefesi.util.plotting import plot_histogram


def main():
    # nefesimodel = NetworkData.load_from_disk(file_name="../Data/vgg16.obj", model_file="../Data/VGG16.h5")
    # layer=nefesimodel.get_layer_by_name('block5_conv2')
    # neurons, act = nefesimodel.decomposition(['block5_conv3', 494], layer)
    # np.save('494Activations',act)
    # np.save('494Neurons', neurons)



    a=np.load('279Activations.npy')
    b = np.load('279Neurons.npy')
    b = b.astype(int)
    marges=3
    Histo , Neurons =plotting.plot_histogram_decomp(b,a,marges)

    print(Histo)
    print(Neurons)

    plt.subplot(221)
    plt.imshow(a[marges:-(marges+1),marges:-(marges+1),0])
    plt.subplot(222)
    plt.imshow(a[marges:-(marges+1),marges:-(marges+1), 1])
    plt.subplot(223)
    plt.imshow(a[marges:-(marges+1),marges:-(marges+1), 2])
    plt.subplot(224)
    plt.imshow(a[marges:-(marges+1), marges:-(marges+1), 3])
    plt.show()



    # d1, d2, d3 = nefesimodel.tree_decomposition(['block5_conv3', 258],10)
    # np.savetxt('258_Activations.txt', d1, fmt='%f')
    # np.savetxt('258_Neurons.txt', d2, fmt='%d')
    #
    # print(d2)
    # d1, d2, d3 = nefesimodel.tree_decomposition(['block5_conv3', 279],10)
    # np.savetxt('279_Activations.txt', d1, fmt='%f')
    # np.savetxt('279_Neurons.txt', d2, fmt='%d')
    #
    # # print(d2)
    # d1, d2, d3 = nefesimodel.tree_decomposition(['block5_conv3', 333],10)
    # np.savetxt('333_Activations.txt', d1, fmt='%f')
    # np.savetxt('333_Neurons.txt', d2, fmt='%d')
    #
    # print(d2)

    # d1,d2,d3=nefesimodel.tree_decomposition(['block5_conv3' , 258])
    # print(d1,d2,d3)
    # res_act, res_neurons, res_loc, res_nf = nefesimodel.decomposition(input_image=['block5_conv3' , 258],target_layer= 'block5_conv2')
    # print(res_act)
    # print(res_neurons)
    # print(res_loc[0])
    # print(res_nf)






if __name__ == '__main__':
    main()

