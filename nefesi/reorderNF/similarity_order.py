import dill as pickle
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.append('../')
from nefesi.network_data import NetworkData

from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import consensus_score
import numpy as np

layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']

my_net = pickle.load(open('block1_3.obj', 'rb'))

data = my_net.layers[1].similarity_index


plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")
model = SpectralBiclustering(n_clusters=5, method='log', affinity='precomputed',
                             random_state=0)

model = SpectralClustering(n_clusters=5,affinity='precomputed',
                             random_state=0)
model.fit(data)


fit_data = data[np.argsort(model.labels_)]
fit_data = fit_data[:, np.argsort(model.labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.matshow(np.outer(np.sort(model.row_labels_) + 1,
                     np.sort(model.column_labels_) + 1),
            cmap=plt.cm.Blues)
plt.title("Checkerboard structure of rearranged data")

plt.show()


from nefesi.util.plotting import plot_neuron_features
list_nf = my_net.layers[1].filters[np.argsort(model.labels_)] 
plot_neuron_features(my_net.layers[1], list_nf )