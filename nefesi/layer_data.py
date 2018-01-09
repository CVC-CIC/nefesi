
import numpy as np
from nefesi.read_activations import get_sorted_activations, get_activations
from nefesi.neuron_feature import compute_nf
from nefesi.similarity_index import get_similarity_index

class LayerData(object):

    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.similarity_index = None
        self.filters = None


    def get_layer_id(self):
        return self.layer_id

    def set_max_activations(self):
        for f in self.filters:
            f.set_max_activations()

    def evaluate_activations(self, file_names, images, model, num_max_activations, batch_size):
        self.filters = get_sorted_activations(file_names, images, model,
                                              self.layer_id, self.filters, num_max_activations, batch_size)

    def build_neuron_feature(self, dataset, model):
        self.filters = compute_nf(dataset, model, self.layer_id, self.filters)

    def get_filters(self):
        return self.filters

    def get_selectivity_idx(self, model, index_name, dataset, labels=None, **kwargs):
        sel_idx = []
        for f in self.filters:
            if index_name == 'color':
                res = f.color_selectivity_idx(model, self.layer_id, self.filters.index(f), dataset)
                sel_idx.append(res)
            elif index_name == 'orientation':
                degrees = kwargs.get('degrees')
                n_rotations = kwargs.get('n_rotations')
                res = f.orientation_selectivity_idx(model, self.layer_id, self.filters.index(f), dataset,
                                                    degrees, n_rotations)
                sel_idx.append(res)
            elif index_name == 'symmetry':
                res = f.symmetry_selectivity_idx(model, self.layer_id, self.filters.index(f), dataset)
                sel_idx.append(res)
            elif index_name == 'class':
                if labels is None:
                    print 'Error message, in layer:', self.layer_id, ', No labels.'
                    return None
                res = f.class_selectivity_idx(labels)
                sel_idx.append(res)
        return sel_idx

    def get_similarity_idx(self, model, dataset):
        if self.similarity_index is not None:
            return self.similarity_index
        else:
            if self.filters is None:
                print 'Error message, in layer:', self.layer_id, ', No filters.'
                return None
            else:
                size = len(self.filters)
                self.similarity_index = np.zeros((size, size))

                for i in xrange(size-1):
                    for j in xrange(i+1, size):
                        sim_idx = get_similarity_index(self.filters[i], self.filters[j], i, j,
                                                       model, self.layer_id, dataset)
                        self.similarity_index[i][j] = sim_idx

                return self.similarity_index

    # decomposition
    def _decomposition_image(self, model, image):

        name_img = self.filters[0].get_images_id()[0]
        # image = load_images('/home/oprades/ImageNet/train/', [name_img])

        activations = get_activations(model, image, print_shape_only=True, layer_name=self.layer_id)

        activations = activations[0]
        _, w, h, c = activations.shape

        hc_activations = np.zeros((w, h, c))
        hc_idx = np.zeros((w, h, c))

        for i in xrange(w):
            for j in xrange(h):
                tmp = activations[0, i, j, :]
                idx = np.argsort(tmp)
                idx = idx[::-1]
                hc_activations[i, j, :] = tmp[idx]
                hc_idx[i, j, :] = idx

        print activations[0, 0, 0, :]
        print hc_activations[0, 0, :]
        print hc_idx[0, 0, :]
        return hc_activations, hc_idx



