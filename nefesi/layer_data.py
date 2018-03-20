
import numpy as np
from itertools import permutations
from read_activations import get_sorted_activations, get_activations
from neuron_feature import compute_nf, get_image_receptive_field
from similarity_index import get_similarity_index

class LayerData(object):

    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.similarity_index = None
        self.filters = None
        self.receptive_field_map = None
        self.receptive_field_size = None


    def get_layer_id(self):
        return self.layer_id

    def set_max_activations(self):
        for f in self.filters:
            f.set_max_activations()

    def evaluate_activations(self, file_names, images, model, num_max_activations, batch_size):
        self.filters = get_sorted_activations(file_names, images, model,
                                              self.layer_id, self.filters, num_max_activations, batch_size)

    def build_neuron_feature(self, network_data):
        compute_nf(network_data, self, self.filters)

    def get_filters(self):
        return self.filters

    def get_selectivity_idx(self, model, index_name, dataset, labels=None, **kwargs):
        sel_idx = []
        for f in self.filters:
            if index_name == 'color':
                res = f.color_selectivity_idx(model, self, self.filters.index(f), dataset)
                sel_idx.append(res)
            elif index_name == 'orientation':
                degrees = kwargs.get('degrees')
                n_rotations = kwargs.get('n_rotations')
                res = f.orientation_selectivity_idx(model, self, self.filters.index(f), dataset,
                                                    degrees, n_rotations)
                sel_idx.append(res)
            elif index_name == 'symmetry':
                res = f.symmetry_selectivity_idx(model, self, self.filters.index(f), dataset)
                sel_idx.append(res)
            elif index_name == 'class':
                if labels is None:
                    print 'Error message, in layer:', self.layer_id, ', No labels.'
                    return None
                res = f.class_selectivity_idx(labels)
                sel_idx.append(res)
        return sel_idx

    def get_similarity_idx(self, model=None, dataset=None, neurons_idx=None):
        if self.similarity_index is not None:
            if neurons_idx is None:
                return self.similarity_index
            else:
                print 1111
                # print self.similarity_index[0:4, 0:4]
                size_new_sim = len(neurons_idx)
                new_sim = np.zeros((size_new_sim, size_new_sim))
                for i in xrange(size_new_sim):
                    idx1 = neurons_idx[i]
                    for j in xrange(size_new_sim):
                        new_sim[i, j] = self.similarity_index[idx1, neurons_idx[j]]


                return new_sim
        else:
            if self.filters is None:
                print 'Error message, in layer:', self.layer_id, ', No filters.'
                return None
            else:
                size = len(self.filters)
                self.similarity_index = np.zeros((size, size))

                idx_a = np.arange(size)
                print idx_a

                for a, b in permutations(idx_a, 2):
                    sim_idx = get_similarity_index(self.filters[a], self.filters[b], a, b,
                                                   model, self.layer_id, dataset)
                    self.similarity_index[a][b] = sim_idx

                return self.similarity_index

                # for i in xrange(size-1):
                #     for j in xrange(i+1, size):
                #         sim_idx = get_similarity_index(self.filters[i], self.filters[j], i, j,
                #                                        model, self.layer_id, dataset)
                #         self.similarity_index[i][j] = sim_idx
                #
                # return self.similarity_index

    def mapping_rf(self, model, w, h):

        self.receptive_field_map = np.zeros(shape=(w, h),
                                            dtype=[('x1', 'i4'), ('x2', 'i4'), ('y1', 'i4'), ('y2', 'i4')])

        for i in xrange(w):
            for j in xrange(h):
                ri, rf, ci, cf = get_image_receptive_field(i, j, model, self.layer_id)
                self.receptive_field_map[i, j] = (ri, rf+1, ci, cf+1)

        # calculate the size of receptive field
        if self.receptive_field_size is None:
            r = int(w/2)
            c = int(h/2)
            ri, rf, ci, cf = self.receptive_field_map[r, c]
            height = rf - ri
            width = cf - ci
            self.receptive_field_size = (width, height)


    def get_location_from_rf(self, location):
        row, col = location

        h, w = self.receptive_field_map.shape

        ri, rf, ci, cf = self.receptive_field_map[0, 0]
        ri2, rf2, ci2, cf2 = self.receptive_field_map[1, 1]
        stride_r = rf2 - rf
        stride_c = cf2 - cf

        return row/stride_r, col/stride_c

    # decomposition
    def decomposition_image(self, model, img):

        # name_img = self.filters[0].get_images_id()[0]
        #
        #
        # image = img.load_images([name_img])

        max_act = []
        for f in self.filters:
            max_act.append(f.get_activations()[0])

        activations = get_activations(model, img, print_shape_only=True, layer_name=self.layer_id)

        activations = activations[0]
        _, w, h, c = activations.shape

        hc_activations = np.zeros((w, h, c))
        hc_idx = np.zeros((w, h, c))

        for i in xrange(c):
            activations[0, :, :, i] = activations[0, :, :, i] / max_act[i]

        for i in xrange(w):
            for j in xrange(h):
                tmp = activations[0, i, j, :]
                idx = np.argsort(tmp)
                idx = idx[::-1]
                hc_activations[i, j, :] = tmp[idx]
                hc_idx[i, j, :] = idx

        # print activations[0, 0, 0, :]
        # print hc_activations[0, 0, :]
        # print hc_idx[0, 0, :]
        return hc_activations, hc_idx


    def decomposition_nf(self, neuron_id, target_layer, model, dataset):

        if self.filters[neuron_id].get_activations()[0] == 0.0:
            # A neuron feature within a neuron with no activations
            # can't be decomposed
            return None

        neuron_images = self.filters[neuron_id].get_images_id()
        neuron_locations = self.filters[neuron_id].get_locations()
        norm_activations = self.filters[neuron_id].get_norm_activations()

        neuron_images = dataset.load_images(neuron_images)

        for i in xrange(len(neuron_images)):
            loc = neuron_locations[i]
            row_ini, row_fin, col_ini, col_fin = self.receptive_field_map[loc[0], loc[1]]
            patch = neuron_images[i]
            patch = patch[row_ini:row_fin, col_ini:col_fin]

            r, c, k = patch.shape
            img_size = dataset.target_size
            new_image = np.zeros((img_size[0], img_size[1], k))
            new_image[0:r, 0:c, :] = patch

            neuron_images[i] = new_image



        max_act = []
        for f in target_layer.get_filters():
            max_act.append(f.get_activations()[0])

        activations = get_activations(model, neuron_images,
                                      print_shape_only=True,
                                      layer_name=target_layer.get_layer_id())



        activations = activations[0]
        n_patches, w, h, k = activations.shape


        # print max_act

        for i in xrange(n_patches):
            tmp = activations[i]
            for j in xrange(k):
                if max_act[j] != 0.0:
                    n_act = tmp[:, :, j] / max_act[j]
                    np.place(n_act, n_act > 1, 1)
                    tmp[:, :, j] = n_act
            tmp = tmp*norm_activations[i]
            activations[i] = tmp

        mean_activations = np.zeros((w, h, k))
        for i in xrange(n_patches):
            a = activations[i]
            mean_activations += a
        mean_activations = mean_activations / n_patches

        hc_activations = np.zeros((w, h, k))
        hc_idx = np.zeros((w, h, k))

        for i in xrange(w):
            for j in xrange(h):
                tmp = mean_activations[i, j, :]
                idx = np.argsort(tmp)
                idx = idx[::-1]
                hc_activations[i, j, :] = tmp[idx]
                hc_idx[i, j, :] = idx

        return hc_activations, hc_idx

    def similar_neurons(self, neuron_idx, inf_thr=0.0, sup_thr=1.0):

        sim_idx = self.similarity_index

        res_neurons = []
        idx_values = []
        n_sim = sim_idx[neuron_idx, :]

        for i in xrange(len(n_sim)):
            idx = n_sim[i]
            if inf_thr <= idx <= sup_thr:
                res_neurons.append(self.filters[i])
                idx_values.append(idx)

        # convert lists to numpy arrays and sort
        res_neurons = np.array(res_neurons)
        idx_values = np.array(idx_values)
        sorted_idx = np.argsort(idx_values)
        sorted_idx = sorted_idx[::-1]
        res_neurons = res_neurons[sorted_idx]
        idx_values = idx_values[sorted_idx]

        return res_neurons, idx_values


