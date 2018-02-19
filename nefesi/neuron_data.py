
import numpy as np
from PIL import ImageOps

from class_index import get_class_selectivity_idx
from color_index import get_color_selectivity_index
from orientation_index import get_orientation_index
from symmetry_index import get_symmetry_index
from neuron_feature import get_crop_image


class NeuronData(object):
    def __init__(self, max_activations, batch_size):
        self.max_activations = max_activations
        self.batch_size = batch_size
        self.norm_activations = None

        self.activations = np.ndarray(shape=(self.max_activations + self.batch_size))
        self.images_id = np.ndarray(shape=self.max_activations + self.batch_size, dtype='a150')
        self.xy_locations = np.ndarray(shape=self.max_activations + self.batch_size, dtype=[('x', 'i4'), ('y', 'i4')])

        self.selectivity_idx = dict()

        self.neuron_feature = None
        self._index = 0

    def add_activation(self, activation, image_id, xy_location):

        self.activations[self._index] = activation
        self.images_id[self._index] = image_id
        self.xy_locations[self._index] = xy_location
        self._index += 1
        if self._index >= self.max_activations+self.batch_size:
            self.sort()
            self._index = self.max_activations

    def sort(self):

        idx = np.argsort(self.activations)
        idx = idx[::-1]

        self.activations = self.activations[idx]
        self.images_id = self.images_id[idx]
        self.xy_locations = self.xy_locations[idx]

    def normalize_activations(self):
        max_activation = max(self.activations)
        if max_activation == 0:
            return -1
        self.norm_activations = self.activations/abs(max_activation)

    def set_max_activations(self):
        self.activations = self.activations[:self.max_activations]
        self.images_id = self.images_id[:self.max_activations]
        self.xy_locations = self.xy_locations[:self.max_activations]

    def set_nf(self, nf):
        self.neuron_feature = nf

    def get_activations(self):
        return self.activations

    def get_images_id(self):
        return self.images_id

    def get_locations(self):
        return self.xy_locations

    def get_norm_activations(self):
        return self.norm_activations

    def get_neuron_feature(self):
        return self.neuron_feature

    def get_patches(self, network_data, layer_data):
        patches = []
        image_dataset = network_data.dataset
        receptive_field = layer_data.receptive_field_map
        rf_size = layer_data.receptive_field_size

        for i in xrange(self.max_activations):
            img = self.images_id[i]
            loc = self.xy_locations[i]
            crop_pos = receptive_field[loc[0], loc[1]]
            p = image_dataset.get_patch(img, crop_pos)
            if rf_size != p.size:
                w, h = p.size

                ri, rf, ci, cf = crop_pos

                bl, bu, br, bd = (0, 0, 0, 0)

                if rf_size[0] != w:
                    if ci == 0:
                        bl = rf_size[0] - w
                    else:
                        bl = rf_size[0] - w
                if rf_size[1] != h:
                    if ri == 0:
                        bu = rf_size[1] - h
                    else:
                        bd = rf_size[1] - h

                p = ImageOps.expand(p, (bl, bu, br, bd))

                # p.show()
            patches.append(p)

        return patches

    def print_params(self):
        for i in xrange(len(self.activations)):
            print(i, self.images_id[i], self.activations[i], self.xy_locations[i])

    # selectivity indexes
    def color_selectivity_idx(self, model, layer, filter_idx, dataset):
        color_idx = self.selectivity_idx.get('color')
        if color_idx is not None:
            return color_idx

        color_idx = get_color_selectivity_index(self, model, layer, filter_idx, dataset)
        self.selectivity_idx['color'] = color_idx
        return color_idx


    def orientation_selectivity_idx(self, model, layer, filter_idx, dataset, degrees=None, n_rotations=None):
        orientation_idx = self.selectivity_idx.get('orientation')
        if orientation_idx is not None:
            return orientation_idx

        orientation_idx = get_orientation_index(self, model, layer, filter_idx, dataset, degrees, n_rotations)
        self.selectivity_idx['orientation'] = orientation_idx
        return orientation_idx

    def symmetry_selectivity_idx(self, model, layer, filter_idx, dataset):
        symmetry_idx = self.selectivity_idx.get('symmetry')
        if symmetry_idx is not None:
            return symmetry_idx

        symmetry_idx = get_symmetry_index(self, model, layer, filter_idx, dataset)
        self.selectivity_idx['symmetry'] = symmetry_idx
        return symmetry_idx

    def class_selectivity_idx(self, labels):
        class_idx = self.selectivity_idx.get('class')
        if class_idx is not None:
            return class_idx

        class_idx = get_class_selectivity_idx(self, labels)
        self.selectivity_idx['class'] = class_idx
        return class_idx


