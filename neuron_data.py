
import numpy as np

from color_index import get_color_selectivity_index


class NeuronData(object):
    def __init__(self, max_activations, batch_size):
        self.max_activations = max_activations
        self.batch_size = batch_size
        self.norm_activations = None

        self.activations = np.ndarray(shape=(self.max_activations + self.batch_size))
        self.images_id = np.ndarray(shape=self.max_activations + self.batch_size, dtype='a150')
        self.xy_locations = np.ndarray(shape=self.max_activations + self.batch_size, dtype=[('x', 'i4'), ('y', 'i4')])

        # self.activations = []
        # self.images_id = []
        # self.xy_locations = []
        self.selectivity_idx = dict()

        self.neuron_feature = None

        self._index = 0


    def add_activation(self, activation, image_id, xy_location):

        # self.activations.append(activation)
        # self.images_id.append(image_id)
        # self.xy_locations.append(xy_location)
        #
        # if len(self.activations) >= self.max_activations + self.batch_size:
        #     self.sort()

        # if len(self.activations) < self.max_activations + self.batch_size:
        #     tmp_act = np.ndarray(shape=self.max_activations + self.batch_size)
        #     tmp_img = np.ndarray(shape=self.max_activations+self.batch_size, dtype='a150')
        #     tmp_xy = np.ndarray(shape=self.max_activations+self.batch_size, dtype=[('x', 'i4'),('y', 'i4')])
        #
        #     tmp_act[:self.max_activations] = self.activations
        #     tmp_img[:self.max_activations] = self.images_id
        #     tmp_xy[:self.max_activations] = self.xy_locations
        #
        #     self.activations = tmp_act
        #     self.images_id = tmp_img
        #     self.xy_locations = tmp_xy



        self.activations[self._index] = activation
        self.images_id[self._index] = image_id
        self.xy_locations[self._index] = xy_location
        self._index += 1
        if self._index >= self.max_activations+self.batch_size:
            self.sort()
            self._index = self.max_activations





    def sort(self):
        # quickSort(self.activations)
        # # print self.activations
        # self.activations.reverse()

        # tmp_act = np.asarray(self.activations)
        # tmp_images = np.asarray(self.images_id)
        # tmp_xy = np.asarray(self.xy_locations)
        #
        # idx = np.argsort(tmp_act)
        # idx = idx[::-1]
        #
        # tmp_act = tmp_act[idx]
        # tmp_images = tmp_images[idx]
        # tmp_xy = tmp_xy[idx]
        #
        # self.activations = tmp_act[:self.max_activations].tolist()
        # self.images_id = tmp_images[:self.max_activations].tolist()
        # self.xy_locations = tmp_xy[:self.max_activations].tolist()

        idx = np.argsort(self.activations)
        idx = idx[::-1]

        self.activations = self.activations[idx]
        self.images_id = self.images_id[idx]
        self.xy_locations = self.xy_locations[idx]

        # self.activations = self.activations[:self.max_activations]
        # self.images_id = self.images_id[:self.max_activations]
        # self.xy_locations = self.xy_locations[:self.max_activations]

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

    def print_params(self):
        for i in xrange(len(self.activations)):
            print(i, self.images_id[i], self.activations[i], self.xy_locations[i])

    # selectivity indexes
    def color_selectivity_idx(self, model, layer, filter_idx, dataset_path):
        color_idx = self.selectivity_idx.get('color')
        if color_idx is not None:
            return color_idx

        color_idx = get_color_selectivity_index(self, model, layer, filter_idx, dataset_path)
        return color_idx


    def orientation_selectivity_idx(self):
        pass

    def symmetry_selectivity_idx(self):
        pass

    def class_selectivity_idx(self):
        pass


