import numpy as np
from PIL import ImageOps

from class_index import get_class_selectivity_idx, get_population_code_idx
from color_index import get_color_selectivity_index
from orientation_index import get_orientation_index
from symmetry_index import get_symmetry_index


class NeuronData(object):
    """This class contains all the results related with a neuron (filter) already
    evaluated, including:
    - The N-top activation values for this neuron (normalized and unnormalized).
    - The selectivity indexes for this neuron.
    - The neuron feature.

    Arguments:
        max_activations: Integer, number of maximum activations stored.
        batch_size: Integer, size of batch.

    Attributes:
        activations: Numpy array of floats, activation values
        images_id: Numpy array of strings, name of the images that provokes
            the maximum activations.
        xy_locations: Numpy array of integer tuples, location of the activation
            in the map activation.
        norm_activations: Numpy array of floats, normalized activation
            values.
        selectivity_idx: Dictionary,
            keys: index name,
            values: index value.
            (List of selectivity indexes implemented: "color", "orientation",
            "symmetry", "class", "population code").

    Properties:
        neuron_feature: PIL image instance.
    """

    def __init__(self, max_activations, batch_size):
        self._max_activations = max_activations
        self._batch_size = batch_size

        self.activations = np.ndarray(shape=(self._max_activations + self._batch_size))
        self.images_id = np.ndarray(shape=self._max_activations + self._batch_size, dtype='a150')
        self.xy_locations = np.ndarray(shape=self._max_activations + self._batch_size, dtype=[('x', 'i4'), ('y', 'i4')])
        self.norm_activations = None

        self.selectivity_idx = dict()
        self._neuron_feature = None

        # index used for ordering activations.
        self._index = 0

    def add_activation(self, activation, image_id, xy_location):
        """Set the information of one activation. When the assigned
         activations reach a certain size, they are ordered.

        :param activation: Float, activation value
        :param image_id: String, image name
        :param xy_location: Tuple of integers, location of the activation
            in the map activation.
        """
        self.activations[self._index] = activation
        self.images_id[self._index] = image_id
        self.xy_locations[self._index] = xy_location
        self._index += 1
        if self._index >= self._max_activations + self._batch_size:
            self.sort()
            self._index = self._max_activations

    def sort(self):
        """Sorting method of activations. Attributes `images_id`
         and `xy_locations` are ordered according to `activations`.
        """
        idx = np.argsort(self.activations)
        idx = idx[::-1]

        self.activations = self.activations[idx]
        self.images_id = self.images_id[idx]
        self.xy_locations = self.xy_locations[idx]

    def _normalize_activations(self):
        """Normalize the activations inside `activations`.
        """
        max_activation = max(self.activations)
        if max_activation == 0:
            return -1
        self.norm_activations = self.activations / abs(max_activation)

    def set_max_activations(self):
        self.activations = self.activations[:self._max_activations]
        self.images_id = self.images_id[:self._max_activations]
        self.xy_locations = self.xy_locations[:self._max_activations]
        self._normalize_activations()

    @property
    def neuron_feature(self):
        return self._neuron_feature

    @neuron_feature.setter
    def neuron_feature(self, neuron_feature):
        self._neuron_feature = neuron_feature

    def get_patches(self, network_data, layer_data):
        """Returns the patches (receptive fields) from images in
        `images_id` for this neuron.

        :param network_data: The `nefesi.network_data.NetworkData` instance.
        :param layer_data: The `nefesi.layer_data.LayerData` instance.

        :return: List of PIL image instances.
        """
        patches = []
        image_dataset = network_data.dataset
        receptive_field = layer_data.receptive_field_map
        rf_size = layer_data.receptive_field_size

        for i in xrange(self._max_activations):
            img = self.images_id[i]
            loc = self.xy_locations[i]

            # get the location of the receptive field
            crop_pos = receptive_field[loc[0], loc[1]]
            # crop the origin image with previous location
            p = image_dataset.get_patch(img, crop_pos)

            # add a black padding to a patch that not match with the receptive
            # field size.
            # This is due that some receptive fields has padding
            # that come of the network architecture.
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
            patches.append(p)
        return patches

    def print_params(self):
        """Returns a string with some information about this neuron.
        Index of neuron, name of the image, activation value,
        activation location in map activation, normalized activation.
        """
        if self.norm_activations is None:
            print("Neuron with no activations.")
        else:
            for i in xrange(len(self.activations)):
                print(i, self.images_id[i],
                      self.activations[i],
                      self.xy_locations[i],
                      self.norm_activations[i])

    def color_selectivity_idx(self, model, layer_data, dataset):
        """Returns the color selectivity index for this neuron.

        :param model: The `keras.models.Model` instance.
        :param layer_data: The `nefesi.layer_data.LayerData` instance.
        :param dataset: The `nefesi.utils.image.ImageDataset` instance.

        :return: Float, value of color selectivity index.
        """
        color_idx = self.selectivity_idx.get('color')
        if color_idx is not None:
            return color_idx

        color_idx = get_color_selectivity_index(self, model,
                                                layer_data, dataset)
        self.selectivity_idx['color'] = color_idx
        return color_idx

    def orientation_selectivity_idx(self, model, layer_data, dataset):
        """Returns the orientation selectivity index for this neuron.

        :param model: The `keras.models.Model` instance.
        :param layer_data: The `nefesi.layer_data.LayerData` instance.
        :param dataset: The `nefesi.utils.image.ImageDataset` instance.

        :return: List of floats, values of orientation selectivity index.
        """
        orientation_idx = self.selectivity_idx.get('orientation')
        if orientation_idx is not None:
            return orientation_idx

        orientation_idx = get_orientation_index(self, model,
                                                layer_data, dataset)
        self.selectivity_idx['orientation'] = orientation_idx
        return orientation_idx

    def symmetry_selectivity_idx(self, model, layer_data, dataset):
        """Returns the symmetry selectivity index for this neuron.

        :param model: The `keras.models.Model` instance.
        :param layer_data: The `nefesi.layer_data.LayerData` instance.
        :param dataset: The `nefesi.utils.image.ImageDataset` instance.

        :return: List of floats, values of symmetry selectivity index.
        """
        symmetry_idx = self.selectivity_idx.get('symmetry')
        if symmetry_idx is not None:
            return symmetry_idx

        symmetry_idx = get_symmetry_index(self, model, layer_data, dataset)
        self.selectivity_idx['symmetry'] = symmetry_idx
        return symmetry_idx

    def class_selectivity_idx(self, labels=None, threshold=1.):
        """Returns the class selectivity index for this neuron.

        :param labels: Dictionary, key: name class, value: label class.
            This argument is needed for calculate the class index.
        :param threshold: Float, required for calculate the class index.

        :return: Float, between 0.1 and 1.0.

        :raise:
            TypeError: If `labels` is None or not a dictionary.
        """
        class_idx = self.selectivity_idx.get('class')
        if class_idx is not None:
            return class_idx

        if labels is None or type(labels) is not dict():
            raise TypeError("The `labels` argument should be "
                            "a dictionary")

        class_idx = get_class_selectivity_idx(self, labels, threshold)
        self.selectivity_idx['class'] = class_idx
        return class_idx

    def population_code_idx(self, labels=None, threshold=0.1):
        """Returns the population code index for this neuron.

        :param labels: Dictionary, key: name class, value: label class.
            This argument is needed for calculate the the population
            code index.
        :param threshold: Float, between 0.1 and 1.0.

        :return: Float, value of population code index.

        :raise:
            TypeError: If `labels` is None or not a dictionary.
        """
        population_code_idx = self.selectivity_idx.get('population code')
        if population_code_idx is not None:
            return population_code_idx

        if labels is None or type(labels) is not dict():
            raise TypeError("The `labels` argument should be "
                            "a dictionary")

        population_code_idx = get_population_code_idx(self, labels, threshold)
        self.selectivity_idx['population code'] = population_code_idx
        return population_code_idx
