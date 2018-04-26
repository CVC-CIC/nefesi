import numpy as np
from itertools import permutations

from read_activations import get_sorted_activations, get_activations
from neuron_feature import compute_nf, get_image_receptive_field
from similarity_index import get_similarity_index


class LayerData(object):
    """This class contains all the information related with the
    layers already evaluated.

    Arguments:
        layer_id: String, name of the layer (This name is the same
            inside of `keras.models.Model` instance)

    Attributes:
        neurons_data: List of `nefesi.neuron_data.NeuronData` instances.
        similarity_index: Non-symmetric matrix containing the result of
            similarity index for each neuron in this layer. When this index is
            calculated, the size of the matrix is len(filters) x len(filters).
        receptive_field_map: Matrix of integer tuples with size equal
            to map activation shape of this layer. Each position i, j from
            the matrix contains a tuple with four values: row_ini, row_fin,
            col_ini, col_fin. This values represents the window of receptive
            field from the input image that provokes the activation there is
            in the location i, j of the map activation.
        receptive_field_size: Tuple of two integers. Size of receptive field
            of the input image in this layer.
    """

    def __init__(self, layer_name):
        self.layer_id = layer_name
        self.neurons_data = None
        self.similarity_index = None
        self.receptive_field_map = None
        self.receptive_field_size = None

    def set_max_activations(self):
        for f in self.neurons_data:
            f.set_max_activations()

    def evaluate_activations(self, file_names, images, model, num_max_activations, batch_size):
        self.neurons_data = get_sorted_activations(file_names, images, model,
                                                   self.layer_id, self.neurons_data,
                                                   num_max_activations, batch_size)

    def build_neuron_feature(self, network_data):
        compute_nf(network_data, self, self.neurons_data)

    def selectivity_idx(self, model, index_name, dataset,
                        labels=None, thr_class_idx=1., thr_pc=0.1):
        """Returns the selectivity index value for the index in `index_name`.

        :param model: The `keras.models.Model` instance.
        :param index_name: String, name of the index.
        :param dataset: The `nefesi.utils.image.ImageDataset` instance.
        :param labels: Dictionary, key: name class, value: label.
            This argument is needed for calculate the class and the population
            code index.
        :param thr_class_idx: Float between 0.0 and 1.0, threshold applied in
            class selectivity index.
        :param thr_pc: Float between 0.0 and 1.0, threshold applied in
            population code index.

        :return: List of floats. The index values for each neuron in this layer.

        :raises:
            ValueError: If `index_name` is not one of theses: "color",
            "orientation", "symmetry", "class" or "population code".
        """
        sel_idx = []
        for f in self.neurons_data:
            if index_name == 'color':
                res = f.color_selectivity_idx(model, self, dataset)
                sel_idx.append(res)
            elif index_name == 'orientation':
                res = f.orientation_selectivity_idx(model, self, dataset)
                sel_idx.append(res)
            elif index_name == 'symmetry':
                res = f.symmetry_selectivity_idx(model, self, dataset)
                sel_idx.append(res)
            elif index_name == 'class':
                res = f.class_selectivity_idx(labels, thr_class_idx)
                sel_idx.append(res)
            elif index_name == 'population code':
                res = f.population_code_idx(labels, thr_pc)
                sel_idx.append(res)
            else:
                raise ValueError("The `index_name` argument should be one "
                                 "of theses: color, orientation, symmetry, "
                                 "class or population code.")
        return sel_idx

    def get_similarity_idx(self, model=None, dataset=None, neurons_idx=None):
        """Returns the similarity index matrix for this layer.
        If `neurons_idx` is not None, returns a subset of the similarity
        matrix where `neurons_idx` is the neuron index of the neurons returned
        within that subset.

        :param model: The `keras.models.Model` instance.
        :param dataset: The `nefesi.utils.image.ImageDataset` instance.
        :param neurons_idx: List of integer. Neuron indexes in the attribute
            class `filters`.

        :return: Non-symmetric matrix of floats. Each position i, j in the matrix
            corresponds to the distance between the neuron with index i and neuron
            with index j, in the attribute class `filters`.
        """
        if self.similarity_index is not None:
            if neurons_idx is None:
                return self.similarity_index
            else:
                size_new_sim = len(neurons_idx)
                new_sim = np.zeros((size_new_sim, size_new_sim))
                for i in xrange(size_new_sim):
                    idx1 = neurons_idx[i]
                    for j in xrange(size_new_sim):
                        new_sim[i, j] = self.similarity_index[idx1, neurons_idx[j]]
                return new_sim
        else:
            size = len(self.neurons_data)
            self.similarity_index = np.zeros((size, size))

            idx_a = np.arange(size)
            print idx_a

            for a, b in permutations(idx_a, 2):
                sim_idx = get_similarity_index(self.neurons_data[a], self.neurons_data[b], a,
                                               model, self.layer_id, dataset)
                self.similarity_index[a][b] = sim_idx
            return self.similarity_index

    def mapping_rf(self, model, w, h):
        """Maps each position in the map activation with the corresponding
        window from the input image (receptive field window).
        Also calculates the size of this receptive field.

        :param model: The `keras.models.Model` instance.
        :param w: Integer, row position in the map activation.
        :param h: Integer, column position in the map activation.
        """
        if self.receptive_field_map is None:
            self.receptive_field_map = np.zeros(shape=(w, h),
                                                dtype=[('x1', 'i4'),
                                                       ('x2', 'i4'),
                                                       ('y1', 'i4'),
                                                       ('y2', 'i4')])
            for i in xrange(w):
                for j in xrange(h):
                    ri, rf, ci, cf = get_image_receptive_field(i, j, model, self.layer_id)
                    # we have to add 1 in row_fin and col_fin due to behaviour
                    # of Numpy arrays.
                    self.receptive_field_map[i, j] = (ri, rf + 1, ci, cf + 1)

        # calculate the size of receptive field
        if self.receptive_field_size is None:
            r = int(w / 2)
            c = int(h / 2)
            ri, rf, ci, cf = self.receptive_field_map[r, c]
            height = rf - ri
            width = cf - ci
            self.receptive_field_size = (width, height)

    def get_location_from_rf(self, location):
        """Given a pixel of an image (x, y), returns a location in the map
        activation that corresponds to receptive field with the center more nearest
        to the pixel position (x, y).

        :param location: Tuple of integers, pixel location from the image.
        :return: Integer tuple, a location of the map activation.
        """
        row, col = location
        ri, rf, ci, cf = self.receptive_field_map[0, 0]
        ri2, rf2, ci2, cf2 = self.receptive_field_map[1, 1]
        stride_r = rf2 - rf
        stride_c = cf2 - cf

        return row / stride_r, col / stride_c

    def decomposition_image(self, model, img):
        """Calculates the decomposition of an image in this layer
        and returns the maximum activation values and the neurons that provoke
        them.

        :param model: The `keras.models.Model` instance.
        :param img: Numpy array. This image should be an image already preprocessed
            by the `nefesi.utils.image.ImageDataset` instance.

        :return: Two numpy array of shape(w, h, k). Where w and h is the size of
            the map activation in this layer, and k is the number of neurons in this
            layer.

            The first array, contains the activation values, sorted by maximum in
            the k dimension for each w, h position.

            The second array, contains the neurons index that provoke them.
            Each position (w, h, k) from this array contains the index neuron that
            corresponds to the activation value in the first array with the same
            position (w, h, k).
        """
        max_act = []
        for f in self.neurons_data:
            max_act.append(f.activations[0])

        # get the activations of image in this layer.
        activations = get_activations(model, img, print_shape_only=True, layer_name=self.layer_id)

        activations = activations[0]

        # get the activations shape, where:
        #  _ = number of images = 1,
        # w, h = size of map activation,
        # c = number of channels
        _, w, h, c = activations.shape

        hc_activations = np.zeros((w, h, c))
        hc_idx = np.zeros((w, h, c))

        # normalize the activations in each map activation
        # for each channel
        for i in xrange(c):
            activations[0, :, :, i] = activations[0, :, :, i] / max_act[i]

        # sort the activations for each w, h position in map activation
        # in the channel dimension
        for i in xrange(w):
            for j in xrange(h):
                tmp = activations[0, i, j, :]
                idx = np.argsort(tmp)
                idx = idx[::-1]
                hc_activations[i, j, :] = tmp[idx]
                hc_idx[i, j, :] = idx

        return hc_activations, hc_idx

    def decomposition_nf(self, neuron_id, target_layer, model, dataset):
        """Calculates the decomposition of a neuron
         from this layer and returns the maximum activation values and
         the neurons that provoke them.

        :param neuron_id: Integer, index of the neuron with the neuron feature
            to be decomposed.
        :param target_layer: The `nefesi.layer_data.LayerData` instance,
            layer where decompose the neuron.
        :param model: The `keras.models.Model` instance.
        :param dataset: The `nefesi.utils.image.ImageDataset` instance.

        :return: Two numpy array of shape(w, h, k). Where w and h is the size of
            the map activation in this layer, and k is the number of neurons in this
            layer.

            The first array, contains the activation values, sorted by maximum in
            the k dimension for each w, h position.

            The second array, contains the neurons index that provoke them.
            Each position (w, h, k) from this array contains the index neuron that
            corresponds to the activation value in the first array with the same
            position (w, h, k).
        """
        if self.neurons_data[neuron_id].activations[0] == 0.0:
            # A neuron with no activations can't be decomposed
            return None

        neuron_images = self.neurons_data[neuron_id].images_id
        neuron_locations = self.neurons_data[neuron_id].xy_locations
        norm_activations = self.neurons_data[neuron_id].norm_activations

        neuron_images = dataset.load_images(neuron_images)

        # build the patches from the neuron on this layer that we want to decompose
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
        for f in target_layer.neurons_data:
            max_act.append(f.activations[0])

        # get the activations for the patches
        activations = get_activations(model, neuron_images,
                                      print_shape_only=True,
                                      layer_name=target_layer.layer_id)

        activations = activations[0]
        # get the activations shape, where:
        # n_patches = number of patches,
        # w, h = size of map activation,
        # k = number of neurons
        n_patches, w, h, k = activations.shape

        # normalize each map activation from each patch for each neuron and multiply
        # each map activation from each patch with the normalized activation
        # of that patch
        for i in xrange(n_patches):
            tmp = activations[i]
            for j in xrange(k):
                if max_act[j] != 0.0:
                    n_act = tmp[:, :, j] / max_act[j]
                    np.place(n_act, n_act > 1, 1)
                    tmp[:, :, j] = n_act
            tmp = tmp * norm_activations[i]
            activations[i] = tmp

        # for each map activation from each patch average them
        mean_activations = np.zeros((w, h, k))
        for i in xrange(n_patches):
            a = activations[i]
            mean_activations += a
        mean_activations = mean_activations / n_patches

        hc_activations = np.zeros((w, h, k))
        hc_idx = np.zeros((w, h, k))

        # sort the activations for each neuron
        for i in xrange(w):
            for j in xrange(h):
                tmp = mean_activations[i, j, :]
                idx = np.argsort(tmp)
                idx = idx[::-1]
                hc_activations[i, j, :] = tmp[idx]
                hc_idx[i, j, :] = idx

        return hc_activations, hc_idx

    def similar_neurons(self, neuron_idx, inf_thr=0.0, sup_thr=1.0):
        """Given a neuron index, returns a sorted list of the neurons
        with a similarity index between `inf_thr` and `sup_thr`.

        :param neuron_idx: Integer, index of the neuron in the attribute
            class `filters`.
        :param inf_thr: Float.
        :param sup_thr: Float.

        :return: Two lists, list of `nefesi.neuron_data.NeuronData` instances
            and values of similarity index.

        :raise:
            ValueError: If `self.similarity_index` is None
        """
        sim_idx = self.similarity_index
        if sim_idx is None:
            raise ValueError("The similarity index in the layer '{}',"
                             " is not calculated.".format(self.layer_id))

        res_neurons = []
        idx_values = []
        n_sim = sim_idx[neuron_idx, :]

        # get the similarity values between the threshold
        for i in xrange(len(n_sim)):
            idx = n_sim[i]
            if inf_thr <= idx <= sup_thr:
                res_neurons.append(self.neurons_data[i])
                idx_values.append(idx)

        # convert lists to numpy arrays and sort
        res_neurons = np.array(res_neurons)
        idx_values = np.array(idx_values)
        sorted_idx = np.argsort(idx_values)
        sorted_idx = sorted_idx[::-1]
        res_neurons = res_neurons[sorted_idx]
        idx_values = idx_values[sorted_idx]

        return list(res_neurons), list(idx_values)
