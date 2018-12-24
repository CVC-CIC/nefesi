import numpy as np

import math
from .read_activations import get_sorted_activations, get_activations
from .neuron_feature import compute_nf, get_each_point_receptive_field,find_layer_idx
from .similarity_index import get_row_of_similarity_index
from .symmetry_index import SYMMETRY_AXES

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

    def sort_neuron_data(self):
        for neuron in self.neurons_data:
            neuron.sortResults()

    def evaluate_activations(self, file_names, images, model, num_max_activations, batch_size,batches_to_buffer = 20):
        self.neurons_data = get_sorted_activations(file_names, images, model,
                                                   self.layer_id, self.neurons_data,
                                                   num_max_activations, batch_size,batches_to_buffer=batches_to_buffer)

    def build_neuron_feature(self, network_data):
        compute_nf(network_data, self)

    def remove_selectivity_idx(self, idx):
        """Removes de idx selectivity index from the neurons of the layer.


        :return: none.
        """
        for n in self.neurons_data:
            n.remove_selectivity_idx(idx)

    def selectivity_idx(self, model, index_name, dataset,
                        labels=None, thr_class_idx=1., thr_pc=0.1,degrees_orientation_idx = 15, verbose=True,
                        network_data=None):
        """Returns the selectivity index value for the index in `index_name`.

        :param model: The `keras.models.Model` instance.
        :param index_name: String, name of the index.
        :param dataset: The `nefesi.util.image.ImageDataset` instance.
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

        if index_name.lower() == 'color':
            sel_idx = np.zeros(len(self.neurons_data), dtype=np.float)
            for i in range(len(self.neurons_data)):
                if verbose:
                    print(self.layer_id+": "+str(i)+"/"+str(len(self.neurons_data)))
                sel_idx[i] = self.neurons_data[i].color_selectivity_idx(model, self, dataset)
        elif index_name.lower() == 'orientation':
            #Size is (number of neurons, number of rotations with not 0 + mean)
            sel_idx = np.zeros((len(self.neurons_data), int(math.ceil(360/degrees_orientation_idx))), dtype=np.float)
            for i in range(len(self.neurons_data)):
                if verbose:
                    print(self.layer_id+": "+str(i)+"/"+str(len(self.neurons_data)))
                sel_idx[i,:-1] = self.neurons_data[i].orientation_selectivity_idx(model, self, dataset,
                                                                        degrees_to_rotate=degrees_orientation_idx)
            sel_idx[:, -1] = np.mean(sel_idx[:,:-1],axis=1)
        elif index_name.lower() == 'symmetry':
            #array of size (len(self.neurons_data) x 5), 5 is the size of [0 deg., 45 deg., 90 deg., 135 deg., mean]
            sel_idx = np.zeros((len(self.neurons_data), len(SYMMETRY_AXES)+1), dtype=np.float)
            for i in range(len(self.neurons_data)):
                if verbose:
                    print(self.layer_id+": "+str(i)+"/"+str(len(self.neurons_data)))
                sel_idx[i,:-1] = self.neurons_data[i].symmetry_selectivity_idx(model, self, dataset)
            #makes the last columns as mean of each neuron. Makes out of function symmetry_selectivity_idx() for efficiency
            sel_idx[:,-1] = np.mean(sel_idx[:,:-1],axis=1)
        elif index_name.lower() == 'class':
            #array that contains in each a tuple (HumanReadableLabelName(max 64 characters str), selectivityIndex)
            sel_idx = np.zeros(len(self.neurons_data), dtype=np.dtype([('label','U64'), ('value',np.float)]))
            for i in range(len(self.neurons_data)):
                if verbose:
                    print(self.layer_id+": "+str(i)+"/"+str(len(self.neurons_data)))
                sel_idx[i] = self.neurons_data[i].class_selectivity_idx(labels, thr_class_idx)
        elif index_name.lower() == 'concept':
            sel_idx = np.zeros(len(self.neurons_data), dtype=np.object)
            for i in range(len(self.neurons_data)):
                if verbose:
                    print(self.layer_id + ": " + str(i) + "/" + str(len(self.neurons_data)))
                sel_idx[i] = self.neurons_data[i].concept_selectivity_idx(network_data=network_data, layer_data=self,labels=labels)
        elif index_name.lower() == 'population code':
            sel_idx = np.zeros(len(self.neurons_data), dtype=np.int)
            for i in range(len(self.neurons_data)):
                if verbose:
                    print(self.layer_id+": "+str(i)+"/"+str(len(self.neurons_data)))
                sel_idx[i] = self.neurons_data[i].population_code_idx(labels, thr_pc)
        else:
            raise ValueError("The 'index_name' argument should be one "
                             "of theses: "+str(network_data.indexs_accepted))
        return sel_idx

    def get_all_index_of_a_neuron(self, network_data, neuron_idx, orientation_degrees=90, thr_class_idx=1., thr_pc=0.1):
        assert(neuron_idx >=0 and neuron_idx<len(self.neurons_data))
        model = network_data.model
        dataset = network_data.dataset
        neuron = self.neurons_data[neuron_idx]
        index = dict()
        index['color'] = neuron.color_selectivity_idx(model, self, dataset)
        orientation = np.zeros(int(math.ceil(360/orientation_degrees)), dtype=np.float)
        orientation[:-1] = neuron.orientation_selectivity_idx(model, self, dataset,
                                                         degrees_to_rotate=orientation_degrees)
        orientation[-1] = np.mean(orientation[:-1])
        index['orientation'] = orientation
        symmetry = np.zeros(len(SYMMETRY_AXES)+1, dtype=np.float)
        symmetry[:-1] = neuron.symmetry_selectivity_idx(model, self, dataset)
        symmetry[-1] = np.mean(symmetry[:-1])
        index['symmetry'] = symmetry
        index['population code'] = neuron.population_code_idx(network_data.default_labels_dict, thr_pc)
        index['class'] = neuron.class_selectivity_idx(network_data.default_labels_dict, thr_class_idx)
        if network_data.addmits_concept_selectivity():
            index['concept'] = neuron.concept_selectivity_idx(layer_data=self, network_data=network_data,
                                                              labels=network_data.default_labels_dict)
        return index



    def get_similarity_idx(self, model=None, dataset=None, neurons_idx=None, verbose = True):
        """Returns the similarity index matrix for this layer.
        If `neurons_idx` is not None, returns a subset of the similarity
        matrix where `neurons_idx` is the neuron index of the neurons returned
        within that subset.

        :param model: The `keras.models.Model` instance.
        :param dataset: The `nefesi.util.image.ImageDataset` instance.
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
                for i in range(size_new_sim):
                    idx1 = neurons_idx[i]
                    for j in range(size_new_sim):
                        new_sim[i, j] = self.similarity_index[idx1, neurons_idx[j]]
                return new_sim
        else:
            size = len(self.neurons_data)
            self.similarity_index = np.zeros((size, size))

            idx = range(size)
            max_activations = np.zeros(len(self.neurons_data))
            norm_activations_sum= np.zeros(len(self.neurons_data))
            for i in range(len(max_activations)):
                max_activations[i] = self.neurons_data[i].activations[0]
                norm_activations_sum[i] = sum(self.neurons_data[i].norm_activations)
            for i in idx:

                sim_idx = get_row_of_similarity_index(self.neurons_data[i], max_activations,norm_activations_sum,
                                               model, self.layer_id, dataset)
                self.similarity_index[:,i] = sim_idx
                if verbose:
                    print("Similarity "+self.layer_id+' '+str(i)+'/'+str(size))
            return self.similarity_index

    def mapping_rf(self, model):
        """Maps each position in the map activation with the corresponding
        window from the input image (receptive field window).
        Also calculates the size of this receptive field.

        :param model: The `keras.models.Model` instance.
        :raise ValueError: If this layer is apparently non convolutional
        """
        layer_idx = find_layer_idx(model, self.layer_id)
        if len(model.layers[layer_idx].output_shape) != 4:
            return

        if self.receptive_field_map is None:
            self.receptive_field_map = get_each_point_receptive_field(model, self.layer_id)

        # calculate the size of receptive field
        if self.receptive_field_size is None:
            if len(model.layers[layer_idx].output_shape) == 4:
                _, w, h, _ = model.layers[layer_idx].output_shape
            else:
                raise ValueError("You're trying to get the receptive field of a NON Convolutional layer? --> "+self.layer_id)
            row = int(w // 2)
            col = int(h // 2)
            row_ini, row_fin, col_ini, col_fin = self.receptive_field_map[row, col]
            height = row_fin - row_ini
            width = col_fin - col_ini
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
            by the `nefesi.util.image.ImageDataset` instance.

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
        activations = get_activations(model, img,  layer_name=self.layer_id)

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
        for i in range(c):
            activations[0, :, :, i] = activations[0, :, :, i] / max_act[i]

        # sort the activations for each w, h position in map activation
        # in the channel dimension
        for i in range(w):
            for j in range(h):
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
        :param dataset: The `nefesi.util.image.ImageDataset` instance.

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
        for i in range(len(neuron_images)):
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
        for i in range(n_patches):
            tmp = activations[i]
            for j in range(k):
                if max_act[j] != 0.0:
                    n_act = tmp[:, :, j] / max_act[j]
                    np.place(n_act, n_act > 1, 1)
                    tmp[:, :, j] = n_act
            tmp = tmp * norm_activations[i]
            activations[i] = tmp

        # for each map activation from each patch average them
        mean_activations = np.zeros((w, h, k))
        for i in range(n_patches):
            a = activations[i]
            mean_activations += a
        mean_activations = mean_activations / n_patches

        hc_activations = np.zeros((w, h, k))
        hc_idx = np.zeros((w, h, k))

        # sort the activations for each neuron
        for i in range(w):
            for j in range(h):
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
        for i in range(len(n_sim)):
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

    def get_index_calculated_keys(self):
        keys = set()
        for neuron in self.neurons_data:
            keys |= set(neuron.get_keys_of_indexs())
        return keys

    def erase_index(self, index_to_erase):
        for neuron in self.neurons_data:
            neuron.remove_selectivity_idx(idx=index_to_erase)

    def is_not_calculated(self, key):
        for neuron in self.neurons_data:
            if key not in neuron.get_keys_of_indexs():
                return True
        return False