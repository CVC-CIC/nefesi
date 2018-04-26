import pickle
import time
import numpy as np
import warnings

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from layer_data import LayerData
from utils.image import ImageDataset


class NetworkData(object):
    """This is the main class of nefesi package.

    Creates an instance of NetworkData, the main object containing
    all evaluated data related with a `keras.models.Model` instance.

    Arguments:
        model: The `keras.models.Model` instance.

    Attributes:
        layers_data: List of `nefesi.layer_data.LayerData` instances.

    Mutable-properties:
        save_path: Path of directory where the results will be saved.
        dataset: The `nefesi.utils.image.ImageDataset` instance.
    """

    def __init__(self, model):
        self.model = model
        self.layers_data = []
        self._save_path = None
        self._dataset = None

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    @property
    def save_path(self):
        return self._save_path

    @save_path.setter
    def save_path(self, save_path):
        self._save_path = save_path

    def _build_layers(self, layers):
        for l in layers:
            self.layers_data.append(LayerData(l))

    def eval_network(self, layer_names,
                     directory=None,
                     save_path=None,
                     num_max_activations=100,
                     target_size=(256, 256),
                     batch_size=100,
                     preprocessing_function=None,
                     color_mode='rgb',
                     save_for_layers=True,
                     build_nf=True,
                     file_name=None,
                     verbose=True):
        """Evaluates the layers in `layer_names`, searching for the maximum
        activations for each neuron and build the neuron feature.

        :param layer_names: List of strings (name of the layers that will be
            evaluated).
        :param directory: Path to the directory to read images from.
        :param save_path: Path of directory to write the results.
        :param num_max_activations: Integer, number of maximum activations
            will be saved for each neuron.
        :param target_size: Tuple of integers, dimensions to resize input images to.
        :param batch_size: Integer, size of a batch.
        :param preprocessing_function: Function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument: batch of images (Numpy tensor with rank 4),
             and should output a Numpy tensor with the same shape.
        :param color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        :param save_for_layers: Boolean, if its True, the results will be saved
            for each layer evaluated. The result file will have the name of the layer
            evaluated and the previous ones.
        :param build_nf:  Boolean, if its True, the neuron feature for each neuron
            will be built.
            If layers in `layer_names` are fully connected layers, `build_nf`
            has to be False.
        :param file_name: String, name of the file containing the result data,
            if its None, the file will be stored with the model name.
        :param verbose: Verbosity, False=Silent, True=one line per batch

        :raises
            TypeError: If some value in `layer_names` is not a String.
            ValueError: If some name in `layer_names` doesn't exist in
            the model `self.model`.
        """
        model_layer_names = [layer.name for layer in self.model.layers]
        for layer_n in layer_names:
            if type(layer_n) != str:
                raise TypeError("`layer_names`, should be a list of Strings.")
            if layer_n not in model_layer_names:
                raise ValueError("Wrong layer name: {}.".format(str(layer_n)))

        # Creates the list of `layers_data`
        self._build_layers(layer_names)
        self._save_path = save_path

        times_ex = []

        # Creates an ImageDataset object
        if self.dataset is None:
            self.dataset = ImageDataset(directory, target_size,
                                        preprocessing_function, color_mode)

        if self.dataset.src_dataset is None:
            raise ValueError("The argument `directory` should be a String.")

        for layer in self.layers_data:
            if layer.layer_id in layer_names:
                datagen = ImageDataGenerator()
                data_batch = datagen.flow_from_directory(
                    self.dataset.src_dataset,
                    target_size=self.dataset.target_size,
                    batch_size=batch_size,
                    shuffle=False,
                    color_mode=self.dataset.color_mode
                )

                num_images = data_batch.samples
                n_batches = 0
                idx = data_batch.batch_index

                for i in data_batch:
                    images = i[0]

                    # Apply the preprocessing function to the inputs
                    if self.dataset.preprocessing_function is not None:
                        images = self.dataset.preprocessing_function(images)

                    file_names = data_batch.filenames[idx: idx + data_batch.batch_size]
                    # Search the maximum activations
                    layer.evaluate_activations(file_names, images, self.model, num_max_activations, batch_size)

                    if verbose:
                        print("Layer: {}, Num batch: {},"
                              " Num images processed: {}/{}".format(
                                layer.layer_id,
                                n_batches,
                                idx + data_batch.batch_size,
                                num_images))

                    idx = data_batch.batch_index * data_batch.batch_size
                    n_batches += 1
                    if n_batches >= num_images / data_batch.batch_size:
                        break

                # Set the number of maximum activations stored in each neuron
                layer.set_max_activations()

                if build_nf:
                    # Build the neuron features
                    layer.build_neuron_feature(self)
                if save_for_layers:
                    # Save the results each time we have a evaluated layer
                    self.save_to_disk(layer.layer_id)

        # Save all data
        self.save_to_disk(file_name=file_name)

    def get_layers_name(self):
        """Builds a list with the name of each layer in `layers_data`.

        :return: List of strings
        """
        names = [l.layer_id for l in self.layers_data]
        return names

    def get_selectivity_idx(self, sel_index, layer_name,
                            labels=None, thr_class_idx=1.,
                            thr_pc=0.1):
        """Returns the selectivity indexes in `sel_index` for each layer
        in `layer_name`.

        :param sel_index: List of strings or string, name of the selectivity indexes.
        :param layer_name: List of strings or string, name of the layers.
        :param labels: Dictionary, key: name class, value: label.
            This argument is needed for calculate the class and the population
            code index.
        :param thr_class_idx: Float between 0.0 and 1.0, threshold applied in
            class selectivity index.
        :param thr_pc: Float between 0.0 and 1.0, threshold applied in
            population code index.

        :return: Dictionary,
            keys: name of selectivity index,
            values: list of lists (list per layer containing the index values).

        :raise:
            ValueError: If layer name in `layer_name` doesn't match with any layer
            inside the class property `layers`.
        """
        sel_idx_dict = dict()

        if type(sel_index) is not list:
            sel_index = [sel_index]
        if type(layer_name) is not list:
            layer_name = [layer_name]

        for index_name in sel_index:
            sel_idx_dict[index_name] = []

            for l in layer_name:
                layer = next((layer_data for layer_data in
                              self.layers_data if l in self.get_layers_name()
                              and l == layer_data.layer_id), None)

                if layer is None:
                    raise ValueError("The layer_id '{}' `layer_name` "
                                     "argument, is not valid.".format(l))
                else:
                    sel_idx_dict[index_name].append(layer.selectivity_idx(
                        self.model, index_name, self.dataset, labels=labels,
                        thr_class_idx=thr_class_idx, thr_pc=thr_pc))

        return sel_idx_dict

    def similarity_idx(self, layer_name):
        """Returns the similarity index for each layer in `layer_name`.

        :param layer_name: List of strings or string, name of the layers.

        :return: List of Numpy arrays, each array belows to one layer.

        :raise:
            ValueError: If layer name in `layer_name` doesn't match with any layer
            inside the class property `layers`.
        """
        sim_idx = []

        if type(layer_name) is not list:
            layer_name = [layer_name]

        for l in layer_name:
            layer = next((layer_data for layer_data in
                          self.layers_data if l in self.get_layers_name()
                          and l == layer_data.layer_id), None)
            if layer is None:
                raise ValueError("The layer_id '{}' `layer_name` "
                                 "argument, is not valid.".format(l))
            else:
                sim_idx.append(layer.get_similarity_idx(self.model, self.dataset))
        return sim_idx

    def get_selective_neurons(self, layers_or_neurons, idx1, idx2=None,
                              inf_thr=0.0, sup_thr=1.0):
        """Returns a list of neuron_data objects with their respective
        values of selectivity indexes between a threshold (`inf_thr` and `sup_thr`).
        This function works for one or two indexes (`idx1` and `idx2`).
        If `idx1` or/and `idx2` are "symmetry" or "orientation" the index value
        evaluated will be the global.

        :param layers_or_neurons: List of strings, a string or a list of
            selective neurons.
            List of strings or string, name of the layer.
            List of selective neurons, the output of this function itself.
        :param idx1: String, index name
        :param idx2: String, index name
        :param inf_thr: Float between 0.0 and 1.0, gets the index values
            above this threshold.
        :param sup_thr: Float between 0.0 and 1.0, gets the index values
            below this threshold.

        :return: Dictionary,
            keys: string or tuple of strings, index name,
            values: dictionary,
                keys: string, layer name,
                values: list of tuples with
                `nefesi.neuron_data.NeuronData` instance, index value, index value,
                (the second index value, only if `idx2` is not None).

        :raise:
            TypeError: If `layers_or_neurons` is not a list of layer names,
            a layer name or a dictionary.
        """
        selective_neurons = dict()
        res_idx2 = None

        if type(layers_or_neurons) is list or type(layers_or_neurons) is str:
            layers = layers_or_neurons
            for l in self.layers_data:
                if l.layer_id in layers:
                    res_idx1 = []
                    index_values = l.selectivity_idx(self.model, idx1, self.dataset)
                    if type(index_values[0]) is list or type(index_values[0]) is tuple:
                        n_idx = len(index_values[0])
                        for i in index_values:
                            res_idx1.append(i[n_idx - 1])
                    else:
                        res_idx1 = index_values

                    if idx2 is not None:
                        res_idx2 = []
                        index_values2 = l.selectivity_idx(self.model, idx2, self.dataset)
                        print type(index_values2[0])

                        if type(index_values2[0]) is list or type(index_values2[0]) is tuple:
                            n_idx = len(index_values2[0])
                            for i in index_values2:
                                res_idx2.append(i[n_idx - 1])
                        else:
                            res_idx2 = index_values2

                    selective_neurons[l.layer_id] = []
                    neurons = l.filters
                    for i in xrange(len(neurons)):
                        if inf_thr <= res_idx1[i] <= sup_thr:
                            if res_idx2 is not None:
                                if inf_thr <= res_idx2[i] <= sup_thr:
                                    tmp = (neurons[i], res_idx1[i], res_idx2[i])
                                    selective_neurons[l.layer_id].append(tmp)
                            else:
                                tmp = (neurons[i], res_idx1[i])
                                selective_neurons[l.layer_id].append(tmp)
        elif type(layers_or_neurons) is dict:
            sel_idx = layers_or_neurons.keys()[0]
            values = layers_or_neurons.values()[0]

            for k, v in values.items():

                selective_neurons[k] = []
                for item in v:
                    neuron = item[0]
                    idx_value = neuron.selectivity_idx.get(idx1)

                    if type(idx_value) is list:
                        n_idx = len(idx_value)
                        idx_value = idx_value[n_idx - 1]

                    if inf_thr <= idx_value <= sup_thr:
                        tmp = list(item)
                        tmp.append(idx_value)
                        selective_neurons[k].append(tuple(tmp))

            if type(sel_idx) is tuple:
                tmp_idx = list(sel_idx)
                tmp_idx.append(idx1)
                idx1 = tuple(tmp_idx)
            else:
                idx1 = (sel_idx, idx1)
        else:
            raise TypeError("Parameter 1 should be a list of layers, layer_id or dict")

        if idx2 is not None:
            idx1 = (idx1, idx2)
        res = {idx1: selective_neurons}
        return res

    def get_max_activations(self, layer_id, img_name, location, num_max):
        """ Given a specific layer, an image and a location of the image,
        returns a list with the maximum activations for that pixel in that layer.

        :param layer_id: String or integer index, layer name or index of the
            layer in `layers`. The layer where the image will be decomposed.
        :param img_name: String, image name. This image has to be in the same
            directory set in `self.dataset.src_dataset`.
        :param location: Tuple of integers, pixel location from the image.
        :param num_max: Integer, Max number of activations returned.

        :return: List of floats, activation values,
            List of `nefesi.neuron_data.NeuronData` instances,
            Location of receptive field related with the `location` of pixel.
        """
        layer = None
        if type(layer_id) is int:
            layer = self.layers_data[layer_id]
        if type(layer_id) is str:
            for l in self.layers_data:
                if layer_id == l.layer_id:
                    layer = l

        img = self.dataset.load_images([img_name])
        hc_activations, hc_idx = layer.decomposition_image(self.model, img)
        loc = layer.get_location_from_rf(location)

        activations = hc_activations[loc[0], loc[1], :num_max]
        neuron_idx = hc_idx[loc[0], loc[1], :num_max]

        f = layer.neurons_data
        neurons = []
        for idx in neuron_idx:
            neurons.append(f[int(idx)])

        return list(activations), neurons, layer.receptive_field_map[loc[0], loc[1]]

    def decomposition(self, input_image, target_layer, overlapping=0.0):
        """ Given an image or a neuron feature, returns a list with
        the maximum activations for each location in the image or neuron feature,
        with a certain percent of overlapping.

        :param input_image: List or string.
            List with two elements: layer name, integer index of a neuron (layer
            and neuron where takes the neuron feature we want decompose).
            String, image name.
        :param target_layer: String, layer name (layer where decompose).
        :param overlapping: Float between 0.0 and 1.0, percent of overlapping
            of the activations in the input image or input neuron feature.

        :return: List of floats, activation values.
            List of `nefesi.neuron_data.NeuronData` instances.
            List of integer tuples, activation locations for each
            receptive field in the input image or input neuron feature.
            PIL image instance, in case we decompose a neuron feature,
            in other case, returns None.
        """
        res_nf = None
        if isinstance(input_image, list):  # Decomposition of neuron feature
            src_layer = input_image[0]
            neuron_idx = input_image[1]

            for l in self.layers_data:
                if src_layer == l.layer_id:
                    src_layer = l
                elif target_layer == l.layer_id:
                    target_layer = l

            hc_activations, hc_idx = src_layer.decomposition_nf(
                neuron_idx, target_layer, self.model, self.dataset)

            neuron_data = src_layer.neurons_data[neuron_idx]
            src_image = neuron_data.neuron_feature
            res_nf = src_image
        else:  # Decomposition of image
            for l in self.layers_data:
                if target_layer == l.layer_id:
                    target_layer = l

            img = self.dataset.load_images([input_image])
            hc_activations, hc_idx = target_layer.decomposition_image(self.model, img)
            src_image = self.dataset.load_image(input_image)

        hc_activations = hc_activations[:, :, 0]
        hc_idx = hc_idx[:, :, 0]

        res_neurons = []
        res_loc = []
        res_act = []

        i = 0
        orp_image = np.zeros(src_image.size)
        end_cond = src_image.size[0] * src_image.size[1]

        while i < end_cond:
            # Search for the maximum activations along the input image or
            # neuron feature.
            i += 1
            max_act = np.amax(hc_activations)
            pos = np.unravel_index(hc_activations.argmax(), hc_activations.shape)
            hc_activations[pos] = 0.0
            neuron_idx = hc_idx[pos]

            loc = target_layer.receptive_field_map[pos]

            # Check overlapping
            ri, rf, ci, cf = loc
            if rf <= src_image.size[0] and cf <= src_image.size[1]:
                clp = orp_image[ri:rf, ci:cf]
                clp_size = clp.shape
                clp_size = clp_size[0] * clp_size[1]
                non_zero = np.count_nonzero(clp)
                c_overlapping = float(non_zero) / float(clp_size)

                if c_overlapping <= overlapping:
                    orp_image[ri:rf, ci:cf] = 1
                    res_neurons.append(target_layer.neurons_data[int(neuron_idx)])
                    res_loc.append(loc)
                    res_act.append(max_act)

        return res_act, res_neurons, res_loc, res_nf

    def save_to_disk(self, file_name=None, save_path=None, save_model=True):
        """Save all results. The file saved will contain the
        `nefesi.network_data.NetworkData` object and the rest
         of containing objects.

        :param file_name: String, name of file.
            If `file_name` is None, the file will be named with the value
            of `model.name`.
        :param save_path: String, path to directory where save the results.
        :param save_model: If its True, the model will be saved
            as a HDF5 file.
        """
        if file_name is None:
            file_name = self.model.name
        if save_path is not None:
            self._save_path = save_path

        model_name = self.model.name
        if self._save_path is not None:
            file_name = self._save_path + file_name
            model_name = self._save_path + self.model.name

        model = self.model
        if save_model:
            self.model.save(model_name + '.h5')
        self.model = None
        pickle.dump(self, open(file_name + '.obj', 'wb'))
        self.model = model

    @staticmethod
    def load_from_disk(file_name, model_file=None):
        """Load a file with all results.

        :param file_name: String, path and file name.
        :param model_file: String, path and file name.
            Expects a file with .h5 extension.

        :return: The `nefesi.network_data.NetworkData` instance.
        """
        my_net = pickle.load(open(file_name, 'rb'))

        if model_file is not None:
            my_net.model = load_model(model_file)
        if my_net.model is None:
            warnings.warn("The model was *not* loaded. Load it manually.")

        return my_net
