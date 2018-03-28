import pickle
import time
import numpy as np
import warnings

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from layer_data import LayerData
from util.image import ImageDataset


class NetworkData(object):

    def __init__(self, model):
        self.model = model
        self.layers = []
        self.save_path = None
        self.num_max_activations = None
        self._dataset = None

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    def _build_layers(self, layers):
            for l in layers:
                self.layers.append(LayerData(l))

    def eval_network(self, directory, layer_names, save_path,
                     num_max_activations=100,
                     target_size=(256, 256), batch_size=100,
                     preprocessing_function=None, color_mode='rgb',
                     save_for_layers=True):

        self._build_layers(layer_names)
        self.save_path = save_path
        self.num_max_activations = num_max_activations

        times_ex = []

        self.dataset = ImageDataset(directory, target_size,
                                    preprocessing_function, color_mode)

        for layer in self.layers:

            datagen = ImageDataGenerator()
            data_batch = datagen.flow_from_directory(
                directory,
                target_size=target_size,
                batch_size=batch_size,
                shuffle=False, color_mode=color_mode
            )

            start = time.time()
            num_images = data_batch.samples
            n_batches = 0
            # filters = None

            for i in data_batch:
                images = i[0]

                if preprocessing_function is not None:
                    images = preprocessing_function(images)

                idx = (data_batch.batch_index - 1) * data_batch.batch_size
                file_names = data_batch.filenames[idx: idx + data_batch.batch_size]

                layer.evaluate_activations(file_names, images, self.model, self.num_max_activations, batch_size)

                print 'On layer ', layer.layer_id, ', Get and sort activations in batch num: ', n_batches,
                ' Images processed: ', idx + data_batch.batch_size

                n_batches += 1
                if n_batches >= num_images / data_batch.batch_size:
                    break

            layer.set_max_activations()

            end_act_time = time.time() - start

            layer.build_neuron_feature(self)

            end_comp_nf_time = time.time() - end_act_time - start

            print 'Time (s) to get and sort activations from ', num_images, ' images: ', end_act_time
            print 'Time (s) to compute_nf and receptive fields from each neuron: ', end_comp_nf_time

            times_ex.append(time.time() - start)

            if save_for_layers:
                self.save_to_disk(layer.layer_id)

        for i in xrange(len(times_ex)):
            print 'total time execution for layer ', i, ' : ', times_ex[i]

        self.save_to_disk()

    def get_layers_name(self):
        names = [l.layer_id for l in self.layers]
        return names

    def get_selectivity_idx(self, sel_index, layer_name,
                            labels=None, thr_class_idx=1., thr_pc=0.1):
        sel_idx_dict = dict()

        if type(sel_index) is not list:
            sel_index = [sel_index]
        if type(layer_name) is not list:
            layer_name = [layer_name]

        for index_name in sel_index:
            sel_idx_dict[index_name] = []
            print type(layer_name), layer_name

            for l in layer_name:
                layer = next((layer_data for layer_data in
                              self.layers if l in self.get_layers_name()), None)

                if layer is None:
                    raise ValueError('The layer_id ' + l + ' in `layer_name` '
                                     'argument, is not valid.')
                else:
                    sel_idx_dict[index_name].append(layer.selectivity_idx(
                        self.model, index_name, self.dataset, labels=labels,
                        thr_class_idx=thr_class_idx, thr_pc=thr_pc))

        return sel_idx_dict

    def similarity_index(self, layers):
        sim_idx = []
        for l in self.layers:
            if l.layer_id in layers:
                sim_idx.append(l.get_similarity_idx(self.model, self.dataset))
        return sim_idx

    def get_selective_neurons(self, layers_or_neurons, idx1, idx2=None, inf_thr=0.0, sup_thr=1.0):
        selective_neurons = dict()
        res_idx2 = None

        if type(layers_or_neurons) is list or type(layers_or_neurons) is str:
            layers = layers_or_neurons
            for l in self.layers:
                if l.layer_id in layers:
                    res_idx1 = []
                    index_values = l.selectivity_idx(self.model, idx1, self.dataset)

                    if type(index_values[0]) is list:
                        n_idx = len(index_values[0])
                        for i in index_values:
                            res_idx1.append(i[n_idx - 1])
                    else:
                        res_idx1 = index_values

                    if idx2 is not None:
                        res_idx2 = []
                        index_values2 = l.selectivity_idx(self.model, idx2, self.dataset)

                        if type(index_values2[0]) is list:
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
            raise TypeError('Parameter 1 should be a list of layers, layer layer_id or dict')
        if idx2 is not None:
            idx1 = (idx1, idx2)
        res = {idx1: selective_neurons}
        return res


    def get_max_activations(self, layer_id, img_name, location, num_max):
        layer = None
        if type(layer_id) is int:
            layer = self.layers[layer_id]
        if type(layer_id) is str:
            for l in self.layers:
                if layer_id == l.layer_id:
                    layer = l

        img = self.dataset.load_images([img_name])
        hc_activations, hc_idx = layer.decomposition_image(self.model, img)
        loc = layer.get_location_from_rf(location)

        activations = hc_activations[loc[0], loc[1], :num_max]
        neuron_idx = hc_idx[loc[0], loc[1], :num_max]

        f = layer.get_filters()
        neurons = []
        for idx in neuron_idx:
            neurons.append(f[int(idx)])

        return list(activations), neurons, layer.receptive_field_map[loc[0], loc[1]]

    def decomposition(self, input_image, target_layer, overlapping=0.0):

        print input_image
        print type(input_image)
        res_nf = None
        if isinstance(input_image, list):  # decomposition of neuron feature

            print 22222
            src_layer = input_image[0]
            neuron_idx = input_image[1]

            for l in self.layers:
                if src_layer == l.layer_id:
                    src_layer = l
                elif target_layer == l.layer_id:
                    target_layer = l

            hc_activations, hc_idx = src_layer.decomposition_nf(
                neuron_idx, target_layer, self.model, self.dataset)

            neuron_data = src_layer.filters[neuron_idx]
            src_image = neuron_data.neuron_feature
            res_nf = src_image

            # src_image.show()
            print src_image.size
            # orp_image = np.zeros(src_image.size)
        else:  # decomposition of an image
            for l in self.layers:
                if target_layer == l.layer_id:
                    target_layer = l

            img = self.dataset.load_images([input_image])
            hc_activations, hc_idx = target_layer.decomposition_image(self.model, img)
            src_image = self.dataset.load_image(input_image)

            # orp_image = np.zeros((src_image.shape[0], src_image.shape[1]))

        hc_activations = hc_activations[:, :, 0]
        hc_idx = hc_idx[:, :, 0]

        # c_overlapping = 0.0
        res_neurons = []
        res_loc = []
        res_act = []

        i = 0
        orp_image = np.zeros(src_image.size)
        end_cond = src_image.size[0] * src_image.size[1]

        while i < end_cond:
            i += 1
            max_act = np.amax(hc_activations)
            pos = np.unravel_index(hc_activations.argmax(), hc_activations.shape)
            hc_activations[pos] = 0.0
            neuron_idx = hc_idx[pos]

            loc = target_layer.receptive_field_map[pos]

            # check overlapping
            ri, rf, ci, cf = loc
            if rf <= src_image.size[0] and cf <= src_image.size[1]:

                clp = orp_image[ri:rf, ci:cf]
                clp_size = clp.shape
                clp_size = clp_size[0]*clp_size[1]
                non_zero = np.count_nonzero(clp)
                c_overlapping = float(non_zero)/float(clp_size)

                if c_overlapping <= overlapping:
                    orp_image[ri:rf, ci:cf] = 1
                    res_neurons.append(target_layer.get_filters()[int(neuron_idx)])
                    res_loc.append(loc)
                    res_act.append(max_act)

            # c_overlapping = 1

        # print hc_activations
        # print hc_idx.shape

        return res_act, res_neurons, res_loc, res_nf

    def save_to_disk(self, file_name=None, save_path=None, save_model=True):
        if file_name is None:
            file_name = self.model.name

        if save_path is not None:
            self.save_path = save_path

        model_name = self.model.name
        if self.save_path is not None:
            file_name = self.save_path + file_name
            model_name = self.save_path + self.model.name

        model = self.model
        if save_model:
            self.model.save(model_name + '.h5')
        self.model = None
        pickle.dump(self, open(file_name + '.obj', 'wb'))
        self.model = model

    @staticmethod
    def load_from_disk(file_name, model_file=None):

        my_net = pickle.load(open(file_name, 'rb'))

        if model_file is not None:
            my_net.model = load_model(model_file)

        if my_net.model is None:
            warnings.warn('The model was *not* loaded. Load it manually.')

        return my_net