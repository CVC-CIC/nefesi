import numpy as np

import math
from .read_activations import get_sorted_activations, get_activations, get_one_neuron_activations
from .neuron_feature import compute_nf
from .similarity_index import get_row_of_similarity_index
from .symmetry_index import SYMMETRY_AXES
from .class_index import get_concept_labels, get_class_selectivity_idx, get_concept_selectivity_of_neuron
from .util.ColorNaming import colors as color_names
from itertools import permutations

MIN_PROCESS_TIME_TO_OVERWRITE = 10

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
        self.entity_coocurrence = {}


    def set_max_activations(self):
        for f in self.neurons_data:
            f.set_max_activations()
            f.mean_activation /= f.images_analyzed
            f.mean_norm_activation = f.mean_activation/f.activations[0]

    def sort_neuron_data(self):
        for neuron in self.neurons_data:
            neuron.sortResults(reduce_data=True)

    def evaluate_activations(self, file_names, images, model, num_max_activations, batch_size,batches_to_buffer = 20):
        self.neurons_data = get_sorted_activations(file_names, images, model,
                                                   self.layer_id, self.neurons_data,
                                                   num_max_activations, batch_size,batches_to_buffer=batches_to_buffer)


    def remove_selectivity_idx(self, idx):
        """Removes de idx selectivity index from the neurons of the layer.


        :return: none.
        """
        for n in self.neurons_data:
            n.remove_selectivity_idx(idx)

    def selectivity_idx(self, model, index_name, dataset,
                        labels=None, thr_pc=0.1,degrees_orientation_idx = 15, verbose=True,
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
                sel_idx[i] = self.neurons_data[i].ivet_color_selectivity_idx(model, self, dataset)
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
                sel_idx[i] = self.neurons_data[i].single_class_selectivity_idx(labels, thr_pc)
        elif index_name.lower() in ['object', 'part']:
            sel_idx = np.zeros(len(self.neurons_data), dtype=np.dtype([('label','U64'), ('value',np.float)]))
            for i in range(len(self.neurons_data)):
                sel_idx[i] = self.neurons_data[i].single_concept_selectivity_idx(network_data=network_data,
                                                                                 layer_data=self, neuron_idx=i,
                                                                                 concept=index_name.lower(), th=thr_pc)
        elif index_name.lower() == 'homogeneized_color':
            sel_idx = np.zeros(len(self.neurons_data), dtype=np.dtype([('label','U64'), ('value',np.float)]))
            for i in range(len(self.neurons_data)):
                if verbose:
                    print(self.layer_id + ": " + str(i) + "/" + str(len(self.neurons_data)))
                sel_idx[i] = self.neurons_data[i].single_color_selectivity_idx(network_data=network_data,
                                                                                 layer_name=self.layer_id, neuron_idx=i,
                                                                                 th=thr_pc)
        elif index_name.lower() == 'population code':
            sel_idx = np.zeros(len(self.neurons_data), dtype=np.int)
            for i in range(len(self.neurons_data)):
                if verbose:
                    print(self.layer_id+": "+str(i)+"/"+str(len(self.neurons_data)))
                sel_idx[i] = self.neurons_data[i].class_population_code(labels, thr_pc)
        else:
            raise ValueError("The 'index_name' argument should be one "
                             "of theses: "+str(network_data.indexs_accepted))
        return sel_idx

    def get_all_index_of_all_neurons(self, network_data, orientation_degrees=90, thr_pc=0.1,
                                  indexes = None, is_first_time=False):
        index_list = []
        if not is_first_time:
            activations_masks = None
        for i in range(len(self.neurons_data)):
            if is_first_time:
                inputs = network_data.dataset.load_images(image_names=self.neurons_data[i].images_id, prep_function=True)
                activations_masks = get_one_neuron_activations(model=network_data.model, model_inputs=inputs,
                                                               layer_name=self.layer_id, idx_neuron=i)

            index_list.append(self.get_all_index_of_a_neuron(network_data=network_data, neuron_idx=i,
                                           orientation_degrees=orientation_degrees, thr_pc=thr_pc,indexes=indexes,
                                                          activations_masks = activations_masks))

        return index_list
    def get_all_index_of_a_neuron(self, network_data, neuron_idx, orientation_degrees=90, thr_pc=0.1,
                                  indexes = None, activations_masks = None):
        """

        :param network_data:
        :param neuron_idx:
        :param orientation_degrees:
        :param thr_pc:
        :param concept:
        :param indexes: list of indexes to calc (or none for all)
        accepted --> ['symmetry', 'orientation', 'color', 'class', 'population code', 'object']
        :return:
        """
        assert(neuron_idx >=0 and neuron_idx<len(self.neurons_data))
        model = network_data.model
        import time
        start_time = time.time()
        dataset = network_data.dataset
        neuron = self.neurons_data[neuron_idx]

        index = {}
        if indexes is None:
            indexes = network_data.indexs_accepted

        if 'orientation' in indexes:
            orientation = np.zeros(int(math.ceil(360/orientation_degrees)), dtype=np.float)
            orientation[:-1] = neuron.orientation_selectivity_idx(model, self, dataset,
                                                             degrees_to_rotate=orientation_degrees)
            orientation[-1] = np.mean(orientation[:-1])
            index['orientation'] = orientation

        if 'symmetry' in indexes:
            symmetry = np.zeros(len(SYMMETRY_AXES)+1, dtype=np.float)
            symmetry[:-1] = neuron.symmetry_selectivity_idx(model, self, dataset)
            symmetry[-1] = np.mean(symmetry[:-1])
            index['symmetry'] = symmetry

        if 'color' in indexes or 'ivet_color' in indexes:
            index['color'] = neuron.color_selectivity_idx(layer_name=self.layer_id, network_data=network_data,
                                                              neuron_idx=neuron_idx, th=thr_pc,
                                                          activations_masks=activations_masks)
            #index['ivet_color'] = neuron.ivet_color_selectivity_idx(model, self, dataset)

        if 'class' in indexes:
            index['class'] = neuron.class_selectivity_idx(network_data.default_labels_dict, thr_pc)

        if 'object' in indexes:
            index['object'] = neuron.concept_selectivity_idx(layer_data=self, network_data=network_data,
                                                              neuron_idx=neuron_idx, concept='object', th=thr_pc,
                                                             activations_masks=activations_masks)
        if 'part' in indexes:
            index['part'] = neuron.concept_selectivity_idx(layer_data=self, network_data=network_data,
                                                             neuron_idx=neuron_idx, concept='part', th=thr_pc,
                                                           activations_masks=activations_masks)

        if network_data.save_changes:
            end_time = time.time()
            if end_time - start_time >= MIN_PROCESS_TIME_TO_OVERWRITE:
                # Update only the modelName.obj
                network_data.save_to_disk(file_name=None, save_model=False)
        return index

    def calculate_all_index_of_a_neuron(self, network_data, neuron_idx, norm_act, orientation_degrees=90, thr_pc=0.1,
                                  indexes = None, activations_masks = None, original_norm_act= None, type='mean'):
        """

        :param network_data:
        :param neuron_idx:
        :param orientation_degrees:
        :param thr_pc:
        :param concept:
        :param indexes: list of indexes to calc (or none for all)
        accepted --> ['symmetry', 'orientation', 'color', 'class', 'population code', 'object']
        :return:
        """
        assert(neuron_idx >=0 and neuron_idx<len(self.neurons_data))
        dataset = network_data.dataset
        neuron = self.neurons_data[neuron_idx]
        index = {}
        if indexes is None:
            indexes = network_data.indexs_accepted

        if 'orientation' in indexes:
            orientation = np.zeros(int(math.ceil(360/orientation_degrees)), dtype=np.float)
            orientation[:-1] = neuron.orientation_selectivity_idx(network_data.model, self, dataset,
                                                             degrees_to_rotate=orientation_degrees)
            orientation[-1] = np.mean(orientation[:-1])
            index['orientation'] = orientation

        if 'symmetry' in indexes:
            symmetry = np.zeros(len(SYMMETRY_AXES)+1, dtype=np.float)
            symmetry[:-1] = neuron.symmetry_selectivity_idx(network_data.model, self, dataset)
            symmetry[-1] = np.mean(symmetry[:-1])
            index['symmetry'] = symmetry
        from nefesi.color_index import get_color_selectivity_index
        if 'color' in indexes or 'ivet_color' in indexes:

            index['color'] = get_color_selectivity_index(network_data=network_data,
                                        layer_name=self.layer_id,
                                        neuron_idx=neuron_idx,
                                        type=type, th=thr_pc, activations_masks=activations_masks)
            #index['ivet_color'] = neuron.ivet_color_selectivity_idx(model, self, dataset)

        if 'class' in indexes:

            index['class'] = get_class_selectivity_idx(neuron, network_data.default_labels_dict, thr_pc,
                                                       norm_act=norm_act, original_norm_act=original_norm_act)

        if 'object' in indexes:
            index['object'] = get_concept_selectivity_of_neuron(network_data=network_data,
                                                                          layer_name=self.layer_id,
                                                                          neuron_idx=neuron_idx,
                                                                          type=type, concept='object', th = thr_pc,
                                                              activations_masks=activations_masks)
        if 'part' in indexes:
            index['part'] = get_concept_selectivity_of_neuron(network_data=network_data,
                                                                          layer_name=self.layer_id,
                                                                          neuron_idx=neuron_idx,
                                                                          type=type, concept='part', th = thr_pc,
                                                              activations_masks=activations_masks)

        return index

    def get_entity_coocurrence_matrix(self,network_data, th=None, entity='class',operation='1/PC'):
        key = entity+'coocurrence-th:'+str(th)+'-op:'+operation
        if key not in self.entity_coocurrence:
            self.entity_coocurrence[key] = self._get_entity_coocurrence_matrix(network_data=network_data, th=th,
                                                                                  entity=entity,operation=operation)
        return self.entity_coocurrence[key]

    def get_relevance_matrix(self,network_data, layer_to_ablate):
        relevance_matrix = []
        for i, neuron in enumerate(self.neurons_data):
            relevance_matrix.append(neuron.get_relevance_idx(network_data= network_data, layer_name= self.layer_id,
                                                             neuron_idx=i,layer_to_ablate=layer_to_ablate,
                                                             return_decreasing=False))
        return np.array(relevance_matrix)

    def _get_entity_coocurrence_matrix(self,network_data, th=None, entity='class', operation='1/PC'):
        """

        :param network_data:
        :param th:
        :param entity: 'class' or 'object'
        :param operation: '1/PC', '1/2' or 'local selecitivity sum'
        :return:
        """
        if entity == 'class':
            dict_labels = network_data.default_labels_dict
            labels = list(dict_labels.values())
        elif entity == 'object':
            labels = list(get_concept_labels(entity))
        elif entity == 'color':
            labels = color_names
        if th is None:
            th = network_data.default_thr_pc

        entity_pairs_matrix = np.zeros((len(labels), len(labels)), dtype=np.float)
        for i, neuron in enumerate(self.neurons_data):
            if entity == 'class':
                selective_entities = neuron.class_selectivity_idx(labels=dict_labels, threshold=th)
            elif entity == 'object':
                selective_entities = neuron.concept_selectivity_idx(layer_data=self,network_data=network_data, neuron_idx=i,th=th)
            elif entity == 'color':
                selective_entities = neuron.color_selectivity_idx(layer_name=self.layer_id, network_data=network_data,
                                                                    neuron_idx=i,th=th)
            if selective_entities['label'][0] == 'None':
                continue
            else:
                pc = len(selective_entities)
                if operation == '1/PC' or operation == '1/2':
                    index_entities = [labels.index(selective_entity['label']) for selective_entity in selective_entities]
                    if pc == 1:
                        entity_pairs_matrix[index_entities[0], index_entities[0]] += 1
                    else:
                        weight = 1/float(pc) if operation == '1/PC' else 0.5
                        for permutation in permutations(index_entities, 2):
                            entity_pairs_matrix[permutation[0], permutation[1]] += weight

                elif operation == 'local selectivity sum':
                    index_entities = [(labels.index(selective_entity['label']), selective_entity['value'])
                                      for selective_entity in selective_entities]
                    if pc == 1:
                        entity_pairs_matrix[index_entities[0][0], index_entities[0][0]] += index_entities[0][1]
                    else:
                        for permutation in permutations(index_entities, 2):
                            weight = permutation[0][1] + permutation[1][1]
                            entity_pairs_matrix[permutation[0][0], permutation[1][0]] += weight

        return entity_pairs_matrix

    def get_entity_representation(self,network_data, th=None, entity='class', operation='1/PC'):
        key = entity+'representation-th:'+str(th)+'-op:'+operation
        if key not in self.entity_coocurrence:
            self.entity_coocurrence[key] = self._get_entity_representation(network_data=network_data, th=th,
                                                                                  entity=entity, operation=operation)
        return self.entity_coocurrence[key]



    def _get_entity_representation(self,network_data, th=None, entity='class', operation='1/PC'):
        """

        :param network_data:
        :param th:
        :param entity:
        :param operation: '1/PC','1', 'local selectivity'
        :return:
        """
        if entity == 'class':
            dict_labels = network_data.default_labels_dict
            labels = list(dict_labels.values())
        elif entity == 'object':
            labels = list(get_concept_labels(entity))
        elif entity == 'color':
            labels = color_names

        if th is None:
            th = network_data.default_thr_pc

        entity_representation = np.zeros(len(labels), dtype=np.float)

        for i, neuron in enumerate(self.neurons_data):
            if entity == 'class':
                selective_entities = neuron.class_selectivity_idx(labels=dict_labels, threshold=th)
            elif entity == 'object':
                selective_entities = neuron.concept_selectivity_idx(layer_data=self,network_data=network_data, neuron_idx=i, th=th)
            elif entity == 'color':
                selective_entities = neuron.color_selectivity_idx(layer_name=self.layer_id, network_data=network_data,
                                                                    neuron_idx=i,th=th)

            if selective_entities['label'][0] == 'None':
                continue
            else:
                if operation == 'local selectivity':
                    index_entities = [(labels.index(selective_entity['label']), selective_entity['value'])
                                      for selective_entity in selective_entities]
                else:
                    index_entities = [labels.index(selective_entity['label']) for selective_entity in selective_entities]

                if operation == '1/PC':
                    pc = len(selective_entities)
                    relation_weight = 1 / float(pc)
                    for index in index_entities:
                        entity_representation[index] += relation_weight
                elif operation == '1':
                    for index in index_entities:
                        entity_representation[index] += 1.
                elif operation == 'local selectivity':
                    for index, weight in index_entities:
                        entity_representation[index] += weight

        return entity_representation


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

        if self.receptive_field_map is None:
            layer_idx = find_layer_idx(model, self.layer_id)
            if len(model.layers[layer_idx].output_shape) != 4:
                _, h, w, _ = model.input_shape
                self.receptive_field_size = (h, w)
                self.receptive_field_map = np.zeros((h, w, 4), dtype=np.int32)
                self.receptive_field_map[:] = [0, h, 0, w]
            else:
                self.receptive_field_map, self.receptive_field_size, self.input_locations = get_each_point_receptive_field(model, self.layer_id)

    def get_location_from_rf(self, location):
        """Given a pixel of an image (x, y), returns a location in the map
        activation that corresponds to receptive field with the center more nearest
        to the pixel position (x, y).

        :param location: Tuple of integers, pixel location from the image.
        :return: Integer tuple, a location of the map activation.
        )"""
        idx = np.argmin(((self.input_locations.reshape(-1,2)-location)**2).sum(axis=1).sqrt())
        r, c = ind2sub(self.input_locations.shape, [idx])
        return self.input_locations[r, c]

        # row, col = location
        # ri, rf, ci, cf = self.receptive_field_map[0, 0]
        # ri2, rf2, ci2, cf2 = self.receptive_field_map[1, 1]
        # stride_r = rf2 - rf
        # stride_c = cf2 - cf
        #
        # return row / stride_r, col / stride_c

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
        xy_locations, activations = get_activations(model, neuron_images,
                                      #print_shape_only=True,
                                      layers_name=[target_layer.layer_id])

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
            keys |= set(neuron.get_keys_of_indexes())
            if len(neuron.relevance_idx.keys()):
                keys.add('Relevance')
        return keys

    def erase_index(self, index_to_erase):
        for neuron in self.neurons_data:
            if index_to_erase.lower() == 'relevance':
                neuron.relevance_idx = {}
            else:
                neuron.remove_selectivity_idx(idx=index_to_erase)

    def is_not_calculated(self, key):
        for neuron in self.neurons_data:
            if key not in neuron.get_keys_of_indexes():
                return True
        return False

def get_each_point_receptive_field(model, layer_name):
    """Takes `weight` and `height` (of a layer output)  and gets the map receptive_field_map of each pixel to the input layer
    (usually same as input image size).

    :param model: The `keras.models.Model` instance.
    :param layer_name: String, name of the layer to get his receptive_field to input.

    :return: The window location of the receptive field in the input image.
    Numpy matrix (3-D) that contains for each point in matrix(i,j) --> [row_ini, row_fin, col_ini, col_fin].
        output[i,j] = 4 points of rectangle or square that corresponds to pixel i,j on a neuron on layer layer name, to
        on the input image. The exact position of the receptive field from an image.
    """
    current_layer_idx = find_layer_idx(model, layer_name=layer_name)
    if len(model.layers[current_layer_idx].output_shape)>2:
        _, w, h, _ = model.layers[current_layer_idx].output_shape
    else:
        h = 1
        _, w = model.layers[current_layer_idx].output_shape
    w_mesh, h_mesh = np.meshgrid(range(h), range(w))
    # array order --> row_ini, row_fin, col_ini, col_fin
    image_points = np.array([h_mesh.flatten(),h_mesh.flatten(), w_mesh.flatten(),w_mesh.flatten()],dtype=np.int32).\
        T.reshape(w, h, 4)
    input_locations = np.zeros((w, h, 2))

    image_points = recursive_receptive_field_per_location(model, model.layers[current_layer_idx], image_points)

    # (is neccesary to add 1 in row_fin and col_fin due to behaviour of Numpy arrays.
    image_points[:, :, [1, 3]] += 1

    _, current_size_w, current_size_h, _ = model.layers[0].input_shape

    # calculate the size of receptive field
    if len(model.layers[current_layer_idx].output_shape) == 4:
        _, w, h, _ = model.layers[current_layer_idx].output_shape
    else:
        raise ValueError(
            "You're trying to get the receptive field of a NON Convolutional layer? --> " + current_layer_idx)
    row = int(w // 2)
    col = int(h // 2)
    row_ini, row_fin, col_ini, col_fin = image_points[row, col]
    height = row_fin - row_ini
    width = col_fin - col_ini
    receptive_field_size = (width, height)

    input_locations[..., 0] = (image_points[..., 2] + image_points[..., 3]) / 2
    input_locations[..., 1] = (image_points[..., 0] + image_points[..., 1]) / 2

    image_points[:, :, [0, 2]] = np.maximum(image_points[:, :, [0, 2]], 0)
    image_points[:, :, 1] = np.minimum(image_points[:, :, 1], current_size_h)
    image_points[:, :, 3] = np.minimum(image_points[:, :, 3], current_size_w)

    return image_points, receptive_field_size, input_locations


def recursive_receptive_field_per_location(model, current_layer, image_points):
    image_points = np.copy(image_points)
    # REVIEW IF W AND H ARE CORRECT!!!!!!!!!!!
    if len(current_layer.input_shape) == 4:
        _, current_size_w, current_size_h, _ = current_layer.input_shape
    else:
        current_size_w, current_size_h = (float('Inf'), float('Inf'))

    # Checks to boundaries of the current layer shape.
    # image_points[:, :, [0, 2]] = np.maximum(image_points[:, :, [0, 2]], 0)
    # image_points[:, :, 1] = np.minimum(image_points[:, :, 1], current_size_w - 1)
    # image_points[:, :, 3] = np.minimum(image_points[:, :, 3], current_size_h - 1)
    # check if the current layer is a convolution layer or
    # a pooling layer (both have to be 2D).
    config_params = current_layer.get_config()

    kernel_size = np.ones(shape=2, dtype=np.int)
    pool_size = np.ones(shape=2, dtype=np.int)
    if 'kernel_size' in config_params:
        kernel_size = np.array(config_params.get('kernel_size'))
    if 'pool_size' in config_params:
        pool_size = config_params.get('pool_size')
    kernel_size = np.maximum(kernel_size, pool_size)

    padding = np.zeros(shape=2, dtype=np.int)
    if 'padding' in config_params:
        padding = config_params['padding']
        if padding == 'same':
            # padding = same, means input shape = output shape
            padding = (kernel_size - 1) // 2
        else:
            padding = np.zeros(shape=2, dtype=np.int)

    strides = np.ones(shape=2, dtype=np.int)
    if 'strides' in config_params:
        strides = np.array(config_params['strides'])

    image_points *= strides[[0, 0, 1, 1]]
    image_points[:, :, [1, 3]] += (kernel_size - 1)

    # apply the padding on the receptive field window.
    image_points -= padding[[0, 0, 1, 1]]

    input_layers, _ = get_layer_inputs(model, current_layer)
    im_points = np.copy(image_points)
    if len(input_layers) > 0:
        image_points = recursive_receptive_field_per_location(model, input_layers[0], im_points)
        for input_layer in input_layers[1:]:
            im2_points = recursive_receptive_field_per_location(model, input_layer, im_points)
            image_points[:, :, [0, 2]] = np.minimum(image_points[:, :, [0, 2]], im2_points[:, :, [0, 2]])
            image_points[:, :, [1, 3]] = np.maximum(image_points[:, :, [1, 3]], im2_points[:, :, [1, 3]])

    return image_points


def find_layer_idx(model, layer_name):
    """Returns the layer index corresponding to `layer_name` from `model`.

    :param model: The `keras.models.Model` instance.
    :param layer_name: String, name of the layer to lookup.

    :return: Integer, the layer index.

    :raise
        ValueError: If there isn't a layer with layer id = `layer_name`
            in the model.
    """
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            return idx
    else:
        raise ValueError("No layer with layer_id '{}' within the model".format(layer_name))


def get_layer_inputs(model, layer):
    inputs = []
    for i, node in enumerate(layer._inbound_nodes):
        node_key = layer.name + '_ib-' + str(i)
        if node_key in model._network_nodes:
            for inbound_layer in node.inbound_layers:
                # if not (isinstance(inbound_layer, Wrapper) and isinstance(inbound_layer.layer, Model)):
                inputs.append(inbound_layer)
    return inputs, layer

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)