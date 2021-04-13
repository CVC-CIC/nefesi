import numpy as np
import os
from .symmetry_index import SYMMETRY_AXES
from . import symmetry_index as sym
from .class_index import get_class_selectivity_idx, get_population_code_idx, get_concept_selectivity_of_neuron
from .color_index import get_ivet_color_selectivity_index, get_color_selectivity_index, get_shape_selectivity_index
from .orientation_index import get_orientation_index
from .util.image import crop_center, expand_im

class NeuronData(object):
    """This class contains all the results related with a neuron (filter) already
    evaluated, including:
    - The N-top activation values for this neuron (normalized and unnormalized).
    - The selectivity indexes for this neuron.
    - The neuron feature.

    Arguments:
        max_activations: Integer, number of maximum activations stored.
        batch_size: Integer, size of batch.
        buffered_iterations: name of iterations that are saved in buffer until sort.
    Attributes:
        activations: Numpy array of floats, activation values
        images_id: Numpy array of strings, name of the images that provokes
            the maximum activations.
        top_labels_count: dict with keys: labels of images that provokes the maximum activations,
         value: count of images with this label in top-scoring images
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

    def __init__(self, max_activations, batch_size, buffered_iterations = 20):
        self._max_activations = max_activations
        self._batch_size = batch_size
        self._buffer_size = self._max_activations + (self._batch_size*buffered_iterations)
        self.activations = np.zeros(shape=self._buffer_size)
        self.images_id = np.zeros(shape=self._buffer_size,dtype='U128')
        self.xy_locations = np.zeros(shape=(self._buffer_size,2), dtype=np.int64)
        self.xy_locations_all = np.zeros(shape=(self._buffer_size, 2), dtype=np.int64)
        self.activations_all = np.zeros(shape=self._buffer_size)
        self.norm_activations = None
        self.relevance_idx = {}
        self.most_relevant_concept = {}
        self.most_relevant_type = {}
        self.selectivity_idx = {}
        self.selectivity_idx_non_normaliced_sum = {}
        self.top_index = {}
        self._neuron_feature = None
        self.top_labels = None
        # index used for ordering activations.
        self._index = 0
        self.mean_activation = 0.
        self.mean_norm_activation = 0.
        self.images_analyzed = 0

    def add_activations(self,activations, image_ids, xy_locations):
        """Set the information of n activation. When the assigned
				 activations reach a certain size, they are ordered.

				:param activations: numpy of Floats, activation values
				:param image_ids: numpy of Strings, image names
				:param xy_locations: numpy of tuples of integers, location of the activations
					in the map activation.
				"""
        end_idx = self._index+len(activations)
        self.mean_activation += np.sum(activations)
        self.images_analyzed += len(activations)
        self.activations[self._index:end_idx] = activations
        self.images_id[self._index:end_idx] = image_ids
        self.xy_locations[self._index:end_idx,:] = xy_locations
        self._index += len(activations)
        if self._index+len(activations) > self._buffer_size:
            self.sortResults(reduce_data=False)
            #self._index = self._max_activations #Is made on function (in order to make more consistent on last iteration)


    def sortResults(self, reduce_data = False):
        idx = np.argpartition(-self.activations[:self._index], range(self._max_activations))[:self._max_activations]
        self._index = self._max_activations
        if reduce_data:
            self.activations = self.activations[idx]
            self.images_id = self.images_id[idx]
            self.xy_locations = self.xy_locations[idx,:]
        else:
            self.activations[:self._max_activations] = self.activations[idx]
            self.images_id[:self._max_activations] = self.images_id[idx]
            self.xy_locations[:self._max_activations,:] = self.xy_locations[idx, :]


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
        max_activation = np.max(self.activations)
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

    def get_patches(self, network_data, layer_data, max_rf_size=None):
        """Returns the patches (receptive fields) from images in
        `images_id` for this neuron.

        :param network_data: The `nefesi.network_data.NetworkData` instance.
        :param layer_data: The `nefesi.layer_data.LayerData` instance.
        :return: Images as numpy .
        """
        image_dataset = network_data.dataset
        receptive_field = layer_data.receptive_field_map
        rf_size = layer_data.receptive_field_size
        if layer_data.receptive_field_map is not None:
            crop_positions = receptive_field[self.xy_locations[:,0],self.xy_locations[:,1]]

            input_locations = layer_data.input_locations[self.xy_locations[:, 0], self.xy_locations[:, 1]]
        else:
            crop_positions = [None]*self.xy_locations.shape[0]
            input_locations = [rf_size]*self.xy_locations.shape[0]

        if max_rf_size is None:
            max_rf_size = layer_data.neurons_data[0].neuron_feature.size
        patch = image_dataset.get_patch(self.images_id[0], crop_positions[0])
        size = rf_size
        size = tuple([min(a,b) for a,b in zip(size, max_rf_size)])
        size = size+(patch.shape[-1],) if len(patch.shape) == 3 else size

        patches = np.zeros(shape=(self._max_activations,)+size, dtype=np.float)

        for i in range(self._max_activations):
            crop_pos = crop_positions[i]
            # crop the origin image with previous location
            patch = image_dataset.get_patch(self.images_id[i], crop_pos)
            cc = patch.shape
            # add a black padding to a patch that not match with the receptive
            # field size.
            # This is due that some receptive fields has padding
            # that come of the network architecture.
            patch = self._adjust_patch_size(patch, crop_pos, rf_size, input_locations[i])
            patches[i] = crop_center(patch, size)

        return patches

    def get_patch_by_idx(self, network_data, layer_data, i, max_rf_size=None):
        image_dataset = network_data.dataset
        receptive_field = layer_data.receptive_field_map
        rf_size = layer_data.receptive_field_size
        crop_position = receptive_field[self.xy_locations[i,0],self.xy_locations[i,1]]
        input_locations = layer_data.input_locations[self.xy_locations[i, 0], self.xy_locations[i, 1]]
        #First iteration of for, maded first in order to set the output array size
        patch = image_dataset.get_patch(self.images_id[i], crop_position)

        if max_rf_size is None:
            max_rf_size = layer_data.neurons_data[0].neuron_feature.size
        patch = self._adjust_patch_size(patch, crop_position, rf_size, input_locations)
        size = patch.shape[:2]
        size = tuple([min(a,b) for a,b in zip(size, max_rf_size )])
        return crop_center(patch, size)


    def get_patches_mask(self, network_data, layer_data, max_rf_size=None):
        """Returns the patches masks (receptive fields) from images in
        `images_id` for this neuron.

        :param network_data: The `nefesi.network_data.NetworkData` instance.
        :param layer_data: The `nefesi.layer_data.LayerData` instance.
        :return: Images as numpy .
        """
        receptive_field = layer_data.receptive_field_map
        rf_size = layer_data.receptive_field_size
        if layer_data.receptive_field_map is not None:
            crop_positions = receptive_field[self.xy_locations[:,0],self.xy_locations[:,1]]
            input_locations = layer_data.input_locations[self.xy_locations[:, 0], self.xy_locations[:, 1]]
        else:
            crop_positions = [None]*self.xy_locations.shape[0]
            input_locations = [rf_size]*self.xy_locations.shape[0]

        size = rf_size
        if max_rf_size is None:
            max_rf_size = layer_data.neurons_data[0].neuron_feature.size
        size = tuple([min(a,b) for a, b in zip(size, max_rf_size)])

        masks = np.ones(shape = (self._max_activations,)+size,dtype=np.bool)
        mask = np.ones(rf_size, dtype=np.bool)

        for i in range(self._max_activations):
            crop_pos = crop_positions[i]
            # add a black padding to a patch that not match with the receptive
            # field size.
            # This is due that some receptive fields has padding
            # that come of the network architecture.

            if rf_size is not None:
                bl, bu, br, bd = self._get_mask_borders(crop_pos, rf_size, input_locations[i])
                mask[:, :] = True
                mask[bu:rf_size[1] - bd, bl:rf_size[0] - br] = False
                masks[i] = crop_center(mask, size)

        return masks

    def _get_mask_borders(self, crop_position, rf_size, input_location=None):
        bl, bu, br, bd = (0, 0, 0, 0)
        if crop_position is not None:
            ri, rf, ci, cf = crop_position
            w = cf - ci
            h = rf - ri
            half1_rf = np.rint(np.array(rf_size)/2)[::-1]
            half2_rf = rf_size[::-1] - half1_rf
            add_before = half1_rf - (np.rint(input_location)-[ci, ri])
            add_after  = half2_rf - ([w, h] - (np.rint(input_location)-[ci, ri]))
            add_before  = np.maximum(add_before , 0)
            add_after = np.maximum(add_after, 0)

            bl, bu = add_before
            br, bd = add_after
            # if rf_size[0] != w:
            #     if ci == 0:
            #         bl = rf_size[0] - w
            #     else:
            #         br = rf_size[0] - w
            # if rf_size[1] != h:
            #     if ri == 0:
            #         bu = rf_size[1] - h
            #     else:
            #         bd = rf_size[1] - h

        return int(bl), int(bu), int(br), int(bd)

    def _adjust_patch_size(self, patch, crop_position, rf_size, input_location=None):
        if rf_size is None or rf_size == patch.size:
            return patch

        bl, bu, br, bd = self._get_mask_borders(crop_position, rf_size, input_location)
        im = expand_im(patch, (bl, bu, br, bd))
        # im = ImageOps.expand(patch, (bl, bu, br, bd), fill=0)

        return im

    def print_params(self):
        """Returns a string with some information about this neuron.
        Index of neuron, name of the image, activation value,
        activation location in map activation, normalized activation.
        """
        if self.norm_activations is None:
            print("Neuron with no activations.")
        else:
            for i in range(len(self.activations)):
                print(i, self.images_id[i],
                      self.activations[i],
                      self.xy_locations[i],
                      self.norm_activations[i])

    def remove_selectivity_idx(self, idx):
        """Removes de idx selectivity index from the neuron.

        :param idx: The selectivity index.

        :return: none.
        """
        self.selectivity_idx.pop(idx,None)

    def ivet_color_selectivity_idx(self, model, layer_data, dataset):
        """Returns the color selectivity index for this neuron.

        :param model: The `keras.models.Model` instance.
        :param layer_data: The `nefesi.layer_data.LayerData` instance.
        :param dataset: The `nefesi.util.image.ImageDataset` instance.

        :return: Float, value of color selectivity index.
        """
        key = 'ivet_color'
        if key not in self.selectivity_idx:
            self.selectivity_idx[key] = get_ivet_color_selectivity_index(self, model,
                                                     layer_data, dataset)
        return self.selectivity_idx[key]



    def shape_selectivity_idx(self, model, layer_data, dataset):
        """Returns the color selectivity index for this neuron.

        :param model: The `keras.models.Model` instance.
        :param layer_data: The `nefesi.layer_data.LayerData` instance.
        :param dataset: The `nefesi.util.image.ImageDataset` instance.

        :return: Float, value of color selectivity index.
        """
        key = 'shape'
        if key not in self.selectivity_idx:
            self.selectivity_idx[key] = get_shape_selectivity_index(self, model,
                                                     layer_data, dataset)
        return self.selectivity_idx[key]

    def get_relevance_idx(self,network_data, layer_name, neuron_idx, layer_to_ablate, for_neuron=None,
                          return_decreasing=False, print_decreasing_matrix=False):

        need_to_calculate = print_decreasing_matrix or (layer_to_ablate not in self.relevance_idx ) or (return_decreasing and
                            (layer_to_ablate not in self.most_relevant_concept or layer_to_ablate not in self.most_relevant_type))
        if not need_to_calculate:
            if for_neuron is None:
                need_to_calculate = np.sum(np.isclose(self.relevance_idx[layer_to_ablate], -1)) > 0 or (return_decreasing
                                and ((np.sum(self.most_relevant_concept[layer_to_ablate]['label'] == '') > 0) or
                                     (np.sum(self.most_relevant_type[layer_to_ablate]['label'] == '') > 0)))
            else:
                need_to_calculate = np.isclose(self.relevance_idx[layer_to_ablate][for_neuron], -1) or (return_decreasing
                            and ((self.most_relevant_concept[layer_to_ablate][for_neuron]['label'] == '') or
                                 (self.most_relevant_type[layer_to_ablate][for_neuron]['label'] == '')))
        if need_to_calculate:
            default_path_of_model = os.path.join(network_data.save_path,network_data.model.name+'.h5')
            if for_neuron is None:
                result = network_data.get_relevance_by_ablation(layer_analysis=layer_name, neuron=neuron_idx,
                                                        layer_to_ablate=layer_to_ablate, path_model=default_path_of_model,
                                                       return_decreasing=return_decreasing, print_decreasing_matrix=print_decreasing_matrix)
                if return_decreasing:
                    self.relevance_idx[layer_to_ablate], self.most_relevant_concept[layer_to_ablate], \
                    self.most_relevant_type[layer_to_ablate] = result
                else:
                    self.relevance_idx[layer_to_ablate] = result
            else:
                #creation if needed
                if layer_to_ablate not in self.relevance_idx:
                    self.relevance_idx[layer_to_ablate] = -np.ones(shape=len(network_data.get_layer_by_name(layer_to_ablate).neurons_data), dtype=np.float)
                if layer_to_ablate not in self.most_relevant_concept:
                    self.most_relevant_concept[layer_to_ablate] = np.zeros(shape=len(network_data.get_layer_by_name(layer_to_ablate).neurons_data), dtype = [('label', 'U64'), ('value', np.float)])
                if layer_to_ablate not in self.most_relevant_type:
                    self.most_relevant_type[layer_to_ablate] = np.zeros(shape=len(network_data.get_layer_by_name(layer_to_ablate).neurons_data), dtype = [('label', 'U64'), ('value', np.float)])
                result = network_data.get_relevance_by_ablation(layer_analysis=layer_name, neuron=neuron_idx,
                                                               layer_to_ablate=layer_to_ablate,
                                                               path_model=default_path_of_model, for_neuron=for_neuron,
                                                               return_decreasing=return_decreasing,
                                                           print_decreasing_matrix=print_decreasing_matrix)
                if return_decreasing:
                    self.relevance_idx[layer_to_ablate][for_neuron], self.most_relevant_concept[layer_to_ablate][for_neuron], \
                    self.most_relevant_type[layer_to_ablate][for_neuron] = result
                else:
                    self.relevance_idx[layer_to_ablate] = result

    def get_relevance_idx_no_abs(self, network_data, layer_name, neuron_idx, layer_to_ablate, for_neuron=None,
                          return_decreasing=False, print_decreasing_matrix=False):

        need_to_calculate = print_decreasing_matrix or (layer_to_ablate not in self.relevance_idx) or (
                    return_decreasing and
                    (
                                layer_to_ablate not in self.most_relevant_concept or layer_to_ablate not in self.most_relevant_type))
        if not need_to_calculate:
            if for_neuron is None:
                need_to_calculate = np.sum(np.isclose(self.relevance_idx[layer_to_ablate], -1)) > 0 or (
                            return_decreasing
                            and ((np.sum(self.most_relevant_concept[layer_to_ablate]['label'] == '') > 0) or
                                 (np.sum(self.most_relevant_type[layer_to_ablate]['label'] == '') > 0)))
            else:
                need_to_calculate = np.isclose(self.relevance_idx[layer_to_ablate][for_neuron], -1) or (
                            return_decreasing
                            and ((self.most_relevant_concept[layer_to_ablate][for_neuron]['label'] == '') or
                                 (self.most_relevant_type[layer_to_ablate][for_neuron]['label'] == '')))
        if need_to_calculate:
            default_path_of_model = os.path.join(network_data.save_path, network_data.model.name + '.h5')
            if for_neuron is None:
                result = network_data.get_relevance_by_ablation_no_abs(layer_analysis=layer_name, neuron=neuron_idx,
                                                                layer_to_ablate=layer_to_ablate,
                                                                path_model=default_path_of_model,
                                                                return_decreasing=return_decreasing,
                                                                print_decreasing_matrix=print_decreasing_matrix)
                if return_decreasing:
                    self.relevance_idx[layer_to_ablate], self.most_relevant_concept[layer_to_ablate], \
                    self.most_relevant_type[layer_to_ablate] = result
                else:
                    self.relevance_idx[layer_to_ablate] = result
            else:
                # creation if needed
                if layer_to_ablate not in self.relevance_idx:
                    self.relevance_idx[layer_to_ablate] = -np.ones(
                        shape=len(network_data.get_layer_by_name(layer_to_ablate).neurons_data), dtype=np.float)
                if layer_to_ablate not in self.most_relevant_concept:
                    self.most_relevant_concept[layer_to_ablate] = np.zeros(
                        shape=len(network_data.get_layer_by_name(layer_to_ablate).neurons_data),
                        dtype=[('label', 'U64'), ('value', np.float)])
                if layer_to_ablate not in self.most_relevant_type:
                    self.most_relevant_type[layer_to_ablate] = np.zeros(
                        shape=len(network_data.get_layer_by_name(layer_to_ablate).neurons_data),
                        dtype=[('label', 'U64'), ('value', np.float)])
                result = network_data.get_relevance_by_ablation(layer_analysis=layer_name, neuron=neuron_idx,
                                                                layer_to_ablate=layer_to_ablate,
                                                                path_model=default_path_of_model, for_neuron=for_neuron,
                                                                return_decreasing=return_decreasing,
                                                                print_decreasing_matrix=print_decreasing_matrix)
                if return_decreasing:
                    self.relevance_idx[layer_to_ablate][for_neuron], self.most_relevant_concept[layer_to_ablate][
                        for_neuron], \
                    self.most_relevant_type[layer_to_ablate][for_neuron] = result
                else:
                    self.relevance_idx[layer_to_ablate] = result

    def get_relevance_idx2(self,network_data, layer_name, neuron_idx, layer_to_ablate, for_neuron=None,
                          return_decreasing=False, print_decreasing_matrix=False):

        need_to_calculate = print_decreasing_matrix or (layer_to_ablate not in self.relevance_idx ) or (return_decreasing and
                            (layer_to_ablate not in self.most_relevant_concept or layer_to_ablate not in self.most_relevant_type))
        if not need_to_calculate:
            if for_neuron is None:
                need_to_calculate = np.sum(np.isclose(self.relevance_idx[layer_to_ablate], -1)) > 0 or (return_decreasing
                                and ((np.sum(self.most_relevant_concept[layer_to_ablate]['label'] == '') > 0) or
                                     (np.sum(self.most_relevant_type[layer_to_ablate]['label'] == '') > 0)))
            else:
                need_to_calculate = np.isclose(self.relevance_idx[layer_to_ablate][for_neuron], -1) or (return_decreasing
                            and ((self.most_relevant_concept[layer_to_ablate][for_neuron]['label'] == '') or
                                 (self.most_relevant_type[layer_to_ablate][for_neuron]['label'] == '')))
        if need_to_calculate:
            default_path_of_model = os.path.join(network_data.save_path,network_data.model.name+'.h5')
            if for_neuron is None:
                result = network_data.get_relevance_by_ablation2(layer_analysis=layer_name, neuron=neuron_idx,
                                                        layer_to_ablate=layer_to_ablate, path_model=default_path_of_model,
                                                      )
                if return_decreasing:
                    self.relevance_idx[layer_to_ablate], self.most_relevant_concept[layer_to_ablate], \
                    self.most_relevant_type[layer_to_ablate] = result
                else:
                    self.relevance_idx[layer_to_ablate] = result
            else:
                #creation if needed
                if layer_to_ablate not in self.relevance_idx:
                    self.relevance_idx[layer_to_ablate] = -np.ones(shape=len(network_data.get_layer_by_name(layer_to_ablate).neurons_data), dtype=np.float)
                if layer_to_ablate not in self.most_relevant_concept:
                    self.most_relevant_concept[layer_to_ablate] = np.zeros(shape=len(network_data.get_layer_by_name(layer_to_ablate).neurons_data), dtype = [('label', 'U64'), ('value', np.float)])
                if layer_to_ablate not in self.most_relevant_type:
                    self.most_relevant_type[layer_to_ablate] = np.zeros(shape=len(network_data.get_layer_by_name(layer_to_ablate).neurons_data), dtype = [('label', 'U64'), ('value', np.float)])
                result = network_data.get_relevance_by_ablation2(layer_analysis=layer_name, neuron=neuron_idx,
                                                               layer_to_ablate=layer_to_ablate,
                                                               path_model=default_path_of_model)
                if return_decreasing:
                    self.relevance_idx[layer_to_ablate][for_neuron], self.most_relevant_concept[layer_to_ablate][for_neuron], \
                    self.most_relevant_type[layer_to_ablate][for_neuron] = result
                else:
                    self.relevance_idx[layer_to_ablate] = result



            print('Relevance: '+layer_name+' '+str(neuron_idx)+'/'+str(len(network_data.get_layer_by_name(layer_name).neurons_data)))
        if not return_decreasing:
            if for_neuron is None:
                return self.relevance_idx[layer_to_ablate]
            else:
                return self.relevance_idx[layer_to_ablate][for_neuron]
        else:
            if for_neuron is None:
                return self.relevance_idx[layer_to_ablate], self.most_relevant_concept[layer_to_ablate], self.most_relevant_type[layer_to_ablate]
            else:
                return self.relevance_idx[layer_to_ablate][for_neuron], self.most_relevant_concept[layer_to_ablate][for_neuron], self.most_relevant_type[layer_to_ablate][for_neuron]









    def get_most_relevant_type(self, network_data, layer_name, neuron_idx, layer_to_ablate, for_neuron=None):
        _,_, most_relevant_type = self.get_relevance_idx(network_data=network_data, layer_name=layer_name, neuron_idx=neuron_idx,
                                                         layer_to_ablate=layer_to_ablate, for_neuron=for_neuron,
                                                         return_decreasing=True)
        return most_relevant_type

    def get_most_relevant_concept(self, network_data, layer_name, neuron_idx, layer_to_ablate, for_neuron=None):
        _, most_relevant_concept, _ = self.get_relevance_idx(network_data=network_data, layer_name=layer_name,
                                                          neuron_idx=neuron_idx,
                                                          layer_to_ablate=layer_to_ablate, for_neuron=for_neuron,
                                                             return_decreasing=True)
        return most_relevant_concept

    def color_selectivity_idx(self, network_data, layer_name, neuron_idx,  type='mean', th = 0.1,
                              activations_masks=None, return_non_normalized_sum = False):
        """Returns the color selectivity index for this neuron.

        :param model: The `keras.models.Model` instance.
        :param layer_data: The `nefesi.layer_data.LayerData` instance.
        :param dataset: The `nefesi.util.image.ImageDataset` instance.

        :return: Float, value of color selectivity index.
        """
        key = 'color'+type+str(th)
        if key not in self.selectivity_idx or \
        (return_non_normalized_sum and (key not in self.selectivity_idx_non_normaliced_sum)):
            self.selectivity_idx[key], self.selectivity_idx_non_normaliced_sum[key] = \
                get_color_selectivity_index(network_data=network_data, layer_name=layer_name,
                                            neuron_idx=neuron_idx, type=type, th = th,
                                            activations_masks=activations_masks, return_non_normaliced_sum=True)
            print('Color_label idx: '+layer_name+' '+str(neuron_idx)+'/'+
                  str(len(network_data.get_layer_by_name(layer_name).neurons_data)))

        if return_non_normalized_sum:
            return (self.selectivity_idx[key], self.selectivity_idx_non_normaliced_sum[key])
        else:
            return self.selectivity_idx[key]

    def max_concept_selectivity_idx(self):

        self.top_index={}
        for i in range(9):
            top_concept = 'None'
            max = 0
            for key in self.selectivity_idx.keys():
                if key != 'ivet_color':
                    suma = sum([x[1] for x in self.selectivity_idx[key] if x[1] > i*0.1])

                    if suma > max:
                        max = suma
                        top_concept = key

            self.top_index[(i+1)*0.1]=[top_concept,max]



    def color_population_code(self,network_data, layer_name, neuron_idx,  type='mean', th = 0.1):
        color_idx = self.color_selectivity_idx(network_data=network_data, layer_name=layer_name,
                                                neuron_idx=neuron_idx, type=type, th = th)
        if color_idx[0]['label'] == 'None':
            return 0
        else:
            return len(color_idx)

    def single_color_selectivity_idx(self,network_data, layer_name, neuron_idx,  type='mean', th = 0.1):
        color_idx = self.color_selectivity_idx(network_data=network_data, layer_name=layer_name,
                                                neuron_idx=neuron_idx, type=type, th = th)

        return (color_idx[0]['label'], round(np.sum(color_idx['value']),3))


    def orientation_selectivity_idx(self, model, layer_data, dataset, degrees_to_rotate = 15):
        """Returns the orientation selectivity index for this neuron.

        :param model: The `keras.models.Model` instance.
        :param layer_data: The `nefesi.layer_data.LayerData` instance.
        :param dataset: The `nefesi.util.image.ImageDataset` instance.
        :param degrees_to_rotate: degrees of each rotation step

        :return: List of floats, values of orientation selectivity index.
        """
        key = 'orientation'+str(int(degrees_to_rotate))
        if key not in self.selectivity_idx:
            self.selectivity_idx[key] = get_orientation_index(self, model,
                                                layer_data, dataset,degrees_to_rotate = degrees_to_rotate)
        return self.selectivity_idx[key]

    def symmetry_selectivity_idx(self, model, layer_data, dataset):
        """Returns the symmetry selectivity index for this neuron.

        :param model: The `keras.models.Model` instance.
        :param layer_data: The `nefesi.layer_data.LayerData` instance.
        :param dataset: The `nefesi.util.image.ImageDataset` instance.

        :return: List of floats, values of symmetry selectivity index.
        """
        key= 'symmetry'+str(SYMMETRY_AXES)
        if key not in self.selectivity_idx:
            self.selectivity_idx[key] = sym.get_symmetry_index(self, model, layer_data, dataset)

        return self.selectivity_idx[key]

    def concept_selectivity_idx(self,layer_data, network_data, neuron_idx, type='mean', concept='object', th = 0.1,
                                activations_masks = None, return_non_normalized_sum = False):
        """Returns the class selectivity index for this neuron.

        :param labels: Dictionary, key: name class, value: label class.
            This argument is needed for calculate the class index.
        :param threshold: Float, required for calculate the class index.

        :return: Float, between 0.1 and 1.0.

        :raise:
            TypeError: If `labels` is None or not a dictionary.
        """
        key = 'concept'+concept+str(th)
        if key not in self.selectivity_idx or \
        (return_non_normalized_sum and (key not in self.selectivity_idx_non_normaliced_sum)):
            if isinstance(layer_data,str):
                layer_data = network_data.get_layer_by_name(layer_data)
            self.selectivity_idx[key], self.selectivity_idx_non_normaliced_sum[key] = \
                get_concept_selectivity_of_neuron(network_data=network_data, layer_name=layer_data.layer_id,
                                                    neuron_idx=neuron_idx, type=type, concept=concept, th = 0.1,
                                                    activations_masks=activations_masks, return_non_normalized_sum=True)
            print(concept.capitalize()+' idx: ' + layer_data.layer_id + ' ' + str(neuron_idx) + '/' +
                  str(len(layer_data.neurons_data)))

        if return_non_normalized_sum:
            return (self.selectivity_idx[key], self.selectivity_idx_non_normaliced_sum[key])
        else:
            return self.selectivity_idx[key]

    def single_concept_selectivity_idx(self,layer_data, network_data, neuron_idx, type='mean', concept='object', th = 0.1):
        """Returns the class selectivity index for this neuron.

        :param labels: Dictionary, key: name class, value: label class.
            This argument is needed for calculate the class index.
        :param threshold: Float, required for calculate the class index.

        :return: Float, between 0.1 and 1.0.

        :raise:
            TypeError: If `labels` is None or not a dictionary.
        """

        concept_idx = self.concept_selectivity_idx(layer_data=layer_data,network_data=network_data,neuron_idx=neuron_idx,
                                                   type=type, concept=concept, th=th)
        return concept_idx[0]['label'], np.sum(concept_idx['value'])

    def concept_population_code(self, layer_data, network_data, neuron_idx, type='mean', concept='object',
                                       th=0.1):
        """Returns the class selectivity index for this neuron.

        :param labels: Dictionary, key: name class, value: label class.
            This argument is needed for calculate the class index.
        :param threshold: Float, required for calculate the class index.

        :return: Float, between 0.1 and 1.0.

        :raise:
            TypeError: If `labels` is None or not a dictionary.
        """

        concept_idx = self.concept_selectivity_idx(layer_data=layer_data, network_data=network_data,
                                                   neuron_idx=neuron_idx,
                                                   type=type, concept=concept, th=th)
        if concept_idx[0]['label'] is 'None':
            return 0
        else:
            return len(concept_idx)


    def class_selectivity_idx(self, labels=None, threshold=.1):
        """Returns the class selectivity index for this neuron.

        :param labels: Dictionary, key: name class, value: label class.
            This argument is needed for calculate the class index.
        :param threshold: Float, required for calculate the class index.

        :return: Float, between 0.1 and 1.0.

        :raise:
            TypeError: If `labels` is None or not a dictionary.
        """
        key = 'class'+str(threshold)
        if key not in self.selectivity_idx:
            #Labels always must to be a dictionary
            if type(labels) is not dict and labels is not None:
                raise TypeError("The 'labels' argument should be a dictionary if is specified")
            self.selectivity_idx[key] = get_class_selectivity_idx(self, labels, threshold)

        return self.selectivity_idx[key]

    def single_class_selectivity_idx(self,labels=None, threshold=.1):
        class_idx = self.class_selectivity_idx(labels=labels, threshold=threshold)
        return (class_idx[0]['label'], round(np.sum(class_idx['value']), 3))

    def classes_in_pc(self, labels=None, threshold=.1):
        return self.class_selectivity_idx(labels=labels, threshold=threshold)['label']

    def class_population_code(self, labels=None, threshold=0.1):
        """Returns the population code index for this neuron.

        :param labels: Dictionary, key: name class, value: label class.
            This argument is needed for calculate the the population
            code index.
        :param threshold: Float, between 0.1 and 1.0.

        :return: Float, value of population code index.

        :raise:
            TypeError: If `labels` is None or not a dictionary.
        """
        class_idx = self.class_selectivity_idx(labels=labels, threshold=threshold)
        if class_idx[0]['label'] == 'None':
            return 0
        else:
            return len(class_idx)
        #population_code_idx = get_population_code_idx(self, labels, threshold)

    def get_keys_of_indexes(self):
        return self.selectivity_idx.keys()
