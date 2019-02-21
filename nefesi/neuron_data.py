import numpy as np
from PIL import ImageOps
from keras.preprocessing import image
from .symmetry_index import SYMMETRY_AXES
from . import symmetry_index as sym
from .class_index import get_class_selectivity_idx, get_population_code_idx, get_concept_selectivity_of_neuron
from .color_index import get_ivet_color_selectivity_index, get_color_selectivity_index
from .orientation_index import get_orientation_index


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

    def __init__(self, max_activations, batch_size,buffered_iterations = 20):
        self._max_activations = max_activations
        self._batch_size = batch_size
        self._buffer_size = self._max_activations + (self._batch_size*buffered_iterations)
        # If is the final iteration reduce the size of activations, images_id and xy_locations to max_activations
        self._reduce_data = True
        self.activations = np.zeros(shape=self._buffer_size)
        self.images_id = np.zeros(shape=self._buffer_size,dtype='U128')
        self.xy_locations = np.zeros(shape=(self._buffer_size,2), dtype=np.int64)
        self.norm_activations = None

        self.selectivity_idx = dict()
        self._neuron_feature = None
        self.top_labels = None
        # index used for ordering activations.
        self._index = 0

    def add_activations(self,activations, image_ids, xy_locations):
        """Set the information of n activation. When the assigned
				 activations reach a certain size, they are ordered.

				:param activations: numpy of Floats, activation values
				:param image_ids: numpy of Strings, image names
				:param xy_locations: numpy of tuples of integers, location of the activations
					in the map activation.
				"""
        end_idx = self._index+len(activations)
        self.activations[self._index:end_idx] = activations
        self.images_id[self._index:end_idx] = image_ids
        self.xy_locations[self._index:end_idx,:] = xy_locations
        self._index += len(activations)
        if self._index+len(activations) > self._buffer_size:
            self._reduce_data = False
            self.sortResults()
            self._reduce_data = True
            #self._index = self._max_activations #Is maded on function (in order to make more consistent on last iteration


    def sortResults(self):
        idx = np.argpartition(-self.activations[:self._index], range(self._max_activations))[:self._max_activations]
        self._index = self._max_activations
        if self._reduce_data:
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

    def get_patches(self, network_data, layer_data, as_numpy=True, return_mask=False):
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
        else:
            crop_positions = [None]*self.xy_locations.shape[0]

        if as_numpy:
            patch = image.img_to_array(image_dataset.get_patch(self.images_id[0], crop_positions[0]))
            size = rf_size+(patch.shape[-1],) if len(patch.shape) == 3 else rf_size
            patches = np.zeros(shape = (self._max_activations,)+size,dtype=np.float)
        else:
            patches = np.zeros(shape = (self._max_activations), dtype=np.object)
        if return_mask:
            masks = np.zeros(shape = (self._max_activations,)+rf_size,dtype=np.bool)
        for i in range(self._max_activations):
            mask = None
            crop_pos = crop_positions[i]
            # crop the origin image with previous location
            patch = image_dataset.get_patch(self.images_id[i], crop_pos)

            # add a black padding to a patch that not match with the receptive
            # field size.
            # This is due that some receptive fields has padding
            # that come of the network architecture.
            if  rf_size is not None and rf_size != patch.size:
                if return_mask:
                    patch,mask = self._adjust_patch_size(patch,crop_pos, rf_size,returns_mask=return_mask)
                else:
                    patch = self._adjust_patch_size(patch,crop_pos, rf_size)
            if as_numpy:
                patch = image.img_to_array(patch)
            patches[i] = patch
            if mask is not None:
                masks[i] = mask
        if return_mask:
            return patches,masks
        else:
            return patches

    def get_patch_by_idx(self, network_data, layer_data, i):
        image_dataset = network_data.dataset
        receptive_field = layer_data.receptive_field_map
        rf_size = layer_data.receptive_field_size
        crop_position = receptive_field[self.xy_locations[i,0],self.xy_locations[i,1]]
        #First iteration of for, maded first in order to set the output array size
        patch = image_dataset.get_patch(self.images_id[i], crop_position)

        if rf_size != patch.size:
            patch = self._adjust_patch_size(patch, crop_position, rf_size)
        return patch
    def _adjust_patch_size(self, patch, crop_position, rf_size,returns_mask=False):
        bl, bu, br, bd = (0, 0, 0, 0)
        if crop_position is not None:
            w, h = patch.size
            ri, rf, ci, cf = crop_position
            if rf_size[0] != w:
                if ci == 0:
                    bl = rf_size[0] - w
                else:
                    br = rf_size[0] - w
            if rf_size[1] != h:
                if ri == 0:
                    bu = rf_size[1] - h
                else:
                    bd = rf_size[1] - h
        image = ImageOps.expand(patch, (bl, bu, br, bd),fill= 0)
        if returns_mask:
            mask = np.ones((image.size[1],image.size[0]),dtype=np.bool)
            mask[bu:bu + patch.size[1], bl:bl + patch.size[0]] = False
            return image, mask
        else:
            return image

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

    def color_selectivity_idx(self, network_data, layer_name, neuron_idx,  type='mean', th = 0.1):
        """Returns the color selectivity index for this neuron.

        :param model: The `keras.models.Model` instance.
        :param layer_data: The `nefesi.layer_data.LayerData` instance.
        :param dataset: The `nefesi.util.image.ImageDataset` instance.

        :return: Float, value of color selectivity index.
        """
        key = 'color'+type+str(th)
        if key not in self.selectivity_idx:
            self.selectivity_idx[key] = get_color_selectivity_index(network_data=network_data,
                                                                        layer_name=layer_name,
                                                                        neuron_idx=neuron_idx,
                                                                        type=type, th = th)
            print('Color idx: '+layer_name+' '+str(neuron_idx)+'/'+
                  str(len(network_data.get_layer_by_name(layer_name).neurons_data)))

        return self.selectivity_idx[key]

    def color_population_code(self,network_data, layer_name, neuron_idx,  type='mean', th = 0.1):
        return len(self.color_selectivity_idx(network_data=network_data, layer_name=layer_name,
                                                neuron_idx=neuron_idx, type=type, th = th))

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
        orientation_idx = self.selectivity_idx.get(key)
        if orientation_idx is not None:
            return orientation_idx

        orientation_idx = get_orientation_index(self, model,
                                                layer_data, dataset,degrees_to_rotate = degrees_to_rotate)
        self.selectivity_idx[key] = orientation_idx
        return orientation_idx

    def symmetry_selectivity_idx(self, model, layer_data, dataset):
        """Returns the symmetry selectivity index for this neuron.

        :param model: The `keras.models.Model` instance.
        :param layer_data: The `nefesi.layer_data.LayerData` instance.
        :param dataset: The `nefesi.util.image.ImageDataset` instance.

        :return: List of floats, values of symmetry selectivity index.
        """
        key= 'symmetry'+str(SYMMETRY_AXES)
        symmetry_idx = self.selectivity_idx.get(key)
        if symmetry_idx is not None:
            return symmetry_idx

        symmetry_idx = sym.get_symmetry_index(self, model, layer_data, dataset)
        self.selectivity_idx[key] = symmetry_idx
        return symmetry_idx

    def concept_selectivity_idx(self,layer_data, network_data, neuron_idx, type='mean', concept='object', th = 0.1):
        """Returns the class selectivity index for this neuron.

        :param labels: Dictionary, key: name class, value: label class.
            This argument is needed for calculate the class index.
        :param threshold: Float, required for calculate the class index.

        :return: Float, between 0.1 and 1.0.

        :raise:
            TypeError: If `labels` is None or not a dictionary.
        """
        concept_idx = self.selectivity_idx.get('concept'+concept+str(th))
        if concept_idx is not None:
            return concept_idx
        if not isinstance(layer_data,str):
            layer_data = layer_data.layer_id
        concept_idx = get_concept_selectivity_of_neuron(network_data=network_data, layer_name=layer_data,
                                                        neuron_idx=neuron_idx, type=type, concept=concept, th = 0.1)

        self.selectivity_idx['concept'+concept+str(th)] = concept_idx
        return concept_idx

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
        class_idx = self.selectivity_idx.get('class'+str(threshold))
        if class_idx is not None:
            return class_idx

        #Labels always must to be a dictionary
        if type(labels) is not dict and labels is not None:
            raise TypeError("The 'labels' argument should be a dictionary if is specified")

        class_idx = get_class_selectivity_idx(self, labels, threshold)
        self.selectivity_idx['class'+str(threshold)] = class_idx
        return class_idx

    def single_class_selectivity_idx(self,labels=None, threshold=.1):
        class_idx = self.class_selectivity_idx(labels=labels, threshold=threshold)
        return (class_idx[0]['label'], round(np.sum(class_idx['value']), 3))

    def classes_in_pc(self, labels=None, threshold=.1):
        class_idx = self.class_selectivity_idx(labels=labels, threshold=threshold)

        return class_idx['label']

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
        key = 'population code'+str(round(threshold,2))
        population_code_idx = self.selectivity_idx.get(key)
        if population_code_idx is not None:
            return population_code_idx

        population_code_idx = get_population_code_idx(self, labels, threshold)

        self.selectivity_idx[key] = population_code_idx
        return population_code_idx

    def get_keys_of_indexs(self):
        return self.selectivity_idx.keys()
