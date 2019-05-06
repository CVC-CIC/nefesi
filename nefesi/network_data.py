import dill as pickle
import os
import time
import datetime
import re #Regular Expresions
import numpy as np
import warnings

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.backend import clear_session
from .neuron_data import NeuronData
from .util.general_functions import get_key_of_index
from .layer_data import LayerData
from .neuron_feature import compute_nf
from .util.image import ImageDataset
from .read_activations import fill_all_layers_data_batch
from .class_index import get_concept_labels
from .util.ColorNaming import colors as color_names
from keras.layers import Conv2D, Input
from keras.models import Sequential, Model, clone_model
from tensorflow import Session
import nefesi.util.GPUtil as gpu
gpu.assignGPU()


MIN_PROCESS_TIME_TO_OVERWRITE = 10
ALL_INDEX_NAMES = ['symmetry', 'orientation', 'color', 'class', 'object', 'part']

class NetworkData(object):
    """This is the main class of nefesi package.

    Creates an instance of NetworkData, the main object containing
    all evaluated data related with a `keras.models.Model` instance.

    Arguments:
        model: The `keras.models.Model` instance.

    Attributes:
        layers_data: List of `nefesi.layer_data.LayerData` instances at the end, setted by:
            RegularExpression ('.*' by default) applyed in model layer names or a list of names(string) that user knows to
            exists in model (can be shown by self.showModelLayerNames). Also accepts a list of nefesi.layer_data.LayerData objects.
            Specifies the layers that will be analyzed.


    Mutable-properties:
        save_path: Path of directory where the results will be saved.
        dataset: The `nefesi.util.image.ImageDataset` instance.
    """

    def __init__(self, model,layer_data = '.*', save_path = None, dataset = None, save_changes = False,
                 default_labels_dict = None, default_degrees_orientation_idx = 15,default_thr_pc = 0.1,
                 default_thr_class_idx = .1, default_file_name = None, indexes_accepted = ALL_INDEX_NAMES):
        self.model = model
        self.layers_data = layer_data
        self.save_path = save_path
        self.dataset = dataset
        self.save_changes = save_changes
        self.default_labels_dict = default_labels_dict
        self.default_degrees_orientation_idx = default_degrees_orientation_idx
        self.default_thr_pc = default_thr_pc
        #self.default_thr_class_idx = default_thr_class_idx
        self.default_file_name = default_file_name
        self.indexs_accepted = indexes_accepted
        self.MIN_PROCESS_TIME_TO_OVERWRITE = MIN_PROCESS_TIME_TO_OVERWRITE


    @property
    def default_file_name(self):
        return self._default_file_name

    @default_file_name.setter
    def default_file_name(self, default_file_name):
        if default_file_name is None:
            default_file_name = self.model.name
        self._default_file_name = default_file_name

    @property
    def default_labels_dict(self):
        return self._default_labels_dict
    @default_labels_dict.setter
    def default_labels_dict(self, default_labels_dict):
        if default_labels_dict is not None:
            if type(default_labels_dict) is str:
                with open(default_labels_dict, "rb") as f:
                    default_labels_dict = pickle.load(f)
            if type(default_labels_dict) is not dict:
                warnings.warn("Default labels dict expect a str(path) or dict, '"+type(default_labels_dict)+"' is "
                            "not valid. Default_labels_dict not modified")
                try:
                    default_labels_dict = self._default_labels_dict
                except:
                    default_labels_dict = None
        self._default_labels_dict = default_labels_dict

    @property
    def layers_data(self):
        return self._layers_data

    @layers_data.setter
    def layers_data(self, layer_data):

        if layer_data is None:
            self._layers_data = None
        #If is a regular expresion
        elif type(layer_data) is str:
            #Compile the Regular expresion
            regEx = re.compile(layer_data)
            #Select the layerNames that satisfies RegEx
            okList = list(filter(regEx.match,[layer.name for layer in self.model.layers]))
            #Put as LayerData objects
            self._layers_data = [LayerData(layer) for layer in okList]
            if self._layers_data == []:
                warnings.warn("No layer was caught from filter: '"+layer_data+"'. For see all layer names of the model"
                                                                              "calls show_model_layer_names()",RuntimeWarning)
        #if layer_data is a list
        elif type(layer_data) is list:
            #list of strings
            if all(type(name) is str for name in layer_data):
                model_layer_names = [layer.name for layer in self.model.layers]
                for layer_n in layer_data:
                    if layer_n not in model_layer_names:
                        raise ValueError("Wrong layer name: '"+layer_n+"'. For see all layer names of the model calls"
                                                                       " showModelLayerNames()")
                self._layers_data = [LayerData(layer) for layer in layer_data]
            #list of layerData
            elif all(type(name) is LayerData for name in layer_data):
                model_layer_names = [layer.name for layer in self.model.layers]
                for layer_n in layer_data:
                    if layer_n.layer_id not in model_layer_names:
                        raise ValueError("Wrong layer at LayerData object with id: '"+layer_n.layer_id+"'."
                                                " For see all layer names of the model calls showModelLayerNames()")
                self._layers_data = [LayerData(layer) for layer in layer_data]
            else:
                raise ValueError("The list types values accepted by 'layer_data' are only String Lists and LayerData Lists")
        else:
            raise ValueError("The value of LayerData must be: 'None', 'String List', 'LayerData List', or a Regular Expresion (Regex)")

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        if type(dataset) is not ImageDataset and dataset is not None:
            raise TypeError("Dataset must be an nefesi.util.Image.ImageDataset object "
                            "(https://github.com/CVC-CIC/nefesi/blob/master/nefesi/util/image.py)")

        self._dataset = dataset

    @property
    def save_path(self):
        return self._save_path

    @save_path.setter
    def save_path(self, save_path):
        if save_path is not None:
            if type(save_path) is not str:
                raise ValueError("save_path must be a str.")
            #Ensures that path ends with '/' (To save confusions to user)
            elif not save_path.endswith('/'):
                save_path = save_path + '/'
            #Looks folder exists
            if not os.path.isdir(save_path):
                warnings.warn(save_path+" not exists or is not a directory. It will be created when needed",RuntimeWarning)
        self._save_path = save_path

    def eval_network(self, layer_names = None,
                     directory=None,
                     save_path=None,
                     num_max_activations=100,
                     target_size=(254, 254),
                     batch_size=100,
                     preprocessing_function=None,
                     color_mode='rgb',
                     save_for_layers=True,
                     build_nf=True,
                     file_name=None,
                     verbose=True):
        """Evaluates the layers in `layer_names`, searching for the maximum
        activations for each neuron and build the neuron feature.

        :param layer_names: List of strings, string, or Regular Expressions (name of the layers of the network that will be
            evaluate (in order to see the disponible layers use show_model_layer_names())).
        :param directory: Path to the directory to read images from.
        :param save_path: Path of directory to write the results. If None the value of self._save_path will be used.
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
        #Assign layer_names if is specified and control that self.layers_data are setted consistently
        if layer_names is not None:
            self.layers_data = layer_names
        elif type(self.layers_data) is not list:
            raise ValueError("layers_data attribute not setted. It must be setted on network_data object, before call"
                             " eval_network(...) or setted as argument (layer_names) in eval_network function")

        #Assign save_path if is specified and control that self.save_path is setted consistently
        if save_path is not None:
            self.save_path = save_path
        elif type(self.save_path) is not str:
            raise ValueError("Save_path attribute not setted. It must be setted on network_data object, before call"
                             " eval_network(...) or setted as argument (save_path) in eval_network function")

        #Creates an ImageDataset object if it not exists and control that self.dataset is setted consistenly
        if directory is not None:
                self.dataset = ImageDataset(directory, target_size,
                                        preprocessing_function, color_mode)
        elif type(self.dataset) is not ImageDataset:
            raise ValueError("Dataset attribute not setted. It must be setted on network_data object, before call"
                             " eval_network(...) or setted as argument(s) (directory,[target_size, preprocessing_function,"
                             " color_mode]) in eval_network function ")
        if preprocessing_function is not None:
            self.dataset.preprocessing_function = preprocessing_function
        #if layer.layer_id in layer_names: #if is not, exception was raised
        datagen = ImageDataGenerator(preprocessing_function=self.dataset.preprocessing_function)
        data_batch = datagen.flow_from_directory(
            self.dataset.src_dataset,
            target_size=self.dataset.target_size,
            batch_size=batch_size,
            shuffle=True,
            color_mode=self.dataset.color_mode
        )

        num_images = data_batch.samples
        idx_start = data_batch.batch_index
        idx_end = idx_start + data_batch.batch_size
        #the min between full size and 0,5MB by array (and 64MB for the img_name)
        #buffer_size = min(num_images//batch_size, 524288//(batch_size*np.dtype(np.float).itemsize))
        buffer_size = 4
        #Init all neuron_data attributes of all layers
        for i in range(len(self.layers_data)):
            neurons_of_layer = self.model.get_layer(self.layers_data[i].layer_id).output_shape[-1]
            self.layers_data[i].neurons_data = np.zeros(neurons_of_layer, dtype=np.object)
            for j in range(neurons_of_layer):
                self.layers_data[i].neurons_data[j] = NeuronData(num_max_activations, batch_size, buffered_iterations=buffer_size)

        file_names = np.array(data_batch.filenames)
        start = time.time()
        for n_batches, imgs in enumerate(data_batch):
            images = imgs[0]
            # Apply the preprocessing function to the inputs
            actual_file_names = file_names[data_batch.index_array[idx_start: idx_end]]
            # Search the maximum activations
            fill_all_layers_data_batch(actual_file_names, images, self.model, self.layers_data)

            if verbose:
                img_sec = idx_end / (time.time() - start)
                print("Num batch: {batch},"
                      " Num images processed: {processed}/{total}. Remaining: {secs}".format(
                        batch=n_batches,
                        processed=idx_end,
                        total=num_images,
                        secs=str(datetime.timedelta(seconds=(num_images-idx_end)/img_sec))))

            idx_start, idx_end = idx_end, idx_end+data_batch.batch_size

            # If the idx of the next last image will overpass the total num of images, ends the analysis
            if idx_end > num_images:
                break
            elif n_batches%1000==0 and n_batches!=0:
                self.save_to_disk(file_name=self.model.name+'PartialSave'+str(int((idx_start/data_batch.samples)*100))+
                                            'PerCent',erase_partials=True)
        if verbose:
            print("Analysis ended. Sorting results")
        for layer in self.layers_data:
            layer.sort_neuron_data()
            # Set the number of maximum activations stored in each neuron
            layer.set_max_activations()

        if build_nf:
            if verbose:
                print("Sort ended. Building neuron features")
            # Build the neuron features
            for layer in self.layers_data:
                compute_nf(self, layer)
                self.save_to_disk(file_name=file_name, erase_partials=True)
        # Save all data
        self.save_to_disk(file_name=file_name,erase_partials=True)

    def recalculateNF(self, file_name=None, layers=None):
        if layers is None:
            layers = self.layers_data
        else:
            layers = [l for l in self.layers_data if l.layer_id in layers]

        for layer in layers:
            print(layer.layer_id)
            layer.receptive_field_map = None
            compute_nf(self, layer)

        self.save_to_disk(file_name=file_name, save_model=False)


    def get_layers_name(self):
        """Builds a list with the name of each layer in `layers_data`.

        :return: List of strings
        """
        names = [l.layer_id for l in self.layers_data]
        return names

    def remove_selectivity_idx(self, idx):
        """Removes de idx selectivity index from the neurons of the layers of the network.

        :param idx: The selectivity index.

        :return: none.
        """
        for l in self.layers_data:
            l.remove_selectivity_idx(idx)

    def get_selectivity_idx(self, sel_index, layer_name, degrees_orientation_idx = None,
                            labels=None,
                            thr_pc=None, verbose = True, only_calc = False):
        """Returns the selectivity indexes in `sel_index` for each layer
        in `layer_name`.

        :param sel_index: List of strings or string, name of the selectivity indexes.
            Values: "color", "orientation", "symmetry", "class" or "population code".
        :param layer_name: List of strings, string or Regular Expression, name of the layers.
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
        start_time = time.time() #in order to update things if something new was be calculated
        sel_idx_dict = {}
        if labels is None:
            labels=self.default_labels_dict
        if thr_pc is None:
            thr_pc = self.default_thr_pc
        if degrees_orientation_idx is None:
            degrees_orientation_idx = self.default_degrees_orientation_idx

        if type(sel_index) is not list:
            sel_index = [sel_index]
        if type(layer_name) is not list:
            # Compile the Regular expresion
            regEx = re.compile(layer_name)
            # Select the layerNames that satisfies RegEx
            layer_name = list(filter(regEx.match, [layer for layer in self.get_layers_name()]))

        for index_name in sel_index:
            sel_idx_dict[index_name] = []
            for l in layer_name:
                layer = self.get_layer_by_name(layer=l)
                if not only_calc:
                    sel_idx_dict[index_name].append(layer.selectivity_idx(
                        self.model, index_name, self.dataset, degrees_orientation_idx=degrees_orientation_idx,
                        labels=labels, thr_pc=thr_pc,verbose=verbose,network_data=self))
                else:
                    layer.get_all_index_of_all_neurons(self, orientation_degrees=degrees_orientation_idx, thr_pc=thr_pc,
                                  indexes = None, is_first_time=True)

                if self.save_changes:
                    end_time = time.time()
                    if end_time - start_time >= MIN_PROCESS_TIME_TO_OVERWRITE:
                        if verbose:
                            print("Layer: "+l+" - Index: "+index_name+" saving changes")
                        # Update only the modelName.obj
                        self.save_to_disk(file_name=None, save_model=False)
                    start_time = end_time

        return sel_idx_dict

    """
    def get_progress(self):
        if not hasattr(self, 'layers_in_analysis') or not hasattr(self, 'last_layers_to_analyze')\
                or self.last_layers_analyzed != self.last_layers_to_analyze:
            self.layers_in_analysis = [layer_data for layer_data in
                                       self.layers_data if layer_data.layer_id in self.last_layers_to_analyze]
            for layer in self.layers_in_analysis:
                layer.neurons_complete = 0
        self.last_layers_analyzed = self.last_layers_to_analyze
        total_neurons = sum([len(l.neurons_data) for l in self.layers_in_analysis])
        neurons_completed = sum([l.neurons_complete for l in self.layers_in_analysis])
        return neurons_completed/total_neurons
    """
    def similarity_idx(self, layer_name):
        """Returns the similarity index for each layer in `layer_name`.

        :param layer_name: List of strings, string or Regular Expression, name of the layers.

        :return: List of Numpy arrays, each array belows to one layer.

        :raise:
            ValueError: If layer name in `layer_name` doesn't match with any layer
            inside the class property `layers`.
        """
        sim_idx = []
        start_time = time.time()  # in order to update things if something new was be calculated
        if type(layer_name) is not list:
            # Compile the Regular expresion
            regEx = re.compile(layer_name)
            # Select the layerNames that satisfies RegEx
            layer_name = list(filter(regEx.match, [layer for layer in self.get_layers_name()]))


        for l in layer_name:
            layer = next((layer_data for layer_data in
                          self.layers_data if l in self.get_layers_name()
                          and l == layer_data.layer_id), None)
            if layer is None:
                raise ValueError("The layer_id '{}' `layer_name` "
                                 "argument, is not valid.".format(l))
            else:
                sim_idx.append(layer.get_similarity_idx(self.model, self.dataset))
            if self.save_changes:
                end_time = time.time()
                if end_time-start_time>=MIN_PROCESS_TIME_TO_OVERWRITE:
                    #Update only the modelName.obj
                    self.save_to_disk(file_name=None, save_model=False)
        return sim_idx


    def get_relevance_idx(self, layer_name = '.*', layer_to_ablate = None, verbose=True):
        relevance_idx = []
        start_time = time.time()  # in order to update things if something new was be calculated
        if type(layer_name) is not list:
            # Compile the Regular expresion
            regEx = re.compile(layer_name)
            # Select the layerNames that satisfies RegEx
            layer_name = list(filter(regEx.match, [layer for layer in self.get_layers_name()]))
        if layer_name[0] == self.layers_data[0].layer_id:
            layer_name = layer_name[1:]
        if layer_to_ablate is not None and type(layer_to_ablate) is not list:
            layer_to_ablate = [layer_to_ablate]
        for i, l in enumerate(layer_name):
            layer = next((layer_data for layer_data in
                          self.layers_data if l in self.get_layers_name()
                          and l == layer_data.layer_id), None)
            if layer is None:
                raise ValueError("The layer_id '{}' `layer_name` "
                                 "argument, is not valid.".format(l))
            else:
                if layer_to_ablate is None:
                    layer_ablated = self.get_ablatable_layers(layer.layer_id)[-1]
                else:
                    layer_ablated = layer_to_ablate[i]
                #layer_ablated don't have sense yet. It will have sense when update to relation with specific layer
                relevance_idx.append(layer.get_relevance_matrix(network_data=self, layer_to_ablate=layer_ablated))

            if self.save_changes:
                end_time = time.time()
                if end_time - start_time >= MIN_PROCESS_TIME_TO_OVERWRITE:
                    # Update only the modelName.obj
                    self.save_to_disk(file_name=None, save_model=False)
                    if verbose:
                        print(layer.layer_id+' relevance saved')
        return relevance_idx


    def get_relevance_by_ablation(self, layer_analysis, neuron, layer_to_ablate,path_model='/home/guillem/nefesi/Data/XceptionAdds/xception.h5', return_decreasing = False):
        """Returns the relevance of each neuron in the previous layer for neuron in layer_analysis

            :param self: Nefesi object
            :param model: Original Keras model
            :param layer_analysis: String with the layer to analyze
            :param neuron: Int with the neuron to analyze
            :return: A list with: the sum of the difference between the original max activations and the max activations after ablating each previous neuron
            """
        current_layer = self.get_layer_by_name(layer_analysis)
        neuron_data = self.get_neuron_of_layer(layer_analysis, neuron)
        xy_locations = neuron_data.xy_locations
        image_names = neuron_data.images_id
        images = self.dataset.load_images(image_names=image_names, prep_function=True)
        clear_session()
        new_keras_sess= Session()
        with new_keras_sess.as_default():
            cloned_model= load_model(path_model)

            layer_names = [x.name for x in cloned_model.layers]
            numb_ablated=layer_names.index(layer_to_ablate)
            numb_analysis=layer_names.index(layer_analysis)+1
            intermediate_layer_model = Model(inputs=cloned_model.input, outputs=cloned_model.get_layer(layer_to_ablate).output)
            intermediate_output = intermediate_layer_model.predict(images)

            DL_input = Input(cloned_model.layers[numb_ablated+1].input_shape[1:],name=layer_to_ablate)

            mymodel_layers=[DL_input]
            mymodel_layer_names=[layer_to_ablate]
            for layer in cloned_model.layers[numb_ablated+1:numb_analysis]:
                if type(layer.input) == list:
                    list_inputs_names=[x.name.split('/')[0] for x in layer.input]
                    list_inputs_nubers=[mymodel_layer_names.index(x) for x in list_inputs_names]
                    list_input_layers=[mymodel_layers[x] for x in list_inputs_nubers]
                    new_layer=layer(list_input_layers)

                else:
                    new_layer=layer(mymodel_layers[mymodel_layer_names.index(layer.input.name.split('/')[0])])

                mymodel_layer_names.append(layer.name)
                mymodel_layers.append(new_layer)


            DL_model = Model(inputs=mymodel_layers[0],outputs=mymodel_layers[-1])
            original_activations = self.get_neuron_of_layer(layer_analysis, neuron).activations
            relevance_idx = []
            if return_decreasing:
                max_concept_decreasing, max_type_decreasing = [], []
            pre_ablation_indexes = current_layer.get_all_index_of_a_neuron(network_data=self, neuron_idx=neuron)
            for i in range(intermediate_output.shape[-1]):
                intermediate_output2 = intermediate_output[..., i]*1 #To copy
                intermediate_output[..., i] = 0
                predictionsf = DL_model.predict(intermediate_output)
                intermediate_output[..., i] = intermediate_output2
                ablated_neurons_predictions = predictionsf[..., neuron]
                #get the activation on the same point
                max_activations = ablated_neurons_predictions[range(0,100),xy_locations[:,0], xy_locations[:,1]]
                relevance_idx.append(np.sum(abs(original_activations - max_activations))/np.sum(original_activations))
                if return_decreasing:
                    post_ablation_indexes = current_layer.calculate_all_index_of_a_neuron(network_data=self,
                                                                                          neuron_idx=neuron,
                                                    norm_act=max_activations/original_activations[0],
                                                    activations_masks = ablated_neurons_predictions, thr_pc=0.0)

                    max_concept, max_type = self.most_decreased_index(pre_indexes=pre_ablation_indexes,
                                                           post_indexes=post_ablation_indexes)
                    max_concept_decreasing.append(max_concept)
                    max_type_decreasing.append(max_type)

            clear_session()
        self.model = load_model(path_model)
        relevance_idx = np.array(relevance_idx)
        if return_decreasing:
            max_concept_decreasing, max_type_decreasing = np.array(max_concept_decreasing), np.array(max_type_decreasing)
            return (relevance_idx, max_concept_decreasing, max_type_decreasing)
        else:
            return relevance_idx

    def most_decreased_index(self, pre_indexes, post_indexes):
        indexes_decreased = self.indexes_decreasing(pre_indexes=pre_indexes, post_indexes=post_indexes)
        index_decreasing_by_key = np.array(list(map(lambda key:
                                                    (key, np.sum(indexes_decreased[key]['value'])), indexes_decreased)),
                                           dtype=[('label', 'U64'), ('value', np.float)])

        max_concept = index_decreasing_by_key[np.argmax(index_decreasing_by_key['value'])]
        if np.isclose(max_concept['value'],0.0):
            max_concept['label'] = 'None'

        max_type_by_index = np.array(list(map(lambda key:
                                         indexes_decreased[key][np.argmax(indexes_decreased[key]['value'])],
                                         indexes_decreased)))

        max_type = max_type_by_index[np.argmax(max_type_by_index['value'])]

        if np.isclose(max_type['value'],0.0):
            max_type['label'] = 'None'


        return (max_concept, max_type)


    def indexes_decreasing(self, pre_indexes, post_indexes):
        indexes_decreased = {}
        for key in post_indexes.keys():
            pre_index, post_index = pre_indexes[key], post_indexes[key]
            labels = list(set(pre_index['label']) | set(post_index['label'])) #Take all labels in two
            decreasing_list = []
            for label in labels:
                if label in pre_index['label'] and label in post_index['label']:
                    decreasing = np.clip(pre_index[pre_index['label'] == [label]]['value'] - \
                                 post_index[post_index['label'] == [label]]['value'], 0.0, 1.0)

                elif label in pre_index['label']:
                    decreasing = np.clip(pre_index[pre_index['label'] == [label]]['value'],0.0,1.0)
                else:
                    #let's use 0 and not a negative for be coheren with use 0.0 as thr_pc on post_index (We are wanting the max, not min)
                    #decreasing = -post_index[post_index['label'] == [label]]['value']
                    decreasing = 0.0
                decreasing_list.append((label, decreasing))
            indexes_decreased[key] = np.array(decreasing_list, dtype=[('label', 'U64'), ('value', np.float)])
        return indexes_decreased


    def get_entinty_co_ocurrence_matrix(self, layers=None, th=None, entity = 'class', operation='1/PC'):
        if layers is None:
            layers = self.get_layer_names_to_analyze()
        elif type(layers) is not list:
            layers = self.get_layers_analyzed_that_match_regEx(layers)

        if th is None:
            th = self.default_thr_pc

        if entity == 'class':
            labels = np.array(list(self.default_labels_dict.values()))
        elif entity == 'object':
            labels = get_concept_labels(entity)
        elif entity == 'color':
            labels =np.array(color_names)

        pairs_matrix = np.zeros((len(layers),len(labels),len(labels)),dtype=np.float)

        for l, layer in enumerate(layers):
            layer_data = self.get_layer_by_name(layer)
            pairs_matrix[l,:,:] = layer_data.get_entity_coocurrence_matrix(network_data=self, th=th,
                                                                                  entity=entity,operation=operation)

        return pairs_matrix,labels

    def get_entinty_representation_vector(self, layers=None, th=None, entity = 'class', operation='1/PC'):
        if layers is None:
            layers = self.get_layer_names_to_analyze()
        elif type(layers) is not list:
            layers = self.get_layers_analyzed_that_match_regEx(layers)

        if th is None:
            th = self.default_thr_pc

        if entity == 'class':
            labels = np.array(list(self.default_labels_dict.values()))
        elif entity == 'object':
            labels = get_concept_labels(entity)
        elif entity == 'color':
            labels = np.array(color_names)
        representation_vector = np.zeros((len(layers),len(labels)),dtype=np.float)

        for l, layer in enumerate(layers):
            layer_data = self.get_layer_by_name(layer)
            representation_vector[l,:] = layer_data.get_entity_representation(network_data=self, th=th,
                                                                                  entity=entity, operation=operation)
        return representation_vector,labels


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
                        print(type(index_values2[0]))

                        if type(index_values2[0]) is list or type(index_values2[0]) is tuple:
                            n_idx = len(index_values2[0])
                            for i in index_values2:
                                res_idx2.append(i[n_idx - 1])
                        else:
                            res_idx2 = index_values2

                    selective_neurons[l.layer_id] = []
                    neurons = l.neurons_data
                    for i in range(len(neurons)):
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

    def save_to_disk(self, file_name=None, save_model=True, erase_partials=False):
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

        save_path = os.path.dirname(file_name)
        file_name = os.path.basename(file_name )

        if file_name is None or file_name is '':
            file_name = self.default_file_name
        if not file_name.endswith('.obj'):
            file_name+='.obj'
        if save_path is not None and save_path is not '':
            self.save_path = save_path

        model_name = self.model.name
        if self.save_path is not None:
            file_name = os.path.join(self.save_path, file_name)
            model_name = os.path.join(self.save_path, self.model.name)
        #If directory not exists create it recursively
        os.makedirs(name=self.save_path, exist_ok=True)
        model = self.model
        if save_model:
            self.model.save(model_name + '.h5')
        #Save the object without model info
        self.model = None
        if erase_partials:
            for file in os.listdir(self.save_path):
                if 'PartialSave' in file:
                    os.remove(self.save_path+file)
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
        self.model = model

    def load_from_disk(self, file_name, model_file=None):
        """Load a file with all results.

        :param file_name: String, path and file name.
        :param model_file: String, path and file name.
            Expects a file with .h5 extension.

        :return: The `nefesi.network_data.NetworkData` instance.
        """
        save_path = os.path.dirname(file_name)
        file_name = os.path.basename(file_name)

        if file_name is None or file_name is '':
            file_name = self.default_file_name
        if not file_name.endswith('.obj'):
            file_name += '.obj'
        if save_path is not None and save_path is not '':
            self.save_path = save_path

        with open(os.path.join(save_path, file_name), 'rb') as f:
            my_net = pickle.load(f)
        """
        TODO: make a copy constructor that copy my_net on another network_data object. In order to compatibilice old
        obj's with news implementations of network_data that can have more attributes
        """
        if model_file is not None:
            my_net.model = load_model(model_file)
        if my_net.model is None:
            warnings.warn("The model was *not* loaded. Load it manually.")

        return my_net
    #--------------------------------HELP FUNCTIONS-------------------------------------------------
    def get_layer_names_to_analyze(self):
        return [layer.layer_id for layer in self._layers_data]
    def get_layers_analyzed_that_match_regEx(self, regEx):
        # Compile the Regular expresion
        regEx = re.compile(regEx)
        # Select the layerNames that satisfies RegEx
        return list(filter(regEx.match, [layer for layer in self.get_layers_name()]))
    def get_len_neurons_of_layer(self, layer):
        for layer_of_model in self.layers_data:
            if layer_of_model.layer_id == layer:
                return len(layer_of_model.neurons_data)
        raise ValueError("Layer: "+layer+" doesn't exists")
    def get_neuron_of_layer(self,layer, neuron_idx):
        for layer_of_model in self.layers_data:
            if layer_of_model.layer_id == layer:
                return layer_of_model.neurons_data[neuron_idx]
        raise ValueError("Layer: " + layer + " doesn't exists")
    def get_all_index_of_neuron(self, layer, neuron_idx,orientation_degrees=None, thr_pc=None):
        if orientation_degrees is None:
            orientation_degrees = self.default_degrees_orientation_idx
        if thr_pc is None:
            thr_pc = self.default_thr_pc
        if type(layer) is not LayerData:
            layer = next((lay for lay in self.layers_data if lay.layer_id == layer), None)

        if layer is not None:
            return layer.get_all_index_of_a_neuron(network_data=self,neuron_idx=neuron_idx,
                                                                orientation_degrees=orientation_degrees,
                                                                thr_pc=thr_pc)
        else:
            raise ValueError("Layer: " + layer + " doesn't exists")

    def get_ablatable_layers(self,actual_layer):
        all_layer_names = self.get_layer_names_to_analyze()
        idx = all_layer_names.index(actual_layer)
        return all_layer_names[:idx]
    def get_layer_by_name(self, layer):
        for layer_of_model in self.layers_data:
            if layer_of_model.layer_id  == layer:
                return layer_of_model

        raise ValueError("Layer: " + layer + " doesn't exists")

    def addmits_concept_selectivity(self):
        try:
            self.layers_data[0].neurons_data[0].concept_selectivity_idx(layer_data=self.layers_data[0],network_data=self,
                                                                        neuron_idx=0)
            return True
        except:
            return False

    def get_calculated_indexes_keys(self):
        keys = set()
        for layer in self.layers_data:
            keys |= layer.get_index_calculated_keys()
        return keys
    def is_index_in_layer(self,layers,index, special_value):
        if special_value is None:
            if index == 'orientation':
                special_value = self.default_degrees_orientation_idx
            elif index == 'population code':
                special_value = self.default_thr_pc
        key = get_key_of_index(index, special_value)
        if layers in [str, np.str_]:
            layers = self.get_layers_analyzed_that_match_regEx(layers)
        for layer in layers:
            layer_data = self.get_layer_by_name(layer=layer)
            if layer_data.is_not_calculated(key):
                return False
            return True
    def get_layers_with_index(self, index_selected):
        return [layer.layer_id for layer in self.layers_data if index_selected in layer.get_index_calculated_keys()]

    def erase_index_from_layers(self, layers, index_to_erase):
        for layer_name in layers:
            self.get_layer_by_name(layer_name).erase_index(index_to_erase)


def get_model_layer_names(model, regEx='.*'):
    if model is None:
        return []
    else:
        # Compile the Regular expresion
        regEx = re.compile(regEx)
        # Select the layerNames that satisfies RegEx
        return list(filter(regEx.match, [layer.name for layer in model.layers]))

