import numpy as np
import os
from .util import general_functions as gf
from . import read_activations as read_act
from anytree import Node
LABEL_NAME_POS = 0
HUMAN_NAME_POS = 1
COUNT_POS = 2
REL_FREQ_POS = 3
CONCEPT_TRANSLATION_BASE_DIR = '../nefesi/util/segmentation/meta_file/'

"""
def get_concept_selectivity_idx(neuron_data, layer_data, network_data,index_by_level=5,
                                normalize_by_activations = False):
    Returns the class selectivity index value.

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param labels: Dictionary, key: name class, value: label class.
    :param threshold: Float.

    :return: A tuple with: label class and class index value.
    if neuron_data.top_labels is None:
        _fill_top_labels(neuron_data)

    receptive_field = layer_data.receptive_field_map
    crop_positions = receptive_field[neuron_data.xy_locations[:, 0], neuron_data.xy_locations[:, 1]]
    image_dataset = network_data.dataset
    images_id = neuron_data.images_id
    activations = neuron_data.norm_activations
    if normalize_by_activations:
        images = image_dataset.load_images(images_id)
        neuron_idx = np.where(layer_data.neurons_data == neuron_data)[0][0]
        image_activations = read_act.get_one_neuron_activations(model = network_data.model, model_inputs=images,
                                                       idx_neuron=neuron_idx, layer_name=layer_data.layer_id)
    concepts=[]
    for i in range(len(images_id)):
        if normalize_by_activations:
            ri, rf, ci, cf = crop_positions[i]
            activations = image_activations[i][ri:rf, ci:cf]
            norm_activations = activations/np.sum(activations)
        else:
            norm_activations = None
        concepts_i = image_dataset.get_concepts_of_region(image_name=images_id[i],
                                                          crop_pos=crop_positions[i], normalized=False,
                                                          norm_activations=norm_activations)

        for level, dic in enumerate(concepts_i):
            for k,v in dic.items():
                if norm_activations is None:
                    v *= activations[i]
                if len(concepts)<=level:
                    concepts.append(dict())
                if k in concepts[level]:
                    concepts[level][k] += v
                else:
                    concepts[level][k] = v
    #image_size_sum = \
    #    np.sum((crop_positions[:, 1] - crop_positions[:, 0]) * (crop_positions[:, 3] - crop_positions[:, 2]))
    for i, level_concept in enumerate(concepts):
        labels = np.array(list(level_concept.items()), dtype=([('class', 'U64'), ('count', np.float)]))
        labels = np.sort(labels, order='count')[::-1]
        #Normalization
        if norm_activations is None:
            labels['count'] /= np.sum(labels['count'])
        else:
            labels['count'] /= len(images_id)
        labels['class'] = np.char.strip(labels['class'])
        concepts[i] = labels[:min(len(labels), index_by_level)]

    return np.array(concepts)
"""
def concept_selectivity_of_image(activations_mask, segmented_image, type='mean'): #activations_mask
    """
    :param type: 'max' = only the max of every region, 'sum' sum of the pixels of a region, 'mean' the mean of the activation
    in each region, 'percent' the plain percent, 'activation' the max activation of the n-topScoring.
    :return:
    """
    if not (type == 'activation' or type == 'percent'):
        activations_mask = activations_mask.reshape(-1)
    ids, correspondency = np.unique(segmented_image, return_inverse=True)
    histogram = np.zeros(shape=len(ids), dtype=np.float)
    if type == 'max':
        for i in range(len(ids)):
            histogram[i] = np.max(activations_mask[correspondency == i])
    elif type == 'sum':
        for i in range(len(ids)):
            histogram[i] = np.sum(activations_mask[correspondency == i])
    elif type == 'mean':
        for i in range(len(ids)):
            histogram[i] = np.mean(activations_mask[correspondency == i])
    elif type == 'percent':
        for i in range(len(ids)):
            histogram[i] = np.sum(correspondency==i) / segmented_image.size
    elif type == 'activation':
        for i in range(len(ids)):
            histogram[i] = activations_mask
    else:
        raise ValueError('Valid types: max, sum and mean. Type '+str(type)+' is not valid')
    #normalized_hist = histogram/np.sum(histogram)
    return ids, histogram


def get_concept_selectivity_of_neuron(network_data, layer_name, neuron_idx, type='mean', concept='object', th = 0.1):
    """

    :param network_data:
    :param layer_name:
    :param neuron_idx:
    :param type:
    :param concept: 'object', 'part', 'material'
    :return:
    """
    neuron = network_data.get_neuron_of_layer(layer_name, neuron_idx)
    layer_data = network_data.get_layer_by_name(layer_name)
    receptive_field = layer_data.receptive_field_map

    image_names = neuron.images_id
    """
    Change it for the code to obtain the object and parts matrix
    """
    if network_data.dataset.src_segmentation_dataset is not None:
        full_image_names = [os.path.join(network_data.dataset.src_segmentation_dataset, image_name) for image_name in image_names]
        segmentation = [np.load(image_name+'.npz') for image_name in full_image_names]
    else:
        full_image_names = [os.path.join(network_data.dataset.src_dataset, image_name) for image_name in
                           image_names]
        from .util.segmentation.Broden_analize import Segment_images
        segmentation = Segment_images(full_image_names)

    if not type == 'activation':
        activations_masks = read_act.get_image_activation(network_data, image_names, layer_name, neuron_idx, type=1)
    """
    Definition as dictionary and not as numpy for don't have constants with sizes that can be mutables on time or between
    segmentators. Less efficient but more flexible (And the execution time of this for is short)
    """
    general_hist = {}
    norm_activations = neuron.norm_activations
    for i in range(len(segmentation)):
        #Crop for only use the receptive field
        ri, rf, ci, cf = receptive_field[neuron.xy_locations[i, 0], neuron.xy_locations[i, 1]]
        ri, rf, ci, cf = abs(ri), abs(rf), abs(ci), abs(cf)
        cropped_segmentation = segmentation[i][concept][ri:rf, ci:cf]
        activation = norm_activations[i] if type=='activation' else activations_masks[i][ri:rf, ci:cf]
        #Make individual hist
        ids, personal_hist = concept_selectivity_of_image(activations_mask=activation,
                                                          segmented_image=cropped_segmentation,
                                                          type=type)
        if not type == 'activation':
            personal_hist *= norm_activations[i]

        for id, value in zip(ids, personal_hist):
            if id in general_hist:
                general_hist[id] += value
            else:
                general_hist[id] = value
    #Dict to Structured Numpy
    general_hist = np.array(list(general_hist.items()), dtype = [('label', np.int), ('value',np.float)])
    #Ordering
    general_hist = np.sort(general_hist, order = 'value')[::-1]
    #Normalized
    general_hist['value'] /= np.sum(general_hist['value'])
    general_hist = general_hist[general_hist['value'] >= th]
    if len(general_hist) is 0:
        return np.array([('None', 0.0)], dtype = [('label', np.object), ('value',np.float)])
    else:
        general_hist['value'] = np.round(general_hist['value'],3)
        general_hist = translate_concept_hist(general_hist, concept)
        return general_hist



def translate_concept_hist(hist, concept):
    # Charge without index (redundant with pos) and without header
    translation = get_concept_labels(concept=concept)
    translated_hist = [(translation[element['label']],element['value']) for element in hist]
    return np.array(translated_hist, dtype=[('label', np.object), ('value', np.float)])

def get_concept_labels(concept='object'):
    concept = concept.lower()
    if concept == 'object':
        return np.genfromtxt(CONCEPT_TRANSLATION_BASE_DIR+concept+'.csv', delimiter=',', dtype=np.str)[1:,1]









def get_class_selectivity_idx(neuron_data, labels = None, threshold=.1, type=2):
    """Returns the class selectivity index value.

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param labels: Dictionary, key: name class, value: label class.
    :param threshold: Float.
    :param type: type=1 As Ivet defined, type=2 sum of classes that overpass the threshold
    :return: A tuple with: label class and class index value.
    """
    num_max_activations = len(neuron_data.activations)
    rel_freq = relative_freq_class(neuron_data, labels)

    if rel_freq is None:
        return np.array([("None", 0.0)],dtype=[('label','U64'),('value',np.float)])
    elif type == 1:
        freq_avoid_th = []
        sum_fr = 0.0
        for rel in rel_freq:
            # check the frequencies that fills the `threshold`.
            sum_fr += rel[REL_FREQ_POS]
            freq_avoid_th.append(rel)
            if sum_fr >= threshold:
                break

        m = len(freq_avoid_th)
        c_select_idx = (num_max_activations - m) / (float(num_max_activations - 1))
        # For the most freqüent class freq_avoid_th[0] the: label ([HUMAN_NAME_POS]) and his selectivity index rounded to three decimals
        return (freq_avoid_th[0][HUMAN_NAME_POS], round(c_select_idx, 3))
    elif type == 2:
        in_pc = rel_freq[rel_freq['value'] >= threshold]
        if len(in_pc) > 0:
            result = np.zeros(len(in_pc), dtype=[('label', 'U64'), ('value', np.float)])
            result['label'], result['value'] = in_pc['label'], in_pc['value']
            return result
        else:
            return np.array([("None", 0.0)],dtype=[('label','U64'),('value',np.float)])




def relative_freq_class(neuron_data, labels = None):
    """Calculates the relative frequencies of appearance of each class among
    the TOP scoring images from `neuron_data`.

    :param neuron_data: The 'nefesi.neuron_data.NeuronData' instance.
    :param labels: Dictionary, key: name class, value: human readable name class.

    :return: Numpy of slices. Each slice contains:
        - The name class 'label_name' (to access as pandas)
        - The human name class 'label' (to access as pandas)
        - Number of appearance in this neuron of this class among all classes. 'count' (to access as pandas)
        - The normalized relative frequency. 'value' (to access as pandas)
    """

    #If the max activation is 0 not continue
    if np.isclose(neuron_data.activations[0], 0.0):
        return None
        #return np.array([('n0000000', 'NoClass', 0, 0.0)])


    #------------------------INITS NEURON_DATA.TOP_LABELS IF NOT IS INITIALIZED---------------------------------
    if neuron_data.top_labels is None:
        _fill_top_labels(neuron_data)
    #-----------------------INITS THE PARAMETERS THAT WILL BE USEFUL TO MAKE CALCULS-----------------------------
    norm_act = neuron_data.norm_activations
    norm_activation_total = np.sum(norm_act)
    classes, classes_idx, classes_counts =  np.unique(neuron_data.top_labels, return_inverse=True, return_counts=True)
    rel_freq = np.zeros(len(classes),dtype=[('label_name','U64'),('label','U64'),('count',np.int), ('value',np.float)])

    #-------------------------------------------CALC THE INDEX-------------------------------------------------
    for label_idx in range(len(classes)):
        appearances_count = classes_counts[label_idx]
        # normalize the sum of the activations with the sum of
        # the whole normalized activations in this neuron.
        norm_activation_label_sum = np.sum(norm_act[classes_idx == label_idx])/norm_activation_total
        if labels is not None:
            rel_freq[label_idx] = (classes[label_idx], labels[classes[label_idx]], appearances_count, norm_activation_label_sum)
        else:
            rel_freq[label_idx] = (classes[label_idx], classes[label_idx], appearances_count, norm_activation_label_sum)

    # sorts the list by their relative frequencies of norm activations.
    rel_freq = np.sort(rel_freq, order='value')[::-1]
    return rel_freq



def get_population_code_idx(neuron_data, labels=None, threshold_pc=0.1):
    """Returns the population code index value

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param labels: Dictionary, key: name class, value: label class.
    :param threshold_pc: Float. Threshold to consider that neuron is well selective to a class

    :return: Integer, population code value. (Number of classes with frequency higher to threshold_pc in N-top activations
    """
    rel_freq = relative_freq_class(neuron_data, labels)
    if rel_freq is None:
        return 0
    else:
        #classes with relative  frequency more than threshold_pc
        return np.count_nonzero(rel_freq['value']>= threshold_pc)

def get_population_code_classes(neuron_data, labels=None, threshold_pc=0.1):
    """Returns the population code index value

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param labels: Dictionary, key: name class, value: label class.
    :param threshold_pc: Float. Threshold to consider that neuron is well selective to a class

    :return: Integer, population code value. (Number of classes with frequency higher to threshold_pc in N-top activations
    """
    rel_freq = relative_freq_class(neuron_data, labels)
    if rel_freq is None:
        return []
    else:
        #classes with relative  frequency more than threshold_pc
        pc = np.count_nonzero(rel_freq['value']>= threshold_pc)
        return rel_freq['label'][:pc]

def get_hierarchical_population_code_idx(neuron_data, xml='../nefesi/imagenet_structure.xml', threshold_pc=0.1,
                                         population_code=0, class_sel=0):
    """Returns the population code index value

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param labels: Dictionary, key: name class, value: label class.
    :param threshold_pc: Float. Threshold to consider that neuron is well selective to a class

    :return: Integer, population code value. (Number of classes with frequency higher to threshold_pc in N-top activations
    """
    rel_freq = relative_freq_class(neuron_data, None)
    if rel_freq is None:
        return Node('root', freq=0, rep=0)
    rel_freq = rel_freq[rel_freq['value'] >= threshold_pc]
    tree = gf.get_hierarchy_of_label(rel_freq['label_name'], rel_freq['value'], xml,population_code,class_sel)
    return tree


def get_ntop_population_code(neuron_data, labels=None, threshold_pc=0.1, n=5):
    rel_freq = relative_freq_class(neuron_data, labels)
    if rel_freq is None:
        return 0
    else:
        ntop = np.zeros(len(rel_freq),dtype=np.dtype([('label','U128'),('value',np.float)]))
        for i, (_,label,_,freq) in enumerate(rel_freq):
            if freq>threshold_pc:
                ntop[i] = (label,np.round(freq,decimals=3))
            else:
                ntop = ntop[:i]
                break
        # classes with relative  frequency more than threshold_pc
        return ntop

def get_entropy_idx(neuron_data,labels=None,base_log=2):

    rel_freq = relative_freq_class(neuron_data, labels)

    if rel_freq is None:
        return  0.0
    else:
        sum_entropy = 0.0
        for rel in rel_freq:
            sum_entropy += rel[REL_FREQ_POS]*log(1/rel[REL_FREQ_POS],base_log)

        # For the most freqüent class freq_avoid_th[0] the: label ([HUMAN_NAME_POS]) and his selectivity index rounded to two decimals
        return (sum_entropy)

def _fill_top_labels(neuron_data):
    """
    Fills the 'nefesi.neuron_data.NeuronData.top_labels' attribute
    :param neuron_data: The 'nefesi.neuron_data.NeuronData' instance.
    """
    image_names = neuron_data.images_id
    neuron_data.top_labels = np.zeros(len(image_names), dtype='U64')
    path_sep = get_path_sep(image_names[0])
    #Compatibility between Linux/Windows/Mac Os filenames
    for i, image_name in enumerate(image_names):
        neuron_data.top_labels[i] = image_name[:image_name.index(path_sep)]

def get_path_sep(image_name):
    path_sep = os.path.sep
    if image_name.find(os.path.sep) < 0:
        if os.path.sep == '\\':
            path_sep = '/'
        else:
            path_sep = '\\'
    return path_sep
    return path_sep