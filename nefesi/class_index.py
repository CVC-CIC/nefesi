from operator import itemgetter
import numpy as np
import os

LABEL_NAME_POS = 0
HUMAN_NAME_POS = 1
COUNT_POS = 2
REL_FREQ_POS = 3

def get_class_selectivity_idx(neuron_data, labels = None, threshold=1.):
    """Returns the class selectivity index value.

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param labels: Dictionary, key: name class, value: label class.
    :param threshold: Float.

    :return: A tuple with: label class and class index value.
    """
    num_max_activations = len(neuron_data.activations)
    rel_freq = relative_freq_class(neuron_data, labels)

    if rel_freq is None:
        return None, 0.0
    else:
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
        # For the most freqüent class freq_avoid_th[0] the: label ([HUMAN_NAME_POS]) and his selectivity index rounded to two decimals
        return (freq_avoid_th[0][HUMAN_NAME_POS], round(c_select_idx, 3))


def relative_freq_class(neuron_data, labels = None):
    """Calculates the relative frequencies of appearance of each class among
    the TOP scoring images from `neuron_data`.

    :param neuron_data: The 'nefesi.neuron_data.NeuronData' instance.
    :param labels: Dictionary, key: name class, value: human readable name class.

    :return: Numpy of slices. Each slice contains:
        - The name class 'label_name' (to access as pandas)
        - The human name class 'human_name' (to access as pandas)
        - Number of appearance in this neuron of this class among all classes. 'count' (to access as pandas)
        - The normalized relative frequency. 'rel_freq' (to access as pandas)
    """

    #If the max activation is 0 not continue
    if np.isclose(neuron_data.activations[0], 0.0):
        return None

    #------------------------INITS NEURON_DATA.TOP_LABELS IF NOT IS INITIALIZED---------------------------------
    if neuron_data.top_labels is None:
        _fill_top_labels(neuron_data)
    #-----------------------INITS THE PARAMETERS THAT WILL BE USEFUL TO MAKE CALCULS-----------------------------
    norm_act = neuron_data.norm_activations
    norm_activation_total = np.sum(norm_act)
    classes, classes_idx, classes_counts =  np.unique(neuron_data.top_labels, return_inverse=True, return_counts=True)
    rel_freq = np.zeros(len(classes),dtype=[('label_name','U64'),('human_name','U64'),('count',np.int), ('rel_freq',np.float)])

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
    rel_freq = np.sort(rel_freq, order='rel_freq')[::-1]
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
        return np.count_nonzero(rel_freq['rel_freq']>= threshold_pc)

def get_ntop_population_code(neuron_data, labels=None, threshold_pc=0.1, n=5):
    rel_freq = relative_freq_class(neuron_data, labels)
    if rel_freq is None:
        return 0
    else:
        ntop = np.zeros(len(rel_freq),dtype=np.dtype([('label','U128'),('rel_freq',np.float)]))
        for i, (_,label,_,freq) in enumerate(rel_freq):
            if freq>threshold_pc:
                ntop[i] = (label,np.round(freq,decimals=3))
            else:
                ntop = ntop[:i]
                break
        # classes with relative  frequency more than threshold_pc
        return ntop


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