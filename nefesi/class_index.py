from operator import itemgetter


def get_class_selectivity_idx(neuron_data, labels, threshold):
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
            sum_fr += rel[3]
            freq_avoid_th.append(rel)
            if sum_fr >= threshold:
                break

        m = len(freq_avoid_th)
        c_select_idx = (num_max_activations - m) / (float(num_max_activations - 1))
        return freq_avoid_th[0][1], round(c_select_idx, 2)


def relative_freq_class(neuron_data, labels):
    """Calculates the relative frequencies of appearance of each class among
    the TOP scoring images from `neuron_data`.

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param labels: Dictionary, key: name class, value: label class.

    :return: List of lists. Each list contains:
        - The name class
        - The label class
        - Number of appearance in this neuron of this class among all classes.
        - The normalized relative frequency.
    """
    activations = neuron_data.activations
    images = neuron_data.images_id
    norm_act = neuron_data.norm_activations

    rel_freq = []
    if activations[0] != 0.0:
        for k, v in labels.items():
            i = 0
            a = 0
            for c in xrange(len(images)):
                # counts the number of times that a class appears
                # among the TOP scoring images in a neuron.
                # Also keeps a sum of normalized activations of that image
                # that belongs to a class.
                if k in images[c]:
                    i += 1
                    a += norm_act[c]
            if i != 0:
                rel_freq.append([k, v, i, a])

        # normalize the sum of the activations with the sum of
        # the whole normalized activations in this neuron.
        for rel in rel_freq:
            rel[3] = rel[3] / sum(norm_act)

        # sorts the list by their relative frequencies.
        rel_freq = sorted(rel_freq, key=itemgetter(3), reverse=True)
        return rel_freq
    else:
        return None


def get_population_code_idx(neuron_data, labels, threshold_pc):
    """Returns the population code index value

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param labels: Dictionary, key: name class, value: label class.
    :param threshold_pc: Float.

    :return: Integer, population code value
    """
    rel_freq = relative_freq_class(neuron_data, labels)
    if rel_freq is None:
        return 0
    else:
        pc = 0
        # count the number of classes above `threshold_pc`
        for r in rel_freq:
            if r[3] >= threshold_pc:
                pc += 1
            else:
                break
        return pc
