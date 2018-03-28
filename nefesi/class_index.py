from operator import itemgetter
import numpy as np


def get_class_selectivity_idx(filter, labels, threshold):

    if labels is None:
        raise TypeError('labels')


    num_max_activations = len(filter.activations)

    rel_freq = relative_freq_class(filter, labels)

    if rel_freq is None:
        print 'Msg'
        return None

    else:
        freq_avoid_th = []
        sum_fr = 0.0
        for rel in rel_freq:
            sum_fr += rel[3]
            freq_avoid_th.append(rel)
            # print sum_fr
            if sum_fr >= threshold:
                break

        m = len(freq_avoid_th)

        c_select_idx = (num_max_activations-m)/(float(num_max_activations-1))

        print freq_avoid_th
        return freq_avoid_th[0][1], round(c_select_idx, 2)



def relative_freq_class(filter, labels):


    activations = filter.activations
    images = filter.images_id
    norm_act = filter.norm_activations

    rel_freq = []

    if activations[0] != 0.0:
        for k, v in labels.items():
            i = 0
            a = 0
            for c in xrange(len(images)):
                if k in images[c]:
                    i += 1
                    a += norm_act[c]

            if i != 0:
                rel_freq.append([k, v, i, a])

        for rel in rel_freq:
            rel[3] = rel[3] / sum(norm_act)

        rel_freq = sorted(rel_freq, key=itemgetter(3), reverse=True)

        print rel_freq
        print len(rel_freq)
        return rel_freq
    else:
        return None


def get_population_code_idx(filter, labels, threshold_pc):
    if labels is None:
        print 'Message error in get_population'
        return None

    rel_freq = relative_freq_class(filter, labels)

    if rel_freq is None:
        print 'Msg'
        return None

    else:
        pc = 0
        for r in rel_freq:
            if r[3] >= threshold_pc:
                pc += 1
            else:
                break

        return pc


if __name__=='__main__':
    import pickle

    labels = pickle.load(open('external/labels_imagenet.obj', 'rb'))
    print labels

    my_net = pickle.load(open('/home/oprades/oscar/block1_3.obj', 'rb'))
    l1 = my_net.get_layers()[6]


    neuron = l1.filters[96]
    # neuron.print_params()
    # neuron.get_neuron_feature().show()

    t = get_class_selectivity_idx(neuron, labels, 1.)
    print t

    print get_population_code_idx(neuron, labels, 0.1)

    r = my_net.get_selectivity_idx('population code', 'fff')
    print r
    # l1.selectivity_idx(my_net.model, 'population code', my_net.dataset)
    # neuron.class_selectivity_idx()

#     import os
#     from vgg_matconvnet import VGG
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
#     labels = dict()
#     fil = open('map_clsloc.txt', 'rb')
#     for line in fil:
#         tmp = line.split(' ')
#         labels[tmp[0]] = tmp[2].rstrip()
#
#     # print labels
#     fil.close()
#
#     import pickle
#     pickle.dump(labels, open('labels_imagenet.obj', 'wb'))

