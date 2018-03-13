from operator import itemgetter


def get_class_selectivity_idx(filter, labels=None, threshold=1.):

    if labels is None:
        print 'Message error'
        return None

    activations = filter.get_activations()
    images = filter.get_images_id()
    norm_act = filter.get_norm_activations()
    num_max_activations = len(activations)

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

        # for f in rel_freq:
        #     print f


        for rel in rel_freq:
            rel[3] = rel[3]/sum(norm_act)

        rel_freq = sorted(rel_freq, key=itemgetter(3), reverse=True)

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

        return freq_avoid_th[0][1], round(c_select_idx, 2)

    else:
        return None



if __name__=='__main__':
    import pickle

    labels = pickle.load(open('external/labels_imagenet.obj', 'rb'))
    print labels

    my_net = pickle.load(open('/home/oprades/oscar/block1_3.obj', 'rb'))
    l1 = my_net.get_layers()[6]


    neuron = l1.get_filters()[96]
    # neuron.print_params()
    # neuron.get_neuron_feature().show()

    t = get_class_selectivity_idx(neuron, labels=labels)
    print t

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

