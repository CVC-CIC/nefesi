import os
import pickle

import read_activations
from util.image import rotate_images


def get_orientation_index(filter, model, layer, idx_neuron, dataset, degrees=None, n_rotations=None):

    if degrees is None:
        degrees = 15
    if n_rotations is None:
        n_rotations = 25
    l_degrees = [x*degrees for x in xrange(1, n_rotations)]  # avoid 0 degrees

    results = []

    avg_orientation_index = 0

    activations = filter.get_activations()
    norm_activations = filter.get_norm_activations()
    image_names = filter.get_images_id()
    locations = filter.get_locations()
    max_act = activations[0]

    if max_act != 0.0:
        images = dataset.load_images(image_names)

        for degrees in l_degrees:
            images_r = rotate_images(model, images, degrees, locations, layer)
            rot_activations = read_activations.get_activation_from_pos(images_r, model, layer, idx_neuron, locations)
            # print axes, rot_activations
            norm_rot_act = rot_activations/max_act

            partial_orientation = 1 - (sum(norm_rot_act)/sum(norm_activations))
            # print axes, partial_symmetry

            avg_orientation_index = avg_orientation_index + partial_orientation

            results.append(partial_orientation)

        avg_orientation_index = avg_orientation_index/len(l_degrees)

        results.append(avg_orientation_index)
        return results
    else:
        return None







if __name__=='__main__':
    from external.vgg_matconvnet import VGG

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = VGG()
    layer_name = 'activation_1'
    filters = pickle.load(open(layer_name + '.obj', 'rb'))

    fil = open('aux_orientation_idx_' + layer_name + '.txt', 'wb')

    for i in xrange(len(filters)):
        print i
        res = get_orientation_index(filters[i], model, layer_name, i)
        fil.write(str(i) + '\t')
        for r in res:
            # print r
            fil.write(str(r) + '\t')
        fil.write('\n')
    # print res
    fil.close()


    # filters[0].print_params()