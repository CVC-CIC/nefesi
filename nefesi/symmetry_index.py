import os
import pickle

import read_activations
from util.image import rotate_images_axis


def get_symmetry_index(filter, model, layer, idx_neuron, dataset):

    results = []
    symm_axes = [0, 45, 90, 135]
    avg_symmetry_idx = 0

    activations = filter.get_activations()
    norm_activations = filter.get_norm_activations()
    image_names = filter.get_images_id()
    locations = filter.get_locations()

    max_act = activations[0]

    if max_act != 0.0:

        images = dataset.load_images(image_names)

        for axes in symm_axes:
            images_r = rotate_images_axis(images, axes, model, layer, locations)

            rot_activations = read_activations.get_activation_from_pos(images_r, model, layer, idx_neuron, locations)


            # print axes, rot_activations
            norm_rot_act = rot_activations/max_act
            partial_symmetry = sum(norm_rot_act)/sum(norm_activations)

            avg_symmetry_idx = avg_symmetry_idx + partial_symmetry

            results.append(partial_symmetry)


        avg_symmetry_idx = avg_symmetry_idx/len(symm_axes)

        results.append(avg_symmetry_idx)
        return results

    else:
        return None





if __name__=='__main__':
    from external.vgg_matconvnet import VGG

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = VGG()
    layer_name = 'activation_1'
    filters = pickle.load(open(layer_name + '.obj', 'rb'))

    # fil = open('symmetry_idx_' + layer_name + '.txt', 'wb')

    for i in xrange(1):
        print i
        res = get_symmetry_index(filters[i], model, layer_name, i)
        print res
    #     fil.write(str(i) + '\t')
    #     for r in res:
    #         fil.write(str(r) + '\t')
    #     fil.write('\n')
    # fil.close()


    # filters[0].print_params()



    #
    # img = filters[0].get_activations()[0][0]
    # pos = filters[0].get_activations()[2]
    # img = load_images([img])
    # #
    # # image.array_to_img(img[0], scale=False).show()
    # #
    # x, y = pos[0]
    # # print x, y
    # # act = get_activations(model, img, print_shape_only=True, layer_name=layer_name)[0]
    # # print act[0, x, y, 0]
    # #
    # img_rot = rotate_images(img, 45, model, layer_name, pos)
    #
    #
    # image.array_to_img(img_rot[0], scale=False).show()
    # #
    #
    #
    # act = get_activations(model, img_rot, print_shape_only=True, layer_name=layer_name)[0]
    # print act[0, x, y, 0]