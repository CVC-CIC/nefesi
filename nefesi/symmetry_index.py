import os
import pickle

import numpy as np
from keras.preprocessing import image

import read_activations
from nefesi.neuron_feature import get_image_receptive_field


def rotate_images(images, rot_axis, model, layer, pos):

    rot_images = []
    for i in xrange(len(images)):
        init_image = np.copy(images[i])
        x, y = pos[i]
        row_ini, row_fin, col_ini, col_fin = get_image_receptive_field(x, y, model, layer)
        receptive_field = init_image[row_ini:row_fin+1, col_ini:col_fin+1]

        rf_shape = receptive_field.shape

        rotated_receptive_field = rotate_rf(receptive_field, rot_axis)

        rot_rf_shape = rotated_receptive_field.shape

        if rf_shape != rot_rf_shape:
            row_fin = row_fin + (-(rf_shape[0]-rot_rf_shape[0]))
            col_fin = col_fin + (-(rf_shape[1]-rot_rf_shape[1]))

        init_image[row_ini:row_fin + 1, col_ini:col_fin + 1] = rotated_receptive_field
        rot_images.append(init_image)

    return rot_images


def rotate_rf(img, rot_axis):
    if rot_axis == 0:
        return np.flipud(img)

    if rot_axis == 45:
        n_image = img.transpose(1, 0, 2)
        n_image = np.flipud(n_image)
        return np.fliplr(n_image)

    if rot_axis == 90:
        return np.fliplr(img)

    if rot_axis == 135:
        return img.transpose(1, 0, 2)

    return None



def load_images(image_names, dataset_path):
    images = []
    for n in image_names:
        i = image.load_img(dataset_path + n, target_size=(224, 224))
        i = i - avg_img
        i = image.img_to_array(i)
        images.append(i)

    # i = image.array_to_img(images[0], scale=False)
    # i.save('origin.png')
    return images


def get_symmetry_index(filter, model, layer, idx_neuron, dataset_path):

    results = []
    symm_axes = [0, 45, 90, 135]
    avg_symmetry_idx = 0

    activations = filter.get_activations()
    norm_activations = filter.get_norm_activations()
    image_names = filter.get_images_id()
    locations = filter.get_locations()

    max_act = activations[0]

    if max_act != 0.0:

        images = load_images(image_names, dataset_path)

        for axes in symm_axes:
            images_r = rotate_images(images, axes, model, layer, locations)

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