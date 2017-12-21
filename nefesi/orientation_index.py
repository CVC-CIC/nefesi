import os
import pickle

import numpy as np
from keras.preprocessing import image
from scipy.ndimage.interpolation import rotate

import read_activations
from nefesi.neuron_feature import get_image_receptive_field



def crop_image(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def rotate_images(model, images, degrees, pos, layer):
    images_rotated = []
    for i in xrange(len(images)):
        init_image = np.copy(images[i])
        x, y = pos[i]
        print x, y
        row_ini, row_fin, col_ini, col_fin = get_image_receptive_field(x, y, model, layer)
        receptive_field = init_image[row_ini:row_fin+1, col_ini:col_fin+1]
        w, h, d = receptive_field.shape
        print w, h, d
        padding_w = int(round(w/2.))
        padding_h = int(round(h/2.))

        new_shape = np.zeros((w + padding_w, h + padding_h, d), dtype=receptive_field.dtype)
        print new_shape.shape

        for dim in xrange(d):
            new_shape[:,:,dim] = np.pad(receptive_field[:,:,dim], ((padding_w/2, padding_w/2),(padding_h/2, padding_h/2)), mode='edge')

        img = rotate(new_shape, degrees, reshape=False)
        img = crop_image(img, h, w)
        init_image[row_ini:row_fin + 1, col_ini:col_fin + 1] = img

        images_rotated.append(init_image)


    return images_rotated



def load_images(image_names, dataset_path):
    images = []
    for n in image_names:
        i = image.load_img(dataset_path + n, target_size=(224, 224))
        i = image.img_to_array(i)
        i = i - avg_img
        images.append(i)

    # i = image.array_to_img(images[0], scale=False)
    # i.save('origin.png')
    return images

def get_orientation_index(filter, model, layer, idx_neuron, dataset_path, degrees=None, n_rotations=None):

    # degrees = 15
    # n_rotations = 25
    if degrees is None:
        degrees = 15
    if n_rotations is None:
        n_rotations = 25
    l_degrees = [x*degrees for x in xrange(1, n_rotations)]  # avoid 0 degrees

    print l_degrees


    results = []

    print results
    avg_orientation_index = 0

    activations = filter.get_activations()
    norm_activations = filter.get_norm_activations()
    image_names = filter.get_images_id()
    locations = filter.get_locations()
    max_act = activations[0]

    if max_act != 0.0:
        images = load_images(image_names, dataset_path)

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
