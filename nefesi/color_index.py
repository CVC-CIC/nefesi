import os
import pickle

import numpy as np
from keras.preprocessing import image

import read_activations
from neuron_feature import get_image_receptive_field



def load_images(dataset_path, image_names):
    images = []
    for n in image_names:
        i = image.load_img(dataset_path + n, target_size=(224, 224))
        i = image.img_to_array(i)
        # i -= avg_img
        images.append(i)

    # i = image.array_to_img(images[0], scale=False)
    # i.save('origin.png')
    return images

def rgb2opp(img):
    """Converts an image from RGB space to OPP (Opponent color space).

    :param img: Numpy array
    :return: Numpy array
    """

    # x = image.img_to_array(img)
    opp = np.zeros(shape=img.shape)

    x = img / 255.
    R = x[:, :, 0]
    G = x[:, :, 1]
    B = x[:, :, 2]
    opp[:, :, 0] = (R + G + B - 1.5) / 1.5
    opp[:, :, 1] = (R - G)
    opp[:, :, 2] = (R + G - 2 * B) / 2

    return opp

def image2max_gray(img):
    """Converts an image to gray scale using the PCA in order to maximally
     preserve the shape pattern of the image.

    :param img: Numpy array
    :return: PIL instance (gray scale image in RGB space)
    """
    x = img.reshape(-1, 3)

    M = (x - np.mean(x, axis=0))

    latent, coeff = np.linalg.eig(np.cov(M.T))
    res = np.dot(M, coeff[:, 0])

    res = res[:, np.newaxis]
    res = np.tile(res, (1, 1, 3))
    res = np.reshape(res, img.shape)

    res = image.array_to_img(res, scale=True)

    return res


def get_color_selectivity_index(filter, model, layer, idx_neuron, dataset_path):

    activations = filter.get_activations()
    # filter.print_params()
    norm_activations = filter.get_norm_activations()
    image_names = filter.get_images_id()
    locations = filter.get_locations()
    max_rgb_activation = activations[0]



    if max_rgb_activation != 0.0:

        images = load_images(dataset_path, image_names)
        images_gray = []

        for i in xrange(len(images)):
            x, y = locations[i]
            row_ini, row_fin, col_ini, col_fin = get_image_receptive_field(x, y, model, layer)
            init_image = images[i]
            img_crop = init_image[row_ini:row_fin+1, col_ini:col_fin+1]


            im_opp = rgb2opp(img_crop)
            im_gray = image2max_gray(im_opp)

            init_image[row_ini:row_fin + 1, col_ini:col_fin + 1] = im_gray

            # image.array_to_img(init_image, scale=False).show()
            init_image -= avg_img
            images_gray.append(init_image)

        new_activations = read_activations.get_activation_from_pos(images_gray, model, layer, idx_neuron, locations)
        # new_activations.print_params()
        norm_gray_activations = new_activations/max_rgb_activation

        print sum(norm_activations)
        print sum(norm_gray_activations)
        return 1 - (sum(norm_gray_activations)/sum(norm_activations))

    else:
        return None



if __name__ == '__main__':
    from external.vgg_matconvnet import VGG

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = VGG()
    # model.summary()

    filters = pickle.load(open('activation_1.obj', 'rb'))
    layer_name = 'activation_1'

    fil = open('aux_color_idx_' + layer_name + '.txt', 'wb')


    for i in xrange(len(filters)):
        res = get_color_selectivity_index(filters[i], model, layer_name, i)
        fil.write(str(i) + '\t' + str(res) + '\n')

    fil.close()

    # img = image.load_img(PATH_FILE+filters[0].get_activations()[0][0], target_size=(224, 224))
    # img.show()
    #
    # x,y = filters[0].get_activations()[2][0]
    # row_ini, row_fin, col_ini, col_fin = get_image_receptive_field(x, y, model, 'activation_1')
    #
    # img2 = image.img_to_array(img)
    # opp = rgb2opp(img2)
    # opp_gray = image2max_gray(opp)
    # opp_gray.show()
    #
    # img_crop = img2[row_ini:row_fin+1, col_ini:col_fin+1]
    # opp = rgb2opp(img_crop)
    # opp_gray = image2max_gray(opp)
    # opp_gray.show()
    #
    # img2[row_ini:row_fin+1, col_ini:col_fin+1] = opp_gray
    # image.array_to_img(img2, scale=False).show()