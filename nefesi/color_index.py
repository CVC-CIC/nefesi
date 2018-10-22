import numpy as np
import sys
sys.path.append('..')
from nefesi import read_activations
from nefesi.util.image import rgb2opp, image2max_gray


def get_color_selectivity_index(neuron_data, model, layer_data, dataset, type='no-ivet'):
    """Returns the color selectivity index for a neuron (`neuron_data`).

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param model: The `keras.models.Model` instance.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    :param dataset: The `nefesi.util.image.ImageDataset` instance.
    :param type: How to calculate color index: Index defined in Ivet Rafegas thesis ('ivet') or
    controlling index between [0,1] (else)

    :return: Float, the color selectivity index value.
    """
    activations = neuron_data.activations
    norm_activations = neuron_data.norm_activations
    image_names = neuron_data.images_id
    locations = neuron_data.xy_locations
    max_rgb_activation = activations[0]

    if max_rgb_activation != 0.0:
        images = dataset.load_images(image_names, prep_function=False)
        idx_neuron = np.where(layer_data.neurons_data == neuron_data)[0][0]
        images_gray = np.ndarray(shape=images.shape, dtype=images.dtype)
        for i in range(len(images)):
            # get the receptive field from the origin image.
            x, y = locations[i]
            row_ini, row_fin, col_ini, col_fin = layer_data.receptive_field_map[x, y]
            init_image = images[i]
            img_crop = init_image[row_ini:row_fin, col_ini:col_fin]

            # image transformation functions.
            im_opp = rgb2opp(img_crop)
            im_gray = image2max_gray(im_opp)
            init_image[row_ini:row_fin, col_ini:col_fin] = im_gray
            images_gray[i] = init_image

        # once the images have been converted to grayscale,
        # apply the preprocessing function, if exist.
        if dataset.preprocessing_function != None:
            images_gray = dataset.preprocessing_function(images_gray)#np.asarray(images_gray))
        new_activations = read_activations.get_activation_from_pos(images_gray, model,
                                                                   layer_data.layer_id,
                                                                   idx_neuron, locations)
        new_activations.sort()
        new_activations = new_activations / np.abs(max_rgb_activation)

        if type=='ivet':
<<<<<<< HEAD
            norm_gray_activations = new_activations / max_rgb_activation
            return 1 - (np.sum(norm_gray_activations) / np.sum(norm_activations))
        elif type=='opcio1':
=======
            norm_gray_activations_sum = np.sum(new_activations) / max_rgb_activation
            return 1 - (norm_gray_activations_sum / np.sum(norm_activations))
        else:
>>>>>>> fb6a3ecb602fd54815a9e4c3832d001c37bdfbd6
            gray_activations = np.minimum(1, new_activations / activations)
            return np.mean(1 - np.maximum(0, gray_activations))
        else:

            norm_gray_activations = new_activations / max_rgb_activation
            return 1 - (np.sum(norm_gray_activations) / np.sum(norm_activations))

    else:
        return 0.0