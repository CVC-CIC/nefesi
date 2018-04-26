import numpy as np

import read_activations
from utils.image import rgb2opp, image2max_gray


def get_color_selectivity_index(neuron_data, model, layer_data, dataset):
    """Returns the color selectivity index for a neuron (`neuron_data`).

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param model: The `keras.models.Model` instance.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    :param dataset: The `nefesi.utils.image.ImageDataset` instance.

    :return: Float, the color selectivity index value.
    """
    activations = neuron_data.activations
    norm_activations = neuron_data.norm_activations
    image_names = neuron_data.images_id
    locations = neuron_data.xy_locations
    max_rgb_activation = activations[0]

    if max_rgb_activation != 0.0:
        images = dataset.load_images(image_names, prep_function=False)
        idx_neuron = layer_data.neurons_data.index(neuron_data)

        images_gray = []
        for i in xrange(len(images)):
            # get the receptive field from the origin image.
            x, y = locations[i]
            row_ini, row_fin, col_ini, col_fin = layer_data.receptive_field_map[x, y]
            init_image = images[i]
            img_crop = init_image[row_ini:row_fin, col_ini:col_fin]

            # image transformation functions.
            im_opp = rgb2opp(img_crop)
            im_gray = image2max_gray(im_opp)
            init_image[row_ini:row_fin, col_ini:col_fin] = im_gray
            images_gray.append(init_image)

        # once the images have been converted to grayscale,
        # apply the preprocessing function.
        images_gray = dataset.preprocessing_function(np.asarray(images_gray))
        new_activations = read_activations.get_activation_from_pos(images_gray, model,
                                                                   layer_data.layer_id,
                                                                   idx_neuron, locations)
        norm_gray_activations = new_activations / max_rgb_activation
        return 1 - (sum(norm_gray_activations) / sum(norm_activations))
    else:
        return 0.0
