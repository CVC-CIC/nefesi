import numpy as np
import sys

sys.path.append('..')
from functions import read_activations
from functions.image import rgb2opp, image2max_gray

NUM_THREADS = 10







def color_selectivity_of_image(activations_mask, color_named_image, type='mean'): #activations_mask
    """
    :param type: 'max' = only the max of every region, 'sum' sum of the pixels of a region, 'mean' the mean of the activation
    in each region, 'percent' the plain percent, 'activation' the max activation of the n-topScoring.
    :return:
    """
    if not (type == 'activation' or type == 'percent'):
        activations_mask = activations_mask.reshape(-1)
    ids, correspondency = np.unique(color_named_image, return_inverse=True)
    if type == 'mean':
        histogram = np.array([np.mean(activations_mask[correspondency == i]) for i in range(len(ids))])
    elif type == 'sum':
        histogram = np.array([np.sum(activations_mask[correspondency == i]) for i in range(len(ids))])
    elif type == 'max':
        histogram = np.array([np.max(activations_mask[correspondency == i]) for i in range(len(ids))])
    elif type == 'percent':
        histogram = np.array([np.sum(correspondency==i) / color_named_image.size for i in range(len(ids))])
    elif type == 'activation':
        histogram = np.array([activations_mask for i in range(len(ids))])
    #normalized_hist = histogram/np.sum(histogram)
    return ids, histogram

def get_ivet_color_selectivity_index(neuron_data, model, layer_data, dataset, type='no-ivet'):
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
        idx_neuron = np.where(layer_data.neurons_data == neuron_data)[0]
        images_gray = np.ndarray(shape=images.shape, dtype=images.dtype)
        for i in range(len(images)):
            # get the receptive field from the origin image.
            x, y = locations[i]
            #a_max=None, because used as slices overflow at the end is the same that clip to the end
            row_ini, row_fin, col_ini, col_fin = np.clip(layer_data.receptive_field_map[x, y],a_min =0, a_max=None)
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
        new_activations = np.sort(new_activations)[::-1]

        if type=='ivet':
            norm_gray_activations_sum = np.sum(new_activations) / max_rgb_activation
            return 1 - (norm_gray_activations_sum / np.sum(norm_activations))
        else:
            new_norm_activations = new_activations / np.abs(max_rgb_activation)
            gray_activations = np.minimum(1, new_norm_activations / norm_activations)
            return np.mean(1 - np.maximum(0, gray_activations))
    else:
        return 0.0


def get_color_selectivity_index_new(neuron_data, model, layer_data, dataset):
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
        images_gray = dataset.load_images(image_names,color_mode='grayscale')
        idx_neuron = layer_data.neurons_data.index(neuron_data)

        new_activations = read_activations.get_activation_from_pos(images_gray, model,
                                                                   layer_data.layer_id,
                                                                   idx_neuron, locations)
        new_activations = np.sort(new_activations)[::-1]


        norm_gray_activations_sum = np.sum(new_activations) / max_rgb_activation
        return 1 - norm_gray_activations_sum/np.sum(norm_activations)

    else:
        return 0.0



def get_shape_selectivity_index(neuron_data, model, layer_data, dataset):
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
    max_activation = activations[0]

    if max_activation != 0.0:
        images = dataset.load_images(image_names, prep_function=False)
        idx_neuron = np.where(layer_data.neurons_data == neuron_data)[0][0]
        images_gray = np.ndarray(shape=images.shape, dtype=images.dtype)
        for i in range(len(images)):
            # get the receptive field from the origin image.
            x, y = locations[i]
            # a_max=None, because used as slices overflow at the end is the same that clip to the end
            row_ini, row_fin, col_ini, col_fin = np.clip(layer_data.receptive_field_map[x, y], a_min=0, a_max=None)
            init_image = images[i]
            img_crop = init_image[row_ini:row_fin, col_ini:col_fin]

            # image transformation functions.
            im_opp = img_crop


            rndImg2 = np.reshape(im_opp, (im_opp.shape[0] * im_opp.shape[1], im_opp.shape[2]))
            # this like could also be written using -1 in the shape tuple
            # this will calculate one dimension automatically
            # rndImg2 = np.reshape(rgbImg, (-1, rgbImg.shape[2]))

            # now shuffle
            np.random.shuffle(rndImg2)

            # and reshape to original shape
            rdmImg = np.reshape(rndImg2, im_opp.shape)
            # plt.subplot(2,1,1)
            # plt.imshow(im_opp)
            # plt.subplot(2, 1, 2)
            # plt.imshow(rdmImg)
            # plt.show()

            init_image[row_ini:row_fin, col_ini:col_fin] = rdmImg
            images_gray[i] = init_image





        # once the images have been converted to grayscale,
        # apply the preprocessing function, if exist.
        if dataset.preprocessing_function != None:
            images_gray = dataset.preprocessing_function(images_gray)  # np.asarray(images_gray))
        new_activations = read_activations.get_activation_from_pos(images_gray, model,
                                                                   layer_data.layer_id,
                                                                   idx_neuron, locations)
        new_activations = np.sort(new_activations)[::-1]

        if type == 'ivet':
            norm_gray_activations_sum = np.sum(new_activations) / max_activation
            return 1 - (norm_gray_activations_sum / np.sum(norm_activations))
        else:
            new_norm_activations = new_activations / np.abs(max_activation)
            gray_activations = np.minimum(1, new_norm_activations / norm_activations)
            return np.mean(1 - np.maximum(0, gray_activations))
    else:
        return 0.0