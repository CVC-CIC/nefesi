import numpy as np
import sys
from . import read_activations as read_act
from .util import ColorNaming as cn
sys.path.append('..')
from nefesi import read_activations
from nefesi.util.image import rgb2opp, image2max_gray






def get_color_selectivity_index(network_data, layer_name, neuron_idx,  type='mean', th = 0.1, activations_masks=None):

    neuron = network_data.get_neuron_of_layer(layer_name, neuron_idx)
    layer_data = network_data.get_layer_by_name(layer_name)
    receptive_field = layer_data.receptive_field_map

    image_names = neuron.images_id

    """
    Change it for the code to obtain the object and parts matrix
    """

    images = network_data.dataset.load_images(image_names, prep_function=False)


    if not type == 'activation' and activations_masks is None:
        complex_type = len(np.unique(receptive_field)) > 2
        activations_masks = read_act.get_image_activation(network_data, image_names, layer_name, neuron_idx,
                                                          complex_type=complex_type)
    """
    Definition as dictionary and not as numpy for don't have constants with sizes that can be mutables on time or between
    segmentators. Less efficient but more flexible (And the execution time of this for is short)
    """
    general_hist = {}
    norm_activations = neuron.norm_activations
    for i, image in enumerate(images):
        #Crop for only use the receptive field
        ri, rf, ci, cf = receptive_field[neuron.xy_locations[i, 0], neuron.xy_locations[i, 1]]
        ri, rf, ci, cf = abs(ri), abs(rf), abs(ci), abs(cf)
        cropped_image= image[ri:rf, ci:cf]
        activation = norm_activations[i] if type=='activation' else activations_masks[i][ri:rf, ci:cf]
        color_named_image = np.argmax(cn.ImColorNamingTSELabDescriptor(cropped_image), axis=-1)
        #Make individual hist
        ids, personal_hist = color_selectivity_of_image(activations_mask=activation,
                                                          color_named_image=color_named_image,
                                                          type=type)
        if not type == 'activation':
            personal_hist *= norm_activations[i]

        for id, value in zip(ids, personal_hist):
            id = cn.colors[id]
            if id in general_hist:
                general_hist[id] += value
            else:
                general_hist[id] = value
    #Dict to Structured Numpy
    general_hist = np.array(list(general_hist.items()), dtype = [('label', '<U64'), ('value',np.float)])
    #Ordering
    general_hist = np.sort(general_hist, order = 'value')[::-1]
    #Normalized
    general_hist['value'] /= np.sum(general_hist['value'])
    general_hist = general_hist[general_hist['value'] >= th]
    if len(general_hist) is 0:
        return np.array([('None', 0.0)], dtype = [('label', np.object), ('value',np.float)])
    else:
        general_hist['value'] = np.round(general_hist['value'],3)
        return general_hist


def color_selectivity_of_image(activations_mask, color_named_image, type='mean'): #activations_mask
    """
    :param type: 'max' = only the max of every region, 'sum' sum of the pixels of a region, 'mean' the mean of the activation
    in each region, 'percent' the plain percent, 'activation' the max activation of the n-topScoring.
    :return:
    """
    if not (type == 'activation' or type == 'percent'):
        activations_mask = activations_mask.reshape(-1)
    ids, correspondency = np.unique(color_named_image, return_inverse=True)
    histogram = np.zeros(shape=len(ids), dtype=np.float)
    if type == 'max':
        for i in range(len(ids)):
            histogram[i] = np.max(activations_mask[correspondency == i])
    elif type == 'sum':
        for i in range(len(ids)):
            histogram[i] = np.sum(activations_mask[correspondency == i])
    elif type == 'mean':
        for i in range(len(ids)):
            histogram[i] = np.mean(activations_mask[correspondency == i])
    elif type == 'percent':
        for i in range(len(ids)):
            histogram[i] = np.sum(correspondency==i) / color_named_image.size
    elif type == 'activation':
        for i in range(len(ids)):
            histogram[i] = activations_mask
    else:
        raise ValueError('Valid types: max, sum and mean. Type '+str(type)+' is not valid')
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
        idx_neuron = np.where(layer_data.neurons_data == neuron_data)[0][0]
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
            images_gray = dataset.preprocessing_function(images_gray, data_format = 'channels_last')#np.asarray(images_gray))
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