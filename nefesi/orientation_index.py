from nefesi import read_activations
from .util.image import rotate_images
import numpy as np

def get_orientation_index(neuron_data, model, layer_data, dataset, degrees_to_rotate = 15):
    """Returns the orientation selectivity index.
    This index is a list with the index value for each 15 degrees of rotation.
    The last value in the list is the average of the rest of index values.

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param model: The `keras.models.Model` instance.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    :param dataset: The `nefesi.util.image.ImageDataset` instance.
    :param degrees_to_rotate: degrees to apply on each rotation (default: 15).

    :return: List of floats, index orientation values.
    """
    degrees_to_complete = 360
    #360 not included (is same as 0), and 0 idx don't have sense
    l_degrees = np.arange(degrees_to_rotate%360, degrees_to_complete, degrees_to_rotate%360, dtype=np.int16)
    results = np.zeros(len(l_degrees),dtype=np.float)

    activations = neuron_data.activations
    norm_activations_sum = np.sum(neuron_data.norm_activations)
    image_names = neuron_data.images_id
    locations = neuron_data.xy_locations
    max_act = activations[0]

    if max_act != 0.0:
        images = dataset.load_images(image_names)
        idx_neuron = np.where(layer_data.neurons_data == neuron_data)[0][0]
        for degrees_to_rotate_idx in range(len(l_degrees)):
            # apply the rotation function
            images_r = rotate_images(images, l_degrees[degrees_to_rotate_idx], locations, layer_data)
            rot_activations = read_activations.get_activation_from_pos(images_r, model,
                                                                       layer_data.layer_id,
                                                                       idx_neuron,
                                                                       locations)
            # normalize activations
            norm_rot_act_sum = np.sum(rot_activations) / max_act
            results[degrees_to_rotate_idx] = 1 - (norm_rot_act_sum / norm_activations_sum)
    results = np.clip(a=results, a_min=0., a_max=1.)
    return results
