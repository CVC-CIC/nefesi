from . import read_activations
from .util.image import rotate_images_axis
import numpy as np

SYMMETRY_AXES = [0, 45, 90, 135]

def get_symmetry_index(neuron_data, model, layer_data, dataset, symm_axes=SYMMETRY_AXES):
    """Returns the symmetry selectivity index.
    This index is a list with four types of mirroring the image,
    having a previous rotation of 0, 45, 90 and 135 degrees.
    The last value in the list is the average of the rest of index values.

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param model: The `keras.models.Model` instance.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    :param dataset: The `nefesi.util.image.ImageDataset` instance.
    :param symm_axes: List. Symmetry axes to use (by default [0,45,90,135]
    :return: List of floats, index symmetry values.
    """
    results = np.zeros(len(symm_axes),dtype=np.float)
    activations = neuron_data.activations
    norm_activations_sum = np.sum(neuron_data.norm_activations)
    image_names = neuron_data.images_id
    locations = neuron_data.xy_locations

    max_act = activations[0]

    if max_act != 0.0:
        images = dataset.load_images(image_names)
        #[0][0] because return a tuple of list
        idx_neuron = np.where(layer_data.neurons_data==neuron_data)[0][0]
        images_r = rotate_images_axis(images, symm_axes, layer_data, locations)
        images_reshaped = images_r.reshape((images_r.shape[0] * images_r.shape[1], images_r.shape[2],
                                            images_r.shape[3], images_r.shape[4]))
        replicated_locations = np.full(shape=(len(symm_axes),) + locations.shape, fill_value=locations).reshape(-1, 2)
        rot_activations = read_activations.get_activation_from_pos(images_reshaped, model,
                                                                   layer_data.layer_id,
                                                                   idx_neuron,
                                                                   replicated_locations)
        rot_activations = rot_activations.reshape((len(symm_axes),len(images)))
        norm_rot_act_sum = np.sum(rot_activations, axis=1) / max_act
        results = norm_rot_act_sum / norm_activations_sum #WARNING: why not 1-results?
        results = np.clip(a=results, a_min=0., a_max=1.)
    print(results)
    return results
