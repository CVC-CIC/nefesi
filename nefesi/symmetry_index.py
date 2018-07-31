from nefesi import read_activations
from .util.image import rotate_images_axis
import numpy as np

def get_symmetry_index(neuron_data, model, layer_data, dataset):
    """Returns the symmetry selectivity index.
    This index is a list with four types of mirroring the image,
    having a previous rotation of 0, 45, 90 and 135 degrees.
    The last value in the list is the average of the rest of index values.

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param model: The `keras.models.Model` instance.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    :param dataset: The `nefesi.util.image.ImageDataset` instance.

    :return: List of floats, index symmetry values.
    """
    symm_axes = [0, 45, 90, 135]
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
        for i in range(len(symm_axes)):
            # apply the mirroring function over the images
            images_r = rotate_images_axis(images, symm_axes[i], layer_data, locations)
            rot_activations = read_activations.get_activation_from_pos(images_r, model,
                                                                       layer_data.layer_id,
                                                                       idx_neuron, locations)
            # normalize activations.
            norm_rot_act_sum = np.sum(rot_activations) / max_act
            symmetry_idx = norm_rot_act_sum / norm_activations_sum
            results[i] = symmetry_idx
    return results
