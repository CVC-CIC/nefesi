import read_activations
from utils.image import rotate_images_axis


def get_symmetry_index(neuron_data, model, layer_data, dataset):
    """Returns the symmetry selectivity index.
    This index is a list with four types of mirroring the image,
    in the axes 0, 45, 90 and 135 degrees.
    The last value in the list is the average of the rest of index values.

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param model: The `keras.models.Model` instance.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    :param dataset: The `nefesi.utils.image.ImageDataset` instance.

    :return: List of floats, index symmetry values.
    """
    results = []
    symm_axes = [0, 45, 90, 135]
    avg_symmetry_idx = 0

    activations = neuron_data.activations
    norm_activations = neuron_data.norm_activations
    image_names = neuron_data.images_id
    locations = neuron_data.xy_locations

    max_act = activations[0]

    if max_act != 0.0:
        images = dataset.load_images(image_names)
        idx_neuron = layer_data.neurons_data.index(neuron_data)
        for axes in symm_axes:
            # apply the mirroring function over the images
            images_r = rotate_images_axis(images, axes, layer_data, locations)
            rot_activations = read_activations.get_activation_from_pos(images_r, model,
                                                                       layer_data.layer_id,
                                                                       idx_neuron, locations)
            # normalize activations.
            norm_rot_act = rot_activations / max_act
            partial_symmetry = sum(norm_rot_act) / sum(norm_activations)
            avg_symmetry_idx = avg_symmetry_idx + partial_symmetry
            results.append(partial_symmetry)

        # averaged symmetry index value (global symmetry).
        avg_symmetry_idx = avg_symmetry_idx / len(symm_axes)
        results.append(avg_symmetry_idx)
        return results
    else:
        for x in xrange(len(symm_axes) + 1):
            results.append(0.0)
        return results
