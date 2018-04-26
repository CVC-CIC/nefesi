import read_activations
from utils.image import rotate_images


def get_orientation_index(neuron_data, model, layer_data, dataset):
    """Returns the orientation selectivity index.
    This index is a list with the index value for each 15 degrees of rotation.
    The last value in the list is the average of the rest of index values.

    :param neuron_data: The `nefesi.neuron_data.NeuronData` instance.
    :param model: The `keras.models.Model` instance.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    :param dataset: The `nefesi.utils.image.ImageDataset` instance.

    :return: List of floats, index orientation values.
    """
    degrees = 15
    n_rotations = 25
    l_degrees = [x * degrees for x in xrange(1, n_rotations)]  # avoid 0 degrees

    results = []
    avg_orientation_index = 0

    activations = neuron_data.activations
    norm_activations = neuron_data.norm_activations
    image_names = neuron_data.images_id
    locations = neuron_data.xy_locations
    max_act = activations[0]

    if max_act != 0.0:
        images = dataset.load_images(image_names)
        idx_neuron = layer_data.neurons_data.index(neuron_data)
        for degrees in l_degrees:
            # apply the rotation function
            images_r = rotate_images(images, degrees, locations, layer_data)
            rot_activations = read_activations.get_activation_from_pos(images_r, model,
                                                                       layer_data.layer_id,
                                                                       idx_neuron,
                                                                       locations)
            # normalize activations
            norm_rot_act = rot_activations / max_act
            partial_orientation = 1 - (sum(norm_rot_act) / sum(norm_activations))
            avg_orientation_index = avg_orientation_index + partial_orientation
            results.append(partial_orientation)

        # averaged orientation index value (global orientation).
        avg_orientation_index = avg_orientation_index / len(l_degrees)
        results.append(avg_orientation_index)
        return results
    else:
        for x in xrange(len(l_degrees) + 1):
            results.append(0.0)
        return results
