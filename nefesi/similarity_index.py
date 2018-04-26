import read_activations


def get_similarity_index(neuron_a, neuron_b, idx_a, model, layer_name, dataset):
    """Returns the similarity index value. Calculates the similarity value
    from `filter_b` to `filter_a`.

    :param neuron_a: The `nefesi.neuron_data.NeuronData` instance.
    :param neuron_b: The `nefesi.neuron_data.NeuronData` instance.
    :param idx_a: Integer, index of `filter_a` in the layer.
    :param model: The `keras.models.Model` instance.
    :param layer_name: String, name of layer (`layer_id`).
    :param dataset: The `nefesi.utils.image.ImageDataset` instance.

    :return: Float, index similarity value.
    """
    act_a = neuron_a.activations
    act_b = neuron_b.activations

    # check if both neurons doesn't have 0 activations.
    if act_a[0] != 0.0 and act_b[0] != 0.0:
        images_b = neuron_b.images_id
        images_b = dataset.load_images(images_b)
        locations_b = neuron_b.xy_locations

        # get the activations of the TOP scored images inside
        # the neuron_data `neuron_b` on the neuron `neuron_a`.
        new_act = read_activations.get_activation_from_pos(images_b, model,
                                                           layer_name, idx_a,
                                                           locations_b)
        norm_new_act = new_act / act_a[0]
        return sum(norm_new_act / sum(neuron_a.norm_activations))
    else:
        return 0.0
