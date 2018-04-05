import numpy as np
import time

import keras.backend as K
from neuron_data import NeuronData


def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    """Returns the output (activations) from the model.

    :param model: The `keras.models.Model` instance.
    :param model_inputs: List of inputs, the inputs expected by the network.
    :param print_shape_only: Boolean, if its True print the shape of the output tensor.
    :param layer_name: String, name of the layer from which get the outputs.
        If its None, returns the outputs from all the layers in the model.

    :return: List of activations, one output for each given layer.
    """
    activations = []
    inp = model.input
    if not isinstance(inp, list):
        inp = [inp]

    # all layer outputs
    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]

    # evaluation functions
    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]

    # K.learning_phase flag = 1 (train mode)
    layer_outputs = [func([model_inputs, 1])[0] for func in funcs]

    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
    return activations


def get_sorted_activations(file_names, images, model, layer_name,
                           filters, num_max_activations, batch_size):
    """Returns the neurons with their maximum activations as the
    inputs (`images`) are processed.

    :param file_names: List of strings, name of the images.
    :param images: List of inputs, the inputs expected by the network.
    :param model: The `keras.models.Model` instance.
    :param layer_name: String, name of the layer from which get the outputs.
    :param filters: List of `nefesi.neuron_data.NeuronData` instances.
        Class attribute `filters` from `nefesi.layer_data.LayerData`.
    :param num_max_activations: Integer, number of TOP activations stored
        in each `nefesi.neuron_data:NeuronData` instance.
    :param batch_size: Integer, size of batch.

    :return: List of `nefesi.neuron_data.NeuronData` instances.
    """
    start = time.time()
    activations = get_activations(model, images,
                                  print_shape_only=False,
                                  layer_name=layer_name)

    print('Time for get activations: ' + str(time.time() - start))
    start = time.time()

    for layer_activation in activations:
        # get the number of images and the number of neurons in this layer
        num_images, _, _, num_filters = layer_activation.shape

        if filters is None:
            # if `filters` is None, creates the list and fill it
            # with the `nefesi.neuron_data.NeuronData` instances
            filters = []
            for i in xrange(num_filters):
                n_data = NeuronData(num_max_activations, batch_size)
                filters.append(n_data)

        for f in filters:
            idx_filter = filters.index(f)
            for j in xrange(num_images):
                # get the map activation for each image and each neuron
                activation_map = layer_activation[j, :, :, idx_filter]
                # look up for the maximum activation
                max_act = np.amax(activation_map)
                image_id = file_names[j]
                # get the location of the maximum activation inside the map
                xy_location = np.unravel_index(activation_map.argmax(),
                                               activation_map.shape)
                f.add_activation(max_act, image_id, xy_location)

    print('Time for order in ', str(num_filters), ': ' + str(time.time() - start))
    return filters


def get_activation_from_pos(images, model, layer_name, idx_neuron, pos):
    """Returns the activations of a neuron, given a location (`pos`)
     on the activation map for each input (`images`).

    :param images: List of inputs, the inputs expected by the network.
    :param model:The `keras.models.Model` instance.
    :param layer_name: String, name of the layer from which get the outputs.
    :param idx_neuron: Integer, index of the neuron from which get
        the activation.
    :param pos: Tuple of integers, location in the activation map.

    :return: List of floats, activation values for each input in `images`.
    """
    activations = get_activations(model, images, print_shape_only=False, layer_name=layer_name)

    new_activations = None
    for layer_activation in activations:
        num_images, _, _, num_filters = layer_activation.shape

        new_activations = np.zeros(num_images)
        for j in xrange(num_images):
            # for each input in `images`, get the activation value in `pos`
            x, y = pos[j]
            f = layer_activation[j, x, y, idx_neuron]
            new_activations[j] = f
    return new_activations
