import numpy as np

import keras.backend as K
from .neuron_data import NeuronData


def get_activations(model, model_inputs, layer_name=None):
    """Returns the output (activations) from the model.

    :param model: The `keras.models.Model` instance.
    :param model_inputs: List of inputs, the inputs expected by the network.
    :param layer_name: String, name of the layer from which get the outputs.
        If its None, returns the outputs from all the layers in the model.

    :return: List of activations, one output for each given layer.
    """
    inp = model.input
    if type(inp) is not list:
        inp = [inp]

    # uses .get_output_at() instead of .output. In case a layer is
    # connected to multiple inputs. Assumes the input at node index=0
    # is the input where the model inputs come from.
    outputs = [layer.get_output_at(0) for layer in model.layers if
               layer.name == layer_name or layer_name is None]

    # evaluation functions
    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]

    # K.learning_phase flag = 1 (train mode)
    layer_outputs = [func([model_inputs, 1])[0] for func in funcs]

    return layer_outputs


def get_sorted_activations(file_names, images, model, layer_name,
                           neurons_data, num_max_activations, batch_size):
    """Returns the neurons with their maximum activations as the
    inputs (`images`) are processed.

    :param file_names: List of strings, name of the images.
    :param images: List of inputs, the inputs expected by the network.
    :param model: The `keras.models.Model` instance.
    :param layer_name: String, name of the layer from which get the outputs.
    :param neurons_data: List of `nefesi.neuron_data.NeuronData` instances.
    :param num_max_activations: Integer, number of TOP activations stored
        in each `nefesi.neuron_data:NeuronData` instance.
    :param batch_size: Integer, size of batch.

    :return: List of `nefesi.neuron_data.NeuronData` instances.
    """
    activations = get_activations(model, images, layer_name=layer_name)
    conv_layer = True
    for layer_activation in activations:
        # get the number of images and the number of neurons in this layer
        if len(layer_activation.shape) == 2:
            num_images, num_filters = layer_activation.shape
            conv_layer = False
        else:
            num_images, _, _, num_filters = layer_activation.shape

        if neurons_data is None:
            # if `neurons_data` is None, creates the list and fill it
            # with the `nefesi.neuron_data.NeuronData` instances
            neurons_data = np.zeros(num_filters, dtype=np.object)
            for idx_filter in range(num_filters):
                neurons_data[idx_filter] = NeuronData(num_max_activations, batch_size)
        if conv_layer:
            unravel_shape = layer_activation.shape[1:-1]
            #the activation map of each image for each filter idx filter, with map reshaped in one dim for optimization
            #(numberImg, activationMapReshaped, idx_filter
            activation_reshaped = layer_activation.reshape(layer_activation.shape[0],
                                                           layer_activation.shape[1]*layer_activation.shape[2],
                                                           layer_activation.shape[3])
            #The position on reshaped activations of max values
            argmax_idx = activation_reshaped.argmax(1)
            #range of num_images, to apply on the first axis
            num_images_range = range(num_images)
            for idx_filter in range(num_filters):
                #The position of argmaxs corresponding to this filter
                argmaxs_of_filter = argmax_idx[:, idx_filter]
                #the list of the num_images max activations. (One value for each image)
                max_acts_of_filter = activation_reshaped[num_images_range, argmaxs_of_filter, idx_filter]
                #The corresponding xy location of this maxs
                xy_locations = np.array(np.unravel_index(argmaxs_of_filter, unravel_shape)).transpose()
                neurons_data[idx_filter].add_activations(max_acts_of_filter, file_names, xy_locations)

        else:
            xy_locations = np.zeros((num_images, 2), dtype=np.int)
            for idx_filter in range(len(neurons_data)):
                neurons_data[idx_filter].add_activations(layer_activation[:,idx_filter], file_names, xy_locations)
    return neurons_data


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
    #REVIEW THIS FUNCTION, it's necessary to CALC ALL activations to only one neuron?
    activations = get_activations(model, images, print_shape_only=False, layer_name=layer_name)

    new_activations = None
    for layer_activation in activations:
        num_images, _, _, num_filters = layer_activation.shape

        new_activations = np.zeros(num_images)
        for j in range(num_images):
            # for each input in `images`, get the activation value in `pos`
            x, y = pos[j]
            f = layer_activation[j, x, y, idx_neuron]
            new_activations[j] = f
    return new_activations
