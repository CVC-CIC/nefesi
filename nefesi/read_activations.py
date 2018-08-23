import numpy as np
import warnings
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
    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]

    # evaluation functions
    funcs = K.function(inp+ [K.learning_phase()], outputs )

    # K.learning_phase flag = 1 (train mode)
    layer_outputs = funcs([model_inputs, 1])

    return layer_outputs

def get_one_neuron_activations(model, model_inputs, idx_neuron, layer_name=None):
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
    outputs = [layer.get_output_at(0)[...,idx_neuron] for layer in model.layers if
               layer.name == layer_name or layer_name is None]

    # evaluation functions
    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]

    # K.learning_phase flag = 1 (train mode)
    layer_outputs = [func([model_inputs, 1])[0] for func in funcs]
    if len(layer_outputs) > 1:
        warnings.warn("Layer outputs is a list of more than one element? REVIEW THIS CODE SECTION!",RuntimeWarning)
    return layer_outputs[0]


def get_sorted_activations(file_names, images, model, layer_name,
                           neurons_data, num_max_activations, batch_size, batches_to_buffer = 20):
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
    :param batches_to_buffer: quantity of results that are saved in buffer before having sort (the best value will be the total
    number of batches, but controlling memory (memory used will be 128*3*batchSize bytes)

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
                neurons_data[idx_filter] = NeuronData(num_max_activations, batch_size, buffered_iterations=batches_to_buffer)
        if conv_layer:
            unravel_shape = layer_activation.shape[1:-1]
            #the activation map of each image for each filter idx filter, with map reshaped in one dim for optimization
            #(numberImg, activationMapReshaped, idx_filter)
            activation_reshaped = layer_activation.reshape(num_images,
                                                           layer_activation.shape[1]*layer_activation.shape[2],
                                                           num_filters)
            #The position on reshaped activations of max values.
            argmax_idx = activation_reshaped.argmax(axis = 1).reshape(-1)
            # The corresponding xy location of this maxs. xy_locations[i,j,:] will be the [x,y] position of the max activation
            # for the image i on neuron j.
            xy_locations = np.array(np.unravel_index(argmax_idx, unravel_shape)).\
                reshape(2,num_images,num_filters).transpose()
            #index of axis 2 and 0, for extract all max activations in one operation
            filters_idx, images_idx = np.meshgrid(range(num_filters),range(num_images))
            # the list of the num_images max activations for each filter (maxActs[i,j] will be the max activation
            # for the image i on neuron j.
            max_acts = activation_reshaped[images_idx.reshape(-1),
                                           argmax_idx,
                                           filters_idx.reshape(-1)].reshape(num_images,num_filters)
            for idx_filter in range(num_filters):
                #Add the results to his correspondent neuron
                neurons_data[idx_filter].add_activations(max_acts[:,idx_filter], file_names, xy_locations[idx_filter,:,:])

        else:
            xy_locations = np.zeros((num_images, 2), dtype=np.int)
            for idx_filter in range(len(neurons_data)):
                neurons_data[idx_filter].add_activations(layer_activation[:,idx_filter], file_names, xy_locations)
    return neurons_data


def get_activation_from_pos(images, model, layer_name, idx_neuron, pos):
    """Returns the activations of a neuron, given a location ('pos')
     on the activation map for each input (`images`).

    :param images: List of inputs, the inputs expected by the network.
    :param model:The `keras.models.Model` instance.
    :param layer_name: String, name of the layer from which get the outputs.
    :param idx_neuron: Integer, index of the neuron from which get
        the activation.
    :param pos: numpy of integers, location in the activation map.

    :return: List of floats, activation values for each input in `images`.
    """
    if idx_neuron is None:
        #Get the activation of all neuron
        activations = get_activations(model, images, layer_name=layer_name)[0]
    else:
        activations = get_one_neuron_activations(model, images,idx_neuron=idx_neuron, layer_name=layer_name)
    # for each input in 'images' (range(len(activations))), get the activation value in 'pos'
    return activations[range(len(activations)),pos[...,0],pos[...,1]]
