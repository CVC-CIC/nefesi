import numpy as np
import warnings
import keras.backend as K
import mxnet as mx
#from .neuron_data import NeuronData
from multiprocessing.pool import ThreadPool  # ThreadPool don't have documentation :( But uses threads
import PIL
import time
from scipy.interpolate import RectBivariateSpline

ACTIVATIONS_BATCH_SIZE = 200

def get_activations(model, model_inputs, layers_name):
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
    if isinstance(layers_name, str):
        layers_name = [layers_name]

    # uses .get_output_at() instead of .output. In case a layer is
    # connected to multiple inputs. Assumes the input at node index=0
    # is the input where the model inputs come from.
    outputs = [model.get_layer(layer).output for layer in layers_name]
    # evaluation functions
    # K.learning_phase flag = 1 (train mode)
    #funcs = K.function(inp+ [K.learning_phase()], outputs) #modifies learning parameters
    #layer_outputs = funcs([model_inputs, 1])
    K.learning_phase = 0
    funcs = K.function(inp, outputs)
    layer_outputs = funcs([model_inputs])
    #locations_and_max = [get_argmax_and_max(layer) for layer in layer_outputs]
    with ThreadPool(processes=None) as pool:  # use all cpu cores
        async_results = [pool.apply_async(get_argmax_and_max, (layer,)) for layer in layer_outputs]
        locations_and_max = [async_result.get() for async_result in async_results]
        pool.close()#if don't close pickle not allows to save :( 'with' seems have nothing...-
        pool.terminate()
        pool.join()
    return locations_and_max


def get_argmax_and_max(layer):
    if len(layer.shape) == 2: #Is not conv
        return layer
    #The height and width of the image
    unravel_shape = layer.shape[1:-1]
    #The batch size
    num_images = layer.shape[0]
    #The num of neurons in this layer
    num_filters = layer.shape[-1]
    #The same layer but with height and width in one vector, to find the max point.
    layer = layer.reshape(num_images, unravel_shape[0] * unravel_shape[1], num_filters)
    #find the max point on each vector that represent height and width
    argmax_idx = layer.argmax(axis=1).reshape(-1)
    #Unravel this argmax (convert from one dimensional to two dimensional (x,y)
    xy_locations = np.array(np.unravel_index(argmax_idx, unravel_shape)). \
        reshape(2, num_images, num_filters).transpose()
    #Create a meshgrid for make posible to find the max in one vectorized operation
    filters_idx, images_idx = np.meshgrid(range(num_filters), range(num_images))
    # the list of the num_images max activations for each filter (maxActs[i,j] will be the max activation
    # for the image i on neuron j.
    max_acts = layer[images_idx.reshape(-1),
                                   argmax_idx,
                                   filters_idx.reshape(-1)].reshape(num_images, num_filters)
    return (xy_locations, max_acts)
"""
def get_maxs_and_argmax(layer, gpu_thr = 10000):
    
    Make the calcs of the argmax and max in the fastest way
    :param layer: activations of the layer
    :return: the max and argmax
    shape = layer.shape
    layer = layer.reshape(shape[0], shape[1] * shape[2], shape[3])
    if layer.shape[1]>gpu_thr:
        return mx.nd.argmax(mx.nd.array(layer), axis=1)
    else:
        return layer.argmax(axis = 1).reshape(-1)
"""

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
    outputs = [layer.output[...,idx_neuron] for layer in model.layers if
               layer.name == layer_name or layer_name is None]

    # evaluation functions
    funcs = K.function(inp + [K.learning_phase()], outputs)
    # K.learning_phase flag = 1 (train mode)
    layer_outputs = funcs([model_inputs, 1])
    if len(layer_outputs) > 1:
        warnings.warn("Layer outputs is a list of more than one element? REVIEW THIS CODE SECTION!",RuntimeWarning)
    return layer_outputs[0]

def fill_all_layers_data_batch(file_names, images, model, layers_data):
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
    layer_names = [layer.layer_id for layer in layers_data]
    activations = get_activations(model, images, layer_names)
    for i, layer_activation in enumerate(activations):
        conv_layer = type(layer_activation) is tuple
        if conv_layer:
            num_filters = layer_activation[1].shape[-1]
            xy_locations, max_acts = layer_activation
            for idx_filter in range(num_filters):
                #Add the results to his correspondent neuron
                layers_data[i].neurons_data[idx_filter].add_activations(max_acts[:, idx_filter], file_names, xy_locations[idx_filter, :, :])

        else:
            num_images, num_filters = layer_activation.shape
            xy_locations = np.zeros((num_images, 2), dtype=np.int)
            for idx_filter in range(num_filters):
                layers_data[i].neurons_data[idx_filter].add_activations(layer_activation[:, idx_filter], file_names, xy_locations)


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
    activations = get_activations(model, images, layer_name=None)#layer_name)
    for layer_activation in activations:
        conv_layer = type(layer_activation) is tuple
        if neurons_data is None:
            if conv_layer:
                num_filters = layer_activation[1].shape[-1]
            else:
                num_filters = layer_activation.shape[-1]
            # if `neurons_data` is None, creates the list and fill it
            # with the `nefesi.neuron_data.NeuronData` instances
            neurons_data = np.zeros(num_filters, dtype=np.object)
            for idx_filter in range(num_filters):
                neurons_data[idx_filter] = NeuronData(num_max_activations, batch_size, buffered_iterations=batches_to_buffer)
        if conv_layer:
            num_filters = layer_activation[1].shape[-1]
            xy_locations, max_acts = layer_activation
            for idx_filter in range(num_filters):
                #Add the results to his correspondent neuron
                neurons_data[idx_filter].add_activations(max_acts[:,idx_filter], file_names, xy_locations[idx_filter,:,:])

        else:
            num_images, num_filters = layer_activation.shape
            xy_locations = np.zeros((num_images, 2), dtype=np.int)
            for idx_filter in range(num_filters):
                neurons_data[idx_filter].add_activations(layer_activation[:,idx_filter], file_names, xy_locations)
    return neurons_data


def get_activation_from_pos(images, model, layer_name, idx_neuron, pos, batch_size = ACTIVATIONS_BATCH_SIZE):
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
    batches = np.array(np.arange(0,len(images), batch_size).tolist()+[len(images)])
    if idx_neuron is None:
        for layer in model.layers:
            if layer.name == layer_name:
                neurons_of_layer = layer.output.shape[-1]
                break
        else:
            raise ValueError('Layer '+layer_name+"don't exist in the model "+model.name)
        activations = np.zeros(shape=(len(images),neurons_of_layer), dtype=np.float)
        #Get the activation of all neuron
        for i in range(1,len(batches)):
            total_activations = get_activations(model, images[batches[i-1]:batches[i]], layer_name=layer_name)[0]
            activations[batches[i - 1]:batches[i]] = total_activations[range(len(total_activations)),
                                                                       pos[batches[i - 1]:batches[i],0],
                                                                       pos[batches[i - 1]:batches[i],1]]
    else:
        activations = np.zeros(shape=len(images), dtype=np.float)
        for i in range(1,len(batches)):
            total_activations = get_one_neuron_activations(model, images[batches[i-1]:batches[i]],idx_neuron=idx_neuron, layer_name=layer_name)
            activations[batches[i - 1]:batches[i]] = total_activations[range(len(total_activations)), pos[batches[i - 1]:batches[i],0],pos[batches[i - 1]:batches[i],1]]
    # for each input in 'images' (range(len(activations))), get the activation value in 'pos'
    return activations


def get_image_activation(network_data, image_names, layer_name, neuron_idx, complex_type = True):
    """
    Returns the image correspondant to image_name with a mask of the place that most response has for the neuron
    neuron_idx of layer layer_name
    :param network_data: Network_data object representing the nefesi network
    :param image_name: the name of the image to analyze
    :param layer_name: the name of the layer of the network where is the neuron to analyze
    :param neuron_idx: the index of the neuron to analyze
    :param complex_type: True as torralba, False as vedaldi  (falta posar referencies)
    :return: An image that is activation on a neuron but in the original image size
    """
    #input = network_data.dataset._load_image(image_name, as_numpy=True,
    #                                              prep_function=True)[np.newaxis, ...]
    inputs = network_data.dataset.load_images(image_names=image_names, prep_function=True)

    activations = get_one_neuron_activations(model=network_data.model, model_inputs=inputs,
                                             layer_name=layer_name, idx_neuron=neuron_idx)
    activations_upsampleds = []
    if complex_type:
        rec_field_map = network_data.get_layer_by_name(layer_name).receptive_field_map
        rec_field_sz = network_data.get_layer_by_name(layer_name).receptive_field_size
        rec_field_map_2 = np.zeros(rec_field_map.shape, dtype=np.int32)
    for input, activation in zip (inputs, activations):
        sz_img = input.shape[0:2]
        if not complex_type:
            activations_upsampled = np.array(PIL.Image.fromarray(activation).resize(tuple(sz_img), PIL.Image.BILINEAR))

        else:
            pos = np.zeros(list(activation.shape)[::-1]+[2])
            # Generates the Mask that defines the data that need to be operate
            rec_field_map_mask = np.full(rec_field_map.shape, [0, sz_img[0], 0 , sz_img[1]], dtype=np.int32) == rec_field_map
            #If have some point to be readjusted
            if rec_field_map_mask.max():
                #Generate the mask with all points calculated
                rec_field_map_2[:, :, 0] = rec_field_map[:, :, 1] - rec_field_sz[1]
                rec_field_map_2[:, :, 1] = rec_field_map[:, :, 0] + rec_field_sz[1] - 1
                rec_field_map_2[:, :, 2] = rec_field_map[:, :, 3] - rec_field_sz[0]
                rec_field_map_2[:, :, 3] = rec_field_map[:, :, 2] + rec_field_sz[0] - 1

                #Crosses two matrix
                rec_field_map[rec_field_map_mask] = rec_field_map_2[rec_field_map_mask]

            pos[:, :, 0] = (rec_field_map[:, :, 3] + rec_field_map[:, :, 2]) * 0.5
            pos[:, :, 1] = (rec_field_map[:, :, 1] + rec_field_map[:, :, 0]) * 0.5

            spline = RectBivariateSpline(np.unique(np.sort(pos[1], axis=None)), np.unique(np.sort(pos[0], axis=None)),
                                         activation, kx=2, ky=2)


            activations_upsampled = spline(np.arange(sz_img[0]), np.arange(sz_img[1]))


        activations_upsampled[activations_upsampled<0] = 0
        activations_upsampleds.append(activations_upsampled)

    return activations_upsampleds
