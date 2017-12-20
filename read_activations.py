import keras.backend as K
import numpy as np

from neuron_data import NeuronData

def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(1.)
    else:
        list_inputs = [model_inputs, 0]

    # Learning phase. 1 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 1.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations

def get_sorted_activations(file_names, images, model, layer, filters, num_max_activations, batch_size):

    import time
    # given a model, a set of inputs and specific layer, returns the activations
    # from each neuron for each input
    start = time.time()
    activations = get_activations(model, images, print_shape_only=True, layer_name=layer)
    print('Time for get activations: '+ str(time.time()-start))

    start = time.time()

    for layer_activation in activations:
        num_images, filter_size, _, num_filters = layer_activation.shape

        if filters is None:
            filters = []
            for i in xrange(num_filters):
                n_data = NeuronData(num_max_activations, batch_size)
                filters.append(n_data)

        for filt in filters:
            filt_idx = filters.index(filt)
            for j in xrange(num_images):
                f = layer_activation[j, :, :, filt_idx]

                max_act = np.amax(f)
                image_id = file_names[j]
                xy_location = np.unravel_index(f.argmax(), f.shape)
                filt.add_activation(max_act, image_id, xy_location)
            # filt.sort()

    print('Time for order in ', str(num_filters), ': ' + str(time.time()-start))
    return filters


def get_activation_from_pos(images, model, layer, idx_neuron, pos):

    activations = get_activations(model, images, print_shape_only=True, layer_name=layer)

    for layer_activation in activations:
        num_images, filter_size, _, num_filters = layer_activation.shape

        new_activations = np.zeros(num_images)

        for j in xrange(num_images):
            x, y = pos[j]
            f = layer_activation[j, x, y, idx_neuron]

            new_activations[j] = f

    return new_activations


