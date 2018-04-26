import numpy as np

from keras.preprocessing import image
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import _Pooling2D


def compute_nf(network_data, layer_data, neurons_data):
    """This function build the neuron features (NF) for all neurons
    in `filters`.

    :param network_data: The `nefesi.network_data.NetworkData` instance.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    :param neurons_data: List of `nefesi.neuron_data.NeuronData` instances.
    """
    if layer_data.receptive_field_map is None:
        model = network_data.model
        layer_idx = find_layer_idx(model, layer_data.layer_id)
        _, w, h, _ = model.layers[layer_idx].output_shape
        layer_data.mapping_rf(model, w, h)

    for f in neurons_data:
        if f.norm_activations is not None:
            norm_activations = f.norm_activations
            # get the receptive fields from a neuron
            patches = f.get_patches(network_data, layer_data)
            num_a = len(patches)

            total_act = np.zeros(image.img_to_array(patches[0]).shape)
            for i in xrange(num_a):
                # multiply each receptive field per the normalized
                # activation of the image to comes from.
                img = image.img_to_array(patches[i])
                norm_act = norm_activations[i]
                total_act = total_act + (img * norm_act)

            # average them with total number of patches (receptive field).
            nf = total_act / num_a

            if nf.shape[2] == 3:  # RGB images
                # maximize the contrast of the NF
                min_v = np.min(nf.ravel())
                max_v = np.max(nf.ravel())
                nf = nf - min_v
                nf = nf / (max_v - min_v)

            f.neuron_feature = image.array_to_img(nf)
        else:
            # if `norm_activations` from a neuron is None, that means this neuron
            # doesn't have activations. NF is setting with None.
            f.neuron_feature = None


def get_image_receptive_field(x, y, model, layer_name):
    """This function takes `x` and `y` position from the map activation generated
    on the output in the layer `layer_name`, and returns the window location
    of the receptive field in the input image.

    :param x: Integer, row position in the map activation.
    :param y: Integer, column position in the map activation.
    :param model: The `keras.models.Model` instance.
    :param layer_name: String, name of the layer.

    :return: Tuple of integers, (x1, x2, y1, y2).
        The exact position of the receptive field from an image.
    """
    current_layer_idx = find_layer_idx(model, layer_name=layer_name)

    row_ini = x
    col_ini = y
    row_fin = row_ini
    col_fin = col_ini

    # goes throw the current layer until the first input layer.
    # (input shape of the network)
    for i in xrange(current_layer_idx, -1, -1):
        current_layer = model.layers[i]

        # print current_layer.name
        # print current_layer.input_shape

        if len(current_layer.input_shape) == 4:
            _, current_size, _, _ = current_layer.input_shape

        # some checks to boundaries of the current layer shape.
        if row_ini < 0:
            row_ini = 0
        if col_ini < 0:
            col_ini = 0
        if row_fin > current_size - 1:
            row_fin = current_size - 1
        if col_fin > current_size - 1:
            col_fin = current_size - 1

        # check if the current layer is a convolution layer or
        # a pooling layer (both have to be 2D).
        if isinstance(current_layer, Conv2D) or isinstance(current_layer, _Pooling2D):
            # get some configuration parameters,
            # padding, kernel_size, strides
            config_params = current_layer.get_config()
            padding = config_params['padding']
            strides = config_params['strides']
            kernel_size = config_params.get('kernel_size', config_params.get('pool_size'))

            if padding == 'same':
                # padding = same, means input shape = output shape
                padding = (kernel_size[0] - 1) / 2, (kernel_size[1] - 1) / 2
            else:
                padding = (0, 0)

            # calculate the window location applying the proper displacements.
            row_ini = row_ini*strides[0]
            col_ini = col_ini*strides[1]
            row_fin = row_fin*strides[0] + kernel_size[0]-1
            col_fin = col_fin*strides[1] + kernel_size[1]-1

            # apply the padding on the receptive field window.
            row_ini -= padding[0]
            col_ini -= padding[1]
            row_fin -= padding[0]
            col_fin -= padding[1]

    return row_ini, row_fin, col_ini, col_fin


def find_layer_idx(model, layer_name):
    """Returns the layer index corresponding to `layer_name` from `model`.

    :param model: The `keras.models.Model` instance.
    :param layer_name: String, name of the layer to lookup.

    :return: Integer, the layer index.

    :raise
        ValueError: If there isn't a layer with layer id = `layer_name`
            in the model.
    """
    layer_idx = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            layer_idx = idx
            break

    if layer_idx is None:
        raise ValueError("No layer with layer_id '{}' within the model".format(layer_name))
    return layer_idx
