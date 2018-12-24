import numpy as np

from keras.preprocessing import image
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import _Pooling2D


def compute_nf(network_data, layer_data, verbose=True, maximize_contrast = False, mode = 1,threshold_to_noncount = 0.1):
    """This function build the neuron features (NF) for all neurons
    in `filters`.

    :param network_data: The `nefesi.network_data.NetworkData` instance.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    """

    if layer_data.receptive_field_map is None:
        layer_data.mapping_rf(network_data.model)

    for i, neuron in enumerate(layer_data.neurons_data):
        if neuron.norm_activations is not None:
            norm_activations = neuron.norm_activations
            # get the receptive fields from a neuron
            patches,masks = neuron.get_patches(network_data, layer_data,return_mask=True)
            channels = 1 if len(patches.shape) < 4 else patches.shape[-1]
            if mode == 1:
            #set values out of the image (black ones) to a given values (127)
                patches[np.repeat(masks,channels).reshape(patches.shape)] = 127
                nf = np.sum(patches.reshape(patches.shape[0], -1) * (norm_activations / np.sum(
                norm_activations))[:, np.newaxis], axis=0).reshape(patches.shape[1:])
            else:
            # Set the neuron feature but not having in count pixels of paddings.
                #each pixel that corresponds to a true image give norm_activation weigth, paddings don't contribute
                contributions_per_pixel = np.sum((masks==False)*np.repeat(norm_activations,masks[0].size).
                                                 reshape(masks.shape),axis=0)
                #each pixel of patchs multiplies by his norm_activation
                patches_weighted = patches*np.repeat(norm_activations,patches[0].size).reshape(patches.shape)
                #normalized having that only pixels that appears counts
                nf = np.sum(patches_weighted,axis=0) / np.repeat(contributions_per_pixel, channels).\
                    reshape(patches[0].shape)
                #only pixels that have more than 10% of info from ntop scoring counts the rest shows gray
                relevant_pixels = contributions_per_pixel>(np.sum(norm_activations)*threshold_to_noncount)
                nf[relevant_pixels == False] = 127
                """
                non_black_pixels_count = np.count_nonzero(np.sum(patches, axis=-1)==1, axis=0)
                assignment_multiplier = np.repeat(len(patches) / non_black_pixels_count, 3).reshape(
                    non_black_pixels_count.shape + (3,))
                """
            """
            nf = np.sum(patches.reshape(patches.shape[0],-1)*(norm_activations/np.sum(norm_activations))[:,np.newaxis],axis=0).\
                reshape(patches.shape[1:])
            #Better to maximize contrast after evaluation, in order to save a more fidedign NF.
            if nf.shape[2] == 3 and maximize_contrast:  # RGB images
                # maximize the contrast of the NF
                min_v = np.min(nf.ravel())
                max_v = np.max(nf.ravel())
                #If max-min not is 0
                if not np.isclose(min_v, max_v):
                    nf -= min_v
                    nf /= (max_v - min_v)
            """
            #save as PIL image
            neuron.neuron_feature = image.array_to_img(nf)
            if verbose and i%50==0:
                print("NF - "+layer_data.layer_id+". Neurons completed: "+str(i)+"/"+str(len(layer_data.neurons_data)))
        else:
            # if `norm_activations` from a neuron is None, that means this neuron
            # doesn't have activations. NF is setting with None.
            neuron.neuron_feature = None


def get_each_point_receptive_field(model, layer_name):
    """Takes `weight` and `height` (of a layer output)  and gets the map receptive_field_map of each pixel to the input layer
    (usually same as input image size).

    :param model: The `keras.models.Model` instance.
    :param layer_name: String, name of the layer to get his receptive_field to input.

    :return: The window location of the receptive field in the input image.
    Numpy matrix (3-D) that contains for each point in matrix(i,j) --> [row_ini, row_fin, col_ini, col_fin].
        output[i,j] = 4 points of rectangle or square that corresponds to pixel i,j on a neuron on layer layer name, to
        on the input image. The exact position of the receptive field from an image.
    """
    current_layer_idx = find_layer_idx(model, layer_name=layer_name)
    if len(model.layers[current_layer_idx].output_shape)>2:
        _, w, h, _ = model.layers[current_layer_idx].output_shape
    else:
        h = 1
        _, w = model.layers[current_layer_idx].output_shape
    w_mesh, h_mesh = np.meshgrid(range(h), range(w))
    # array order --> row_ini, row_fin, col_ini, col_fin
    image_points = np.array([h_mesh.flatten(),h_mesh.flatten(), w_mesh.flatten(),w_mesh.flatten()],dtype=np.int32).\
        T.reshape(w, h, 4)

    # goes throw the current layer until the first input layer.
    # (input shape of the network)
    for i in range(current_layer_idx, -1, -1):
        current_layer = model.layers[i]

        #REVIEW IF W AND H ARE CORRECT!!!!!!!!!!!
        if len(current_layer.input_shape) == 4:
            _, current_size_w, current_size_h, _ = current_layer.input_shape
        else:
            current_size_w, current_size_h = (float('Inf'), float('Inf'))
        #Checks to boundaries of the current layer shape.
        image_points[:,:,[0,2]] = np.maximum(image_points[:,:,[0,2]],0)
        image_points[:,:,1] = np.minimum(image_points[:,:,1], current_size_w - 1)
        image_points[:, :, 3] = np.minimum(image_points[:, :, 3], current_size_h - 1)
        # check if the current layer is a convolution layer or
        # a pooling layer (both have to be 2D).
        if isinstance(current_layer, Conv2D) or isinstance(current_layer, _Pooling2D):
            # get some configuration parameters,
            # padding, kernel_size, strides
            config_params = current_layer.get_config()
            padding = config_params['padding']
            strides = np.array(config_params['strides'])
            kernel_size = np.array(config_params.get('kernel_size', config_params.get('pool_size')))

            if padding == 'same':
                # padding = same, means input shape = output shape
                padding = (kernel_size-1)//2
            else:
                padding = np.zeros(shape=2, dtype=np.int)
            # calculate the window location applying the proper displacements.
            image_points*=strides[[0,0,1,1]]
            image_points[:, :, [1, 3]]+=(kernel_size-1)

            # apply the padding on the receptive field window.
            image_points-=padding[[0,0,1,1]]
        #(is neccesary to add 1 in row_fin and col_fin due to behaviour of Numpy arrays.
    image_points[:,:,[1,3]]+= 1
    return image_points



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

    row_ini, row_fin = x, x
    col_ini, col_fin = y, y

    # goes throw the current layer until the first input layer.
    # (input shape of the network)
    for i in range(current_layer_idx, -1, -1):
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
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            return idx
    else:
        raise ValueError("No layer with layer_id '{}' within the model".format(layer_name))
