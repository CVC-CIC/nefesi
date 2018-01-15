from keras.preprocessing import image

import numpy as np


def compute_nf(dataset, model, layer, filters):

    # we have to normalize the activations for each neuron before
    # calculate the NF
    for f in filters:
        f.normalize_activations()


    # now, we can calculate the NF for each neuron
    for f in filters:
        if f.norm_activations is not None:
            activations = f.get_activations()
            num_a = len(activations)

            img_id = f.get_images_id()
            norm_activations = f.get_norm_activations()
            locations = f.get_locations()
            total_act = 0
            for i in xrange(num_a):
                img_name = img_id[i]
                norm_act = norm_activations[i]
                xy = locations[i]

                im_crop = get_crop_image(dataset, xy, img_name, model, layer)

                w, h = im_crop.size

                if type(total_act) is not np.ndarray:
                    im_crop = image.img_to_array(im_crop)
                    total_act = im_crop
                else:
                    if total_act.shape[0] > h or total_act.shape[1] > w:
                        im_crop = im_crop.resize((total_act.shape[1], total_act.shape[0]))
                    elif total_act.shape[0] < h or total_act.shape[1] < w:
                        tmp = image.array_to_img(total_act, scale=False)
                        total_act = tmp.resize((w, h))
                        total_act = image.img_to_array(total_act)

                    im_crop = image.img_to_array(im_crop)
                    total_act = total_act + (im_crop * norm_act)

            f.set_nf(image.array_to_img(total_act/num_a))




    # prints for debugging

    # number of filters (neurons)
    # print len(filters)

    # print all neurons
    # for f in filters:
    #     print f

    # print the first neuron in a layer
    # ff = filters[0]
    # x = len(ff.activations)
    # for i in range(x):
    #     print ff.images[i], ff.activations[i], ff.xy[i]
    #
    # # print all neuron features (NF) values, from the same layer
    # for f in filters:
    #     print f.neuron_feature

    return filters


def get_crop_image(dataset, xy, img_name, model, layer):
    x = xy[0]
    y = xy[1]
    row_ini, row_fin, col_ini, col_fin = get_image_receptive_field(x, y, model, layer)

    img = image.load_img(dataset.src_dataset + img_name, target_size=dataset.target_size)

    im_crop = img.crop((col_ini, row_ini, col_fin+1, row_fin+1))
    return im_crop


def get_image_receptive_field(x, y, model, layer):
    from keras.layers.convolutional import Conv2D
    from keras.layers.pooling import _Pooling2D

    # input_layer = model.layers[0]
    # _, init_size, _, _ = input_layer.output_shape

    current_layer_idx = find_layer_idx(model, layer_name=layer)
    # _, current_size, _, _ = model.layers[current_layer_idx].output_shape

    row_ini = x
    col_ini = y

    total_padding = 0

    row_fin = row_ini
    col_fin = col_ini

    for i in xrange(current_layer_idx, -1, -1):
        current_layer = model.layers[i]
        _, current_size, _, _ = current_layer.input_shape

        if row_ini < 0:
            row_ini = 0
        if col_ini < 0:
            col_ini = 0

        if row_fin > current_size - 1:
            row_fin = current_size - 1
        if col_fin > current_size - 1:
            col_fin = current_size - 1

        if isinstance(current_layer, Conv2D) or isinstance(current_layer, _Pooling2D):
            config_params = current_layer.get_config()
            padding = config_params['padding']
            strides = config_params['strides'][0]
            kernel_size = config_params.get('kernel_size', config_params.get('pool_size'))[0]

            # if padding == 'same':  # padding = same, means input shape = output shape
            #     padding = (kernel_size - 1) / 2
            #     total_padding += padding
            # else:
            #     padding = 0

            # This is a workaround for VGG model translated from Matlab. For any sequential keras model, comment these
            # lines and uncomment the above.
            if i < 5:
                padding = 0
            else:
                padding = 1


            row_ini = row_ini*strides
            col_ini = col_ini*strides

            row_fin = row_fin*strides + kernel_size-1
            col_fin = col_fin*strides + kernel_size-1

            row_ini -= padding
            col_ini -= padding

            row_fin -= padding
            col_fin -= padding


    #     print 'Layer:', current_layer.name, ' ri:', row_ini, ' rf:', row_fin, ' ci:', col_ini, ' cf:', col_fin
    #     print 'RF size: ', row_fin - row_ini, col_fin - col_ini
    # print 'Final values: ', row_ini, row_fin, col_ini, col_fin

    return row_ini, row_fin, col_ini, col_fin


def find_layer_idx(model, layer_name):
    """Looks up the layer index corresponding to `layer_name` from `model`.

    Args:
        model: The `keras.models.Model` instance.
        layer_name: The name of the layer to lookup.

    Returns:
        The layer index if found. Raises an exception otherwise.
    """
    layer_idx = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            layer_idx = idx
            break

    if layer_idx is None:
        raise ValueError("No layer with name '{}' within the model".format(layer_name))
    return layer_idx