import numpy as np

# from keras.preprocessing import image
# from PIL import Image
from .interface_DeepFramework.image_processing import array_to_img

def compute_nf(network_data, layer_data, verbose=True, maximize_contrast = False, mode = 1,threshold_to_noncount = 0.1, only_if_not_done=False):
    """This function build the neuron features (NF) for all neurons
    in `filters`.

    :param network_data: The `nefesi.network_data.NetworkData` instance.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    """

    if layer_data.receptive_field_map is None:
        layer_data.mapping_rf(network_data.model)

    for i, neuron in enumerate(layer_data.neurons_data):
        if not only_if_not_done or neuron.neuron_feature is None:
            if neuron.norm_activations is not None:
                norm_activations = neuron.norm_activations
                # get the receptive fields from a neuron
                #max_rf_size = (float('Inf'), float('Inf'))
                max_rf_size = network_data.model.layers[0].input_shape[1:3]
                patches = neuron.get_patches(network_data, layer_data, max_rf_size)
                masks = neuron.get_patches_mask(network_data, layer_data, max_rf_size)
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
                neuron.neuron_feature = array_to_img(nf)
                # neuron.neuron_feature = Image.fromarray(nf.astype('uint8'), mode='RGB')
                if verbose and i%50==0:
                    print("NF - "+layer_data.layer_id+". Neurons completed: "+str(i)+"/"+str(len(layer_data.neurons_data)))
            else:
                # if `norm_activations` from a neuron is None, that means this neuron
                # doesn't have activations. NF is setting with None.
                neuron.neuron_feature = None

