import numpy as np
from PIL import Image
from interface_DeepFramework.image_processing import array_to_img
import matplotlib.pyplot as plt
def compute_nf(network_data, layer_data,  only_if_not_done=False):
    """This function build the neuron features (NF) for all neurons
    in `filters`.

    :param network_data: The `nefesi.network_data.NetworkData` instance.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    """



    for i, neuron in enumerate(layer_data.neurons_data):
        if not only_if_not_done or neuron.neuron_feature is None:
            neuron.norm_activations = neuron.activations / abs(neuron.activations)[0]
            if neuron.norm_activations is not None:
                norm_activations = neuron.norm_activations


                patches = neuron.get_patches(network_data, layer_data)



            else:
                print('why???')


        nf = np.sum(patches.reshape(patches.shape[0], -1) * (norm_activations / np.sum(norm_activations))[:, np.newaxis], axis=0).reshape(patches.shape[1:])
        neuron.neuron_feature = nf



def compute_nf_out(network_data, layer_data, model, only_if_not_done=False):
    """This function build the neuron features (NF) for all neurons
    in `filters`.

    :param network_data: The `nefesi.network_data.NetworkData` instance.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    """



    for i, neuron in enumerate(layer_data.neurons_data):
        if not only_if_not_done or neuron.neuron_feature_out is None:
            neuron.norm_activations = neuron.activations / abs(neuron.activations)[0]
            if neuron.norm_activations is not None:
                norm_activations = neuron.norm_activations

                # get the receptive fields from a neuron
                #max_rf_size = (float('Inf'), float('Inf'))

                patches = neuron.get_patches_out(network_data, layer_data,model)

            else:
                print('why???')


        nf = np.sum(patches.reshape(patches.shape[0], -1) * (abs(norm_activations) / np.sum(abs(norm_activations)))[:, np.newaxis], axis=0).reshape(patches.shape[1:])
        neuron.neuron_feature_out = nf


def add_padding(pil_img,padding=0, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + padding + padding
    new_height = height + padding + padding
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (padding, padding))
    return result


def compute_nf_guillem(network_data, layer_name, verbose=True,only_if_not_done=True):
    """This function build the neuron features (NF) for all neurons
    in `filters`.

    :param network_data: The `nefesi.network_data.NetworkData` instance.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    """
    layer_data=network_data.get_layer_by_name(layer_name)
    rf= layer_data.receptive_field_size
    stride=layer_data.stride
    pading=layer_data.pading
    im_size=network_data.dataset.target_size
    im_folder = network_data.dataset.src_dataset
    for i, neuron in enumerate(layer_data.neurons_data):
        nf= np.zeros((rf,rf,3))
        crops=[]
        if not only_if_not_done or neuron.neuron_feature is None:
            if neuron.norm_activations is not None:
                for n,im_name in enumerate(neuron.images_id):
                    im=Image.open(im_folder+im_name)
                    im=im.resize(im_size)
                    im=add_padding(im,pading)
                    xy=neuron.xy_locations[n]
                    crop= im.crop([xy[0]*stride, xy[1]*stride,xy[0]*stride+rf, xy[1]*stride+rf])
                    crops.append(crop)

                    nf= nf+crop*neuron.norm_activations[n]/100
                all_patches=Image.new('RGB',(rf*10,rf*10),(0,0,0))

                for numb in range(100):
                    all_patches.paste(crops[numb], ( rf*(numb//10), rf*(numb%10) ))


                plt.imshow(all_patches)
                plt.show()

                print(neuron)










                if verbose and i%10==0:
                    print("NF - "+layer_data.layer_id+". Neurons completed: "+str(i)+"/"+str(len(layer_data.neurons_data)))
            else:
                # if `norm_activations` from a neuron is None, that means this neuron
                # doesn't have activations. NF is setting with None.
                neuron.neuron_feature = None

