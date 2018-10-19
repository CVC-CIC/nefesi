import numpy as np
import random
import matplotlib.pyplot as plt
import math
import os
import shutil
import PIL
from ..symmetry_index import SYMMETRY_AXES
from ..read_activations import get_one_neuron_activations

try:
    from tkinter import *
    from tkinter import ttk
    from tkinter import filedialog
except ImportError:
    from Tkinter import *
    from Tkinter import ttk


def get_n_circles_well_distributed(idx_values, color_map='jet', diameter=100):
    cmap = plt.cm.get_cmap(color_map)
    bins, count = np.histogram(idx_values,bins='fd')
    bins, count = bins[::-1],count[::-1] #Gets better distribution
    max_x_size = np.max(bins) * diameter
    every_bin_y_size = (np.sqrt(bins) * diameter).astype(np.int)
    regions = np.zeros((len(bins), 2), dtype=np.int)
    regions[0,1] = every_bin_y_size[0]
    for i in range(1, len(every_bin_y_size)):
        regions[i,0] = regions[i-1,1]-min(every_bin_y_size[i] // 2, regions[i-1,1] // 2, (diameter // 2) + 1)
        regions[i,1] = regions[i,0]+every_bin_y_size[i]

    max_y_size = regions[-1,1]
    max_size = max(max_y_size, max_x_size)
    positions = np.zeros(len(idx_values),dtype=np.dtype([('circle',np.object),
                                                     ('x0', np.float),('x1',np.float),('y0',np.float),('y1',np.float),
                                                         ('x_center',np.float), ('y_center',np.float)]))
    radius = diameter / 2
    for i in range(len(regions)):
        for j in range(bins[i]):
            flag=True
            while flag:
                y_place = random.choice(range(regions[i, 0], regions[i, 1]))
                x_place = random.choice(range(max_size))
                y0 = y_place-radius
                x0 = x_place-radius
                for position in positions[:np.sum(bins[:i])+j]:
                    if collisionDetect(x0, position['x0'], y0, position['y0'], diameter):
                        flag=True
                        break
                else:
                    x1 = x_place+radius
                    y1 = y_place + radius
                    circle = plt.Circle((x_place, y_place), radius=radius, alpha=0.6,
                                        color=cmap(i/float(len(regions))), linewidth=3)
                    positions[np.sum(bins[:i])+j] = (circle,x0,x1,y0,y1,x_place,y_place)
                    flag=False
                    """
                     #image_file = PIL.Image.open('../nefesi/util/bg_image.jpg')#cbook.get_sample_data('bg_image.jpg')
                    if i%2 ==0:
                        image = plt.imread('../nefesi/util/bg_image.jpg')
                    else:
                        image = plt.imread('../nefesi/util/bg_image2.jpg')
                    ax = plt.gca()#plt.axes((x_place-radius,y_place-radius,radius*2,radius*2))
                    im = ax.imshow(image)
                    x1 = x_place+radius
                    y1 = y_place + radius
                    circle = plt.Circle((x_place, y_place), radius=radius, alpha=0.6,
                                         linewidth=3, transform=ax.transData)
                    im.set_clip_path(circle)
                    positions[np.sum(bins[:i])+j] = (circle,x0,x1,y0,y1,x_place,y_place)
                    flag=False
                    """

    return positions


def get_image_masked(network_data, image_name,layer_name,neuron_idx, as_numpy = False):
    """
    Returns the image correspondant to image_name with a mask of the place that most response has for the neuron
    neuron_idx of layer layer_name
    :param network_data: Network_data object representing the nefesi network
    :param image_name: the name of the image to analyze
    :param layer_name: the name of the layer of the network where is the neuron to analyze
    :param neuron_idx: the index of the neuron to analyze
    :param as_numpy: get the result as a numpy array
    :return: An image that is the original image with a mask of the activation camp superposed
    """
    input = network_data.dataset._load_image(image_name, as_numpy=True,
                                                  prep_function=True)[np.newaxis, ...]
    activations = get_one_neuron_activations(model=network_data.model, layer_name=layer_name,
                                             idx_neuron=neuron_idx, model_inputs=input)[0]

    norm_activations = activations / np.max(activations)
    norm_activations_upsampled = np.array(PIL.Image.fromarray(norm_activations).resize((224, 224), PIL.Image.BILINEAR))

    img = network_data.dataset._load_image(image_name, as_numpy=True).astype(np.float)
    img[norm_activations_upsampled < 0.005] *= 0.25
    if as_numpy:
        return img
    else:
        return PIL.Image.fromarray(img.astype('uint8'), 'RGB')



def collisionDetect(x1, x2, y1, y2, margin):
    if x1 > x2 and x1 < x2 + margin or x1 + margin > x2 and x1 + margin < x2 + margin:
        if y1 > y2 and y1 < y2 + margin or y1 + margin > y2 and y1 + margin < y2 + margin:
            return True
    return False


def clean_widget(widget):
    for child in list(widget.children.values()):
        if list(child.children.values()) == []:
            child.destroy()
        else:
            clean_widget(child)
def addapt_ADE20K_dataset(dataset_base_path):
    def _addapt_dataset(dataset_base_path, first_base, file_name=None):
        dirs = os.listdir(dataset_base_path)
        for file in dirs:
            if os.path.isdir(dataset_base_path+'/'+file):
                _addapt_dataset(dataset_base_path+'/'+file,first_base, file_name=file)
            else:

                if file.endswith('.png'):
                    dst_dir = first_base + '/masks/' + file_name
                    if not os.path.exists(dst_dir):
                        os.mkdir(dst_dir)
                    os.rename(dataset_base_path+'/'+file, dst_dir+'/'+file)
                elif file.endswith('.jpg'):
                    dst_dir = first_base + '/imgs/' + file_name
                    if not os.path.exists(dst_dir):
                        os.mkdir(dst_dir)
                    os.rename(dataset_base_path + '/' + file, dst_dir + '/' + file)
                elif file.endswith('.txt'):
                    dst_dir = first_base + '/texts/' + file_name
                    if not os.path.exists(dst_dir):
                        os.mkdir(dst_dir)
                    os.rename(dataset_base_path + '/' + file, dst_dir + '/' + file)

    basic_dirs = os.listdir(dataset_base_path)
    if not os.path.exists(dataset_base_path+'/imgs'):
        os.mkdir(dataset_base_path+'/imgs')
    if not os.path.exists(dataset_base_path + '/masks'):
        os.mkdir(dataset_base_path + '/masks')
    if not os.path.exists(dataset_base_path + '/texts'):
        os.mkdir(dataset_base_path + '/texts')
    _addapt_dataset(dataset_base_path, dataset_base_path)
    for dir in basic_dirs:
        path = dataset_base_path+'/'+dir
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def destroy_canvas_subplot_if_exist(master_canvas):
    if '!canvas' in master_canvas.children:
        oldplot = master_canvas.children['!canvas']
        clean_widget(widget=oldplot)
        oldplot.destroy()

def mosaic_n_images(images):
    images_per_axis = math.ceil(math.sqrt(len(images)))
    mosaic = np.zeros((images.shape[-3]*images_per_axis, images.shape[-2]*images_per_axis, images.shape[-1]))
    #mosaic[:,:,0] = 255 #red
    for i,image in enumerate(images):
        y_offset,x_offset = i//images_per_axis, i%images_per_axis
        mosaic[y_offset*image.shape[0]:(y_offset+1)*image.shape[0],
        x_offset * image.shape[0]:(x_offset + 1) * image.shape[0],
                :] = image
    return mosaic

def add_red_separations(mosaic, images_per_axis):
    image_y_shape, image_x_shape = math.ceil(mosaic.shape[0]/images_per_axis), math.ceil(mosaic.shape[1]/images_per_axis)
    for i in range(1,images_per_axis):
        mosaic[:,(i*(image_x_shape)),:]=[255,0,0]
        mosaic[(i * (image_y_shape)),:, :] = [255, 0, 0]
    return mosaic


def addapt_widget_for_grid(widget):
    for i in range(3):
        Grid.columnconfigure(widget, i, weight=1)
        Grid.rowconfigure(widget, i, weight=1)


def get_listbox_selection(lstbox, selection = None):
    if selection is None:
        selection = lstbox.curselection()
    layers_selected = [lstbox.get(first=selection[i]) for i in range(len(selection))]
    if len(layers_selected) == 1 and layers_selected[0] == 'all':
        layers_selected = list(lstbox.get(1,END))
    return layers_selected

def get_key_of_index(key, special_value):
    if key == 'orientation':
        key+=str(int(special_value))
    elif key == 'symmetry':
        key+=str(SYMMETRY_AXES)
    elif key == 'population code':
        key+=str(round(special_value, 2))
    return key