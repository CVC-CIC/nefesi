import numpy as np
import random
import matplotlib.pyplot as plt
import math
import os
import shutil
import PIL
from anytree import Node, RenderTree
import xml.etree.ElementTree as ET
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

def get_hierarchy_of_label(labels, freqs, xml, population_code=0, class_sel=0):
    if type(xml) is str:
        xml = ET.parse(xml)
    humanLists = []
    for label, freq in zip(labels, freqs):
        xPath = './/synset[@wnid="'+label+'"]'
        result = xml.find(xPath)
        humanList = []
        while len(result.attrib) is not 0:

            humanList.append((result.attrib['words'],freq))
            xPath+='/..'
            result = xml.find(xPath)
        humanLists.append(humanList)

    hierarchy = {'root': Node('root', freq=class_sel, rep=population_code)}
    for humanList in humanLists:
        parent = humanList[-1][0]
        if parent not in hierarchy:
            hierarchy[parent] = Node(parent,parent=hierarchy['root'], freq = humanList[-1][1], rep=1)
        else:
            hierarchy[parent].freq += humanList[-1][1]
            hierarchy[parent].rep += 1
        for i in range(len(humanList)-2,-1,-1):
            if humanList[i][0] not in hierarchy:
                hierarchy[humanList[i][0]] = Node(humanList[i],parent=hierarchy[parent], freq = humanList[-1][1], rep=1)
            else:
                hierarchy[humanList[i][0]].freq += humanList[-1][1]
                hierarchy[humanList[i][0]].rep += 1
            parent = humanList[i][0]

    return hierarchy['root']


def get_image_masked(network_data, image_name,layer_name,neuron_idx, as_numpy = False, type=1, thr_mth = 1, thr = 0.005):
    """
    Returns the image correspondant to image_name with a mask of the place that most response has for the neuron
    neuron_idx of layer layer_name
    :param network_data: Network_data object representing the nefesi network
    :param image_name: the name of the image to analyze
    :param layer_name: the name of the layer of the network where is the neuron to analyze
    :param neuron_idx: the index of the neuron to analyze
    :param as_numpy: get the result as a numpy array
    :param type: 1 as torralba, 2 as vedaldi  (falta posar referencies), type 3  as activation itself
    :param thr_mth = take max from the image (1), take max from all activatiosn (0)
    :return: An image that is the original image with a mask of the activation camp superposed
    """
    input = network_data.dataset._load_image(image_name, as_numpy=True,
                                                  prep_function=True)[np.newaxis, ...]
    activations = get_one_neuron_activations(model=network_data.model, layer_name=layer_name,
                                             idx_neuron=neuron_idx, model_inputs=input)[0]

    if thr_mth==0:
        max_act = np.max(network_data.get_layer_by_name(layer_name).neurons_data[neuron_idx].activations)
    else:
        max_act = np.max(activations)
    norm_activations = activations / max_act


    sz_img = np.array((224, 224))
    if type == 2 or type ==4:
        norm_activations_upsampled = np.array(PIL.Image.fromarray(norm_activations).resize(tuple(sz_img), PIL.Image.BILINEAR))
    elif type ==1 or type==3:
        # vertex = lambda a, b, c, d: [a, b, a, d, c, d, c, b]
        #
        #     1 | 4
        #   ----------
        #     2 | 3
        #
        # ci1=[0,0]
        # ci2=[0,0]
        # ci3=[0,0]
        # ci4=[0,0]
        # if sz_img[0] % 2:  #senar
        #     ci1[0],ci2[0],ci3[0],ci4[0] = sz_img[0]//2
        # else: # parell
        #     ci1[0] = sz_img[0]//2-1
        #     ci4[0] = sz_img[0]//2-1
        #     ci2[0] = sz_img[0]//2
        #     ci3[0] = sz_img[0]//2
        # if sz_img[1] % 2:  #senar
        #     ci1[1],ci2[1],ci3[1],ci4[1] = sz_img[1]//2
        # else: # parell
        #     ci1[1] = sz_img[1]//2-1
        #     ci2[1] = sz_img[1]//2-1
        #     ci3[1] = sz_img[1]//2
        #     ci4[1] = sz_img[1]//2
        #
        # sz_act = np.array(norm_activations.shape)
        # ca1=[0,0]
        # ca2=[0,0]
        # ca3=[0,0]
        # ca4=[0,0]
        # if sz_act[0] % 2:  #senar
        #     ca1[0],ca2[0],ca3[0],ca4[0] = sz_act[0]//2
        # else: # parell
        #     ca1[0] = sz_act[0]//2-1
        #     ca4[0] = sz_act[0]//2-1
        #     ca2[0] = sz_act[0]//2
        #     ca3[0] = sz_act[0]//2
        # if sz_img[1] % 2:  #senar
        #     ca1[1],ca2[1],ca3[1],ca4[1] = sz_act[1]//2
        # else: # parell
        #     ca1[1] = sz_act[1]//2-1
        #     ca2[1] = sz_act[1]//2-1
        #     ca3[1] = sz_act[1]//2
        #     ca4[1] = sz_act[1]//2
        #
        # a = [[     0,     0,       ci1[1],      ci1[0]]]
        # b = [[     0, ci2[0],      ci2[1], sz_img[0]-1]]
        # c = [[ci3[1], ci3[0], sz_img[1]-1, sz_img[0]-1]]
        # d = [[ci4[1],      0, sz_img[1]-1,      ci4[0]]]
        # a.append(vertex(     0,      0,      ca1[1],      ca1[0]))
        # b.append(vertex(     0, ca2[0],      ca2[1], sz_act[0]-1))
        # c.append(vertex(ca3[1], ca3[0], sz_act[1]-1, sz_act[0]-1))
        # d.append(vertex(ca4[1],      0, sz_act[1]-1,      ca4[0]))
        #
        # sz_act = np.array(norm_activations.shape)
        # layer = network_data.get_layer_by_name(layer_name)
        # sz_act = np.array(layer.receptive_field_map.shape[:-1])
        # ct_act = sz_act //2
        # ct_img = np.array(layer.receptive_field_map[ct_act[0], ct_act[1], :])
        # ct_img = np.array([ct_img[1]-ct_img[0],ct_img[3]-ct_img[2]])//2
        # ct_img = sz_img // 2
        #
        # a = ((        0,         0,   ct_img[1],   ct_img[0]), (        0,         0,         0,   ct_act[0],   ct_act[1],   ct_act[0],   ct_act[1],         0))
        # b = ((        0, ct_img[0],   ct_img[1], sz_img[0]-1), (        0, ct_act[0],         0, sz_act[0]-1,   ct_act[1], sz_act[0]-1,   ct_act[1], ct_act[0]))
        # c = ((ct_img[1], ct_img[0], sz_img[1]-1, sz_img[0]-1), (ct_act[1], ct_act[0], ct_act[1], sz_act[0]-1, sz_act[1]-1, sz_act[0]-1, sz_act[1]-1, ct_act[0]))
        # d = ((ct_img[1],         0, sz_img[1]-1,   ct_img[0]), (ct_act[1],         0, ct_act[1],   ct_act[0], sz_act[1]-1,   ct_act[0], sz_act[1]-1,         0))

        # rec_field_map = network_data.get_layer_by_name(layer_name).receptive_field_map
        # rec_field_sz = network_data.get_layer_by_name(layer_name).receptive_field_size
        # mesh = []
        # for y in range(rec_field_map.shape[0]):
        #     for x in range(rec_field_map.shape[1]):
        #         r = rec_field_map[y, x][[2, 0, 3, 1]]
        #         if r[0] == 0:
        #             r[0] = r[2]-rec_field_sz[1]
        #         if r[1] == 0:
        #             r[1] = r[3] - rec_field_sz[0]
        #         if r[2] == sz_img[1]:
        #             r[2] = r[0] + rec_field_sz[1]-1
        #         if r[3] == sz_img[0]:
        #             r[3] = r[1] + rec_field_sz[0] - 1
        #         mesh.append([list(r)] + [vertex(x, y, x, y)])
        #
        # norm_activations_upsampled = np.array(PIL.Image.fromarray(norm_activations).transform(tuple(sz_img), PIL.Image.MESH, mesh, PIL.Image.BILINEAR))


        rec_field_map = network_data.get_layer_by_name(layer_name).receptive_field_map
        pos = np.zeros(list(activations.shape)[::-1]+[2])
        for y in range(activations.shape[0]):
            for x in range(activations.shape[1]):
                rec = rec_field_map[y, x]
                pos[y,x,1] = math.floor((rec[1]+rec[0])/2)
                pos[y,x,0] = math.floor((rec[3]+rec[2])/2)
        from scipy.interpolate import griddata

        gx, gy = np.mgrid[0:sz_img[1], 0:sz_img[0]]
        norm_activations_upsampled = griddata(pos.reshape((-1, 2)), activations.reshape(-1), (gx, gy), method='cubic', fill_value=0).T
        norm_activations_upsampled[norm_activations_upsampled<0] = 0
        norm_activations_upsampled *= np.max(activations)/np.max(norm_activations_upsampled)
        norm_activations_upsampled /= max_act

    if type > 2:
        img = norm_activations_upsampled * 255
        img = np.dstack((img, img, img))
    else:
        img = network_data.dataset._load_image(image_name, as_numpy=True).astype(np.float)
        img[norm_activations_upsampled < thr] *= 0.25

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