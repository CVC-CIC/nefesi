import numpy as np
import random
import matplotlib.pyplot as plt
import math
import os
import shutil
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

    return positions



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


def move_to_root_folder(root_path, cur_path):
    for filename in os.listdir(cur_path):
        if os.path.isfile(os.path.join(cur_path, filename)):
            return 1
        elif os.path.isdir(os.path.join(cur_path, filename)):
            move_to_root_folder(root_path, os.path.join(cur_path, filename))
        else:
            sys.exit("Should never reach here.")

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