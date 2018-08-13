import numpy as np
import random
import matplotlib.pyplot as plt

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


def destroy_canvas_subplot_if_exist(master_canvas):
    if '!canvas' in master_canvas.children:
        oldplot = master_canvas.children['!canvas']
        clean_widget(widget=oldplot)
        oldplot.destroy()