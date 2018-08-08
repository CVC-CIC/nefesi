import numpy as np
import random
import matplotlib.pyplot as plt

def get_n_circles_well_distributed(idx_values,color_map='jet', max_radius=100):
    cmap = plt.cm.get_cmap(color_map)
    bins, count = np.histogram(idx_values,bins='fd')
    bins, count = bins[::-1],count[::-1] #Gets better distribution
    max_x_size = np.max(bins)*max_radius
    every_bin_y_size = (np.sqrt(bins)*max_radius).astype(np.int)
    regions = np.zeros((len(bins), 2), dtype=np.int)
    regions[0,1] = every_bin_y_size[0]
    for i in range(1, len(every_bin_y_size)):
        regions[i,0] = regions[i-1,1]-min(every_bin_y_size[i]//2, regions[i-1,1]//2,(max_radius//2)+1)
        regions[i,1] = regions[i,0]+every_bin_y_size[i]

    max_y_size = regions[-1,1]
    max_size = max(max_y_size, max_x_size)
    positions = np.zeros(len(idx_values),dtype=np.dtype([('circle',np.object),
                                                     ('x0', np.float),('x1',np.float),('y0',np.float),('y1',np.float),
                                                         ('x_center',np.float), ('y_center',np.float)]))
    print(regions)
    for i in range(len(regions)):
        for j in range(bins[i]):
            radius = count[i] * (max_radius/2)
            flag=True
            while flag:
                y_place = random.choice(range(regions[i, 0], regions[i, 1]))
                x_place = random.choice(range(max_size))
                y0 = y_place-(max_radius/2)
                x0 = x_place-(max_radius/2)
                for position in positions[:np.sum(bins[:i])+j]:
                    if collisionDetect(x0,position['x0'], y0, position['y0'],max_radius):
                        flag=True
                        break
                else:
                    x1 = x_place+(max_radius/2)
                    y1 = y_place + (max_radius / 2)
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