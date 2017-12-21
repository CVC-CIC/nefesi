import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

from external.vgg_matconvnet import VGG
from nefesi.neuron_feature import get_image_receptive_field

PATH_FILE = '/home/oprades/ImageNet/train/'
model = VGG()
act_layers = ['activation_1', 'activation_2']


def save_topNimages():

    filters = pickle.load(open('activation_1.obj', 'rb'))

    for n_filter in xrange(len(filters)):
        activations = zip(*filters[n_filter].activations)

        xy = activations[2]
        n_images = activations[0]
        activations = activations[1]
        res_images = []
        titles = []

        for i in xrange(100):
            x = xy[i][0]
            y = xy[i][1]

            row_ini, row_fin, col_ini, col_fin = get_image_receptive_field(x, y, model, 'activation_1')
            img = image.load_img(PATH_FILE + n_images[i], target_size=(224, 224))
            im_crop = img.crop((col_ini, row_ini, col_fin + 1, row_fin + 1))
            res_images.append(im_crop)
            titles.append(activations[i])

        cols = 10
        n_images = len(res_images)
        # titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
        fig = plt.figure()
        for n, (img, title) in enumerate(zip(res_images, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
            # if image.ndim == 2:
            #     plt.gray()
            plt.imshow(img, interpolation='bicubic')
            plt.axis('off')
            # a.set_title(title)
        # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        fig.savefig('topNimages/activation_1/L1_' + str(n_filter) + '.png')
        fig.clear()


def save_NFimages_color_index():
    from PIL import ImageOps

    for layer_n in act_layers:
        ci = np.loadtxt('new_results/index_color/' + layer_n + '.txt', delimiter=',')
        filters = pickle.load(open('new_results/filters_res/' + layer_n + '.obj', 'rb'))

        images_ci = [[] for _ in xrange(3)]
        titles_ci = [[] for _ in xrange(3)]

        for i in xrange(len(filters)):
            idx = int(ci[i, 0])
            res = ci[i, 1]

            if not math.isnan(res):

                if res <= 0.10: # non-color
                    images_ci[0].append(filters[idx].neuron_feature)
                    titles_ci[0].append(idx)

                elif res >= 0.25: # color
                    images_ci[1].append(filters[idx].neuron_feature)
                    titles_ci[1].append(idx)
                else: # low-color
                    images_ci[2].append(filters[idx].neuron_feature)
                    titles_ci[2].append(idx)

        for i in xrange(3):

            images = images_ci[i]
            titles = titles_ci[i]

            if i == 0: # low-color (only plot 100 images)
                images = images[:100]
                titles = titles[:100]

            cols = int(math.sqrt(len(images)))
            n_images = len(images)
            # titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
            fig = plt.figure()
            for n, (img, title) in enumerate(zip(images, titles)):
                a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
                # if image.ndim == 2:
                #     plt.gray()
                img = ImageOps.autocontrast(img)
                plt.imshow(img, interpolation='bicubic')
                plt.axis('off')
                a.set_title(title)
            # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
            fig.savefig('new_results/index_color/bicubic_NF_L' + layer_n + str(i) + '.png')

def save_neuron_feature(layer_name):

    filters = pickle.load(open('activation_1.obj', 'rb'))

    for fil in filters:
        nf = fil.neuron_feature
        nf.save('NF/' + layer_name + '/NF_' + str(filters.index(fil))+'.png', 'PNG')


if __name__ == '__main__':

    save_topNimages()

    # save_NFimages_color_index()

    # save_neuron_feature('activation_1')



