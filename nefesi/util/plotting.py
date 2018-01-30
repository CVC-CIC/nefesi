import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
from PIL import ImageOps
from sklearn.manifold import TSNE


def plot_sel_idx_summary(selectivity_idx, bins=10):
    # TODO: change the map color of bars
    # TODO: check what kind of index it is. For example, for symmetry plot only the avg

    for k, v in selectivity_idx.items():
        N = len(v)
        pos = 0

        for l in v:
            num_f = len(l)
            counts, bins = np.histogram(l, bins=bins, range=(0, 1))
            prc = np.zeros(len(counts))

            for i in xrange(len(counts)):
                prc[i] = float(counts[i])/num_f*100.

            y_offset = 0

            bars = []
            for i in xrange(len(prc)):
                p = plt.bar(pos, prc[i], bottom=y_offset, width=0.35)
                bars.append(p)
                y_offset = y_offset+prc[i]
            pos += 1

        xticks = []
        for i in xrange(N):
            xticks.append('Layer ' + str(i + 1))
        plt.xticks(np.arange(N), xticks)
        plt.yticks(np.arange(0, 101, 10))

        labels = [str(bins[i]) + ':' + str(bins[i+1]) for i in xrange(len(prc))]

        plt.ylabel('% of Neurons')
        plt.title(k + ' selectivity')
        plt.legend(bars, labels, bbox_to_anchor=(1.02, 1.02), loc=2)
        plt.subplots_adjust(right=0.75)
        plt.show()

def plot_top_scoring_images(network_data, layer_id, neuron_idx, n_max=50):

    layers = network_data.get_layers()
    neuron = None
    for l in layers:
        if l.get_layer_id() in layer_id:
            neuron = l.get_filters()[neuron_idx]

    if neuron is None:
        print 'Some msg error'
        return

    images = neuron.get_patches(network_data, layer_id)
    activations = neuron.get_norm_activations()
    images = images[:n_max]
    activations = activations[:n_max]

    cols = int(math.sqrt(len(images)))
    n_images = len(images)
    titles = [round(act, 2) for act in activations]
    fig = plt.figure()
    for n, (img, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.imshow(img, interpolation='bicubic')
        plt.axis('off')
        # a.set_title(title)
    # fig.set_size_inches(n_max*3,n_max*3)
    plt.show()
    fig.clear()

def plot_activation_curve(network_data, layer_id, neuron_idx, num_images=2):

    layers = network_data.get_layers()
    neuron = None
    for l in layers:
        if l.get_layer_id() in layer_id:
            neuron = l.get_filters()[neuron_idx]

    if neuron is None:
        print 'Some msg error'
        return

    images = neuron.get_patches(network_data, layer_id)
    activations = neuron.get_norm_activations()
    print activations

    idx_images = np.arange(0, len(images), num_images)
    cols = len(idx_images)

    fig = plt.figure()

    for n, img_idx in enumerate(idx_images):
        img = images[img_idx]
        t = round(activations[img_idx], 2)
        a = fig.add_subplot(2, cols, n + 1)
        plt.imshow(img, interpolation='bicubic')
        plt.axis('off')
        a.set_title(t)

    fig.add_subplot(2,1,2)
    plt.plot(activations)


    # plt.yticks(np.arange(0, 1.1, 0.1))

    plt.show()

def plot_pixel_decomposition(activations, neurons, img):
    nf = []
    for n in neurons:
        nf.append(n.get_neuron_feature())

    print nf

    idx_images = np.arange(0, len(nf))
    cols = len(idx_images)

    fig = plt.figure()
    fig.add_subplot(2,1,1)
    plt.imshow(img)
    plt.axis('off')

    for n, img_idx in enumerate(idx_images):
        img = nf[img_idx]
        t = round(activations[img_idx], 2)
        a = fig.add_subplot(2, cols, n + cols+1)
        plt.imshow(img, interpolation='bicubic')
        plt.axis('off')
        a.set_title(t)

    # plt.yticks(np.arange(0, 1.1, 0.1))

    plt.show()

def plot_decomposition(activations, neurons, locations, img):
    nf = []
    offset = 1
    for n in neurons:
        nf.append(n.get_neuron_feature())

    for i in xrange(len(activations)-1, -1, -1):
        ri, rf, ci, cf = locations[i]
        nf[i] = ImageOps.expand(nf[0], offset, fill='white')
        img.paste(nf[i], (ri-offset, ci-offset, rf+offset, cf+offset))

    plt.imshow(img)
    plt.axis('off')
    plt.show()

def plot_similarity_tsne(layer_data):

    # obtain similarity matrix from layer
    # x = similarity_matrix

    neurons = layer_data.get_filters()
    num_neurons = len(neurons)

    x = np.random.rand(num_neurons, num_neurons)

    x_result = TSNE(n_components=2, metric='precomputed', random_state=0).fit_transform(x)

    nf = [n.get_neuron_feature() for n in neurons]
    fig, ax = plt.subplots()

    for i, x, y in zip(range(num_neurons), x_result[:, 0], x_result[:, 1]):
        print x, y
        # plt.scatter(x, y)
        # plt.imshow(nf[i], interpolation='bicubic')
        imscatter(x, y, nf[i], zoom=3, ax=ax, label=str(i))
        ax.plot(x, y)
    plt.axis('off')
    plt.show()


def imscatter(x, y, image, ax=None, zoom=1, label=None):
    if ax is None:
        ax = plt.gca()

    im = OffsetImage(image, zoom=zoom, interpolation='bicubic')
    x, y = np.atleast_1d(x, y)

    ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)

    # ax.text(x, y, label) #TODO: put labels (number of the neuron)
    ax.add_artist(ab)
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


def plot_nf_search(selective_neurons):
    index_name = selective_neurons.keys()
    layers_v = selective_neurons.values()[0]

    for k, v in layers_v.items():
        layer_name = k
        neurons = v

        neurons = sorted(neurons, key=lambda x: x.selectivity_idx.get(index_name[0]))

        cols = int(math.sqrt(len(neurons)))
        n_images = len(neurons)
        titles = [round(n.selectivity_idx.get(index_name[0]), 2) for n in neurons]

        fig = plt.figure()
        fig.suptitle('Layer: ' + layer_name)

        for i, (n, title) in enumerate(zip(neurons, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), i + 1,)
            plt.imshow(n.get_neuron_feature(), interpolation='bicubic')
            plt.axis('off')
            a.set_title(title)
        plt.show()
        fig.clear()


def main():
    from keras.applications.vgg16 import VGG16
    from keras.preprocessing.image import load_img
    import pickle
    from image import ImageDataset
    from image import rotate_images


    # sel_idx = dict()
    # sel_idx['color'] = []
    # sel_idx['color'].append(np.random.rand(96))
    # sel_idx['color'].append(np.random.rand(128))
    #
    # sel_idx['symmetry'] = []
    # a = []
    # for i in xrange(96):
    #     a.append(np.random.rand(5))
    # sel_idx['symmetry'].append(a)
    # a = []
    # for i in xrange(128):
    #     a.append(np.random.rand(5))
    # sel_idx['symmetry'].append(a)


    # plot_sel_idx_summary(sel_idx)

    dataset = '/home/oprades/ImageNet/train/'  # dataset path
    save_path = '/home/oprades/oscar/oscar/'
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    num_max_activations = 100

    model = VGG16()

    my_net = pickle.load(open(save_path + 'vgg16.obj', 'rb'))
    my_net.model = model
    img_dataset = ImageDataset(dataset, (224, 224))
    my_net.dataset = img_dataset
    my_net.save_path = save_path

    # plot_top_scoring_images(my_net, layer_names[2], 5, n_max=100)

    # plot_activation_curve(my_net, layer_names[2], 5)

    activations = np.random.random(2)
    idx = np.argsort(activations)
    idx = idx[::-1]
    activations = activations[idx]

    neurons = my_net.get_layers()[0].get_filters()[0:2]

    img = load_img(dataset + 'n01440764/n01440764_97.JPEG', target_size=(224, 224))

    # plot_pixel_decomposition(activations, neurons, img)

    locations = [(0, 5, 0, 5), (15, 20, 124, 129)]
    # plot_decomposition(activations, neurons, locations, img)

    # l = my_net.get_layers()[0]
    # plot_similarity_tsne(l)

    selective_neurons = my_net.get_selective_neurons('color', layer_names[0:2], inf_thr=0.5)
    plot_nf_search(selective_neurons)


    # my_net.get_layers()[3].mapping_rf(model, 28, 28)
    # print my_net.get_layers()[3].receptive_field_map
    #
    # from keras.models import Model
    # int_layer = Model(inputs=model.input,
    #                   outputs=model.get_layer(layer_names[3]).output)
    #
    # i = np.random.rand(1,224,224,3)
    # print int_layer.predict(i).shape
    #
    # print my_net.get_layers()[3].get_filters()[504].get_neuron_feature().size
    # print my_net.get_layers()[3].get_filters()[504].print_params()
    #
    # my_net.get_layers()[3].get_filters()[504].get_neuron_feature().show()
    #
    # from nefesi.neuron_feature import get_image_receptive_field
    # print get_image_receptive_field(27,27,model, layer_names[3])

    # n = my_net.get_layers()[0].get_filters()[15]
    # n.test()
    # patches = n.get_patches(my_net, layer_names[0])
    # patches[0].show()

    # o_idx = my_net.selectivity_idx_summary(['orientation'], layer_names[0])
    # o_idx = o_idx['orientation']
    # o_idx_layer0 = o_idx[0]  # list of 64 neurons indexes
    # l = np.array(o_idx_layer0)
    # print l.shape
    #
    # max = np.amax(l[:,3])
    # idx = np.unravel_index(l[:,3].argmax(), l[:,3].shape)
    #
    # print max, idx
    #
    # print l[:, 3]
    #
    # f = my_net.get_layers()[0].get_filters[57]
    # f.get_neuron_feature.show()









if __name__ == '__main__':
    main()