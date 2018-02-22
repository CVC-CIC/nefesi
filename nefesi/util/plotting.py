import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
from PIL import ImageDraw
from sklearn.manifold import TSNE


def plot_sel_idx_summary(selectivity_idx, bins=10, color_map='jet'):
    # TODO: check what kind of index it is. For example, for symmetry plot only the avg

    # avoid indexes lower than 0.0

    cmap = plt.cm.get_cmap(color_map)
    colors = []
    for i in xrange(bins):
        colors.append(cmap(1.*i/bins))

    for k, v in selectivity_idx.items():

        # if k == 'symmetry':
        #     k = 'global symmetry'
        #     idx = 4
        #     new_l = []
        #     for l in v:
        #         for f in l:
        #             new_l.append(f[idx])
        #         l = new_l
        #
        #
        # if k == 'orientation':
        #     k = 'global orientation'
        #     idx = 25



        N = len(v)
        pos = 0

        for l in v:
            counts, bins = np.histogram(l, bins=bins, range=(0, 1))
            num_f = sum(counts)
            prc = np.zeros(len(counts))

            for i in xrange(len(counts)):
                prc[i] = float(counts[i])/num_f*100.
            y_offset = 0

            bars = []
            for i in xrange(len(prc)):
                p = plt.bar(pos, prc[i], bottom=y_offset, width=0.35, color=colors[i])
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


def plot_symmetry_distribution_summary(selectivity_idx, color_map='jet'):

    bins = 4
    cmap = plt.cm.get_cmap(color_map)
    colors = []
    for i in xrange(bins):
        colors.append(cmap(1. * i / bins))

    for k, v in selectivity_idx.items():
        N = len(v)
        pos = 0

        for l in v:
            counts = [0, 0, 0, 0]

            for f in l:
                max_idx = max(f)
                counts[f.index(max_idx)] += 1
            print counts
            num_f = sum(counts)
            prc = np.zeros(len(counts))

            for i in xrange(len(counts)):
                prc[i] = float(counts[i]) / num_f * 100.
            y_offset = 0

            bars = []
            for i in xrange(len(prc)):
                p = plt.bar(pos, prc[i], bottom=y_offset, width=0.35, color=colors[i])
                bars.append(p)
                y_offset = y_offset + prc[i]
            pos += 1

        xticks = []
        for i in xrange(N):
            xticks.append('Layer ' + str(i + 1))
        plt.xticks(np.arange(N), xticks)
        plt.yticks(np.arange(0, 101, 10))

        labels = ['0', '45', '90', '135']

        plt.ylabel('% of Neurons')
        plt.title(k + ' selectivity')
        plt.legend(bars, labels, bbox_to_anchor=(1.02, 1.02), loc=2)
        plt.subplots_adjust(right=0.75)
        plt.show()


def plot_top_scoring_images(network_data, layer_data, neuron_idx, n_max=50):

    layers = network_data.get_layers()
    neuron = None
    for l in layers:
        if l.get_layer_id() in layer_data.layer_id:
            neuron = l.get_filters()[neuron_idx]

    if neuron is None:
        print 'Some msg error'
        return

    images = neuron.get_patches(network_data, layer_data)
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

def plot_activation_curve(network_data, layer_data, neuron_idx, num_images=5):


    neuron = layer_data.get_filters()[neuron_idx]

    if neuron is None:
        print 'Some msg error'
        return

    images = neuron.get_patches(network_data, layer_data)
    activations = neuron.get_norm_activations()

    idx_images = np.arange(0, len(images), num_images)
    cols = len(idx_images)

    fig = plt.figure()
    for n, img_idx in enumerate(idx_images):
        img = images[img_idx]
        t = round(activations[img_idx], 2)
        a = fig.add_subplot(2, cols, n + 1)
        plt.imshow(img)
        plt.axis('off')
        a.set_title(t)

    fig.add_subplot(2,1,2)
    plt.plot(activations)
    plt.ylabel('Neuron activations')
    plt.xlabel('Ranking of image patches')
    plt.subplots_adjust(hspace=-0.1)

    # another approach with the images on the curve ploted

    # fig, ax = plt.subplots()
    # ax.plot(activations)
    # for n, img_idx in enumerate(idx_images):
    #     img = images[img_idx]
    #     im = OffsetImage(img, zoom=1, interpolation='bicubic')
    #
    #     ab = AnnotationBbox(im, (img_idx, activations[img_idx]),
    #                         xycoords='data', frameon=False)
    #     ax.add_artist(ab)

    plt.show()

def plot_pixel_decomposition(activations, neurons, img, loc, rows=1):

    nf = []
    for n in neurons:
        nf.append(n.get_neuron_feature())

    n_images = len(nf)

    ri, rf, ci, cf = loc


    dr = ImageDraw.Draw(img)
    dr.rectangle([(ci,ri),(cf,rf)], outline='red')
    del dr

    fig = plt.figure()
    fig.add_subplot(rows+1, 1, 1)
    plt.imshow(img)
    plt.axis('off')

    for n in xrange(n_images):
        img = nf[n]
        c = np.ceil(n_images/float(rows))
        a = fig.add_subplot(rows+1, c, n + c +1)
        plt.imshow(img, interpolation='bicubic')
        plt.axis('off')
        t = round(activations[n], 2)
        a.set_title(t)

    # plt.yticks(np.arange(0, 1.1, 0.1))

    plt.show()

def plot_decomposition(activations, neurons, locations, img):
    nf = []
    offset = 1
    for n in neurons:
        nf.append(n.get_neuron_feature())

    w, h = nf[0].size
    print w, h

    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(img)
    plt.axis('off')

    for i in xrange(len(activations)-1, -1, -1):
        ri, rf, ci, cf = locations[i]
        print locations[i]
        if rf < h:
            ri = ri - (h - rf)
        if cf < w:
            ci = ci - (w - cf)
        if rf-ri < h:
            rf = ri + h
        if cf - ci < w:
            cf = ci + w

        print ri, rf, ci, cf
        # nf[i] = ImageOps.expand(nf[i], offset, fill='white')
        # img.paste(nf[i], (ri-offset, ci-offset, rf+offset, cf+offset))
        img.paste(nf[i], (ri, ci, rf, cf))

    fig.add_subplot(1,2,2)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def plot_similarity_tsne(layer_data):


    neurons = layer_data.get_filters()
    num_neurons = len(neurons)

    # x = np.random.rand(num_neurons, num_neurons)
    x = layer_data.similarity_index
    # np.savetxt('sim_l1.txt', x)
    # x = np.array([[0,0.2,1],[0.2,0,0.8],[0.6,0.1,0]])
    print x
    x_result = TSNE(n_components=2, metric='euclidean',
                    random_state=0).fit_transform(x)
    print x_result
    nf = [n.get_neuron_feature() for n in neurons]
    fig, ax = plt.subplots()

    for i, x, y in zip(range(num_neurons), x_result[:, 0], x_result[:, 1]):
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

    #ax.text(x+2, y+2, label) #TODO: put labels (number of the neuron)
    ax.add_artist(ab)
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


def plot_2d_index(selectivity_neurons):
    index_name = selectivity_neurons.keys()[0]
    layers_v = selectivity_neurons.values()[0]
    print index_name, layers_v

    for k, v in layers_v.items():
        layer_name = k
        neurons = v


def plot_nf_search(selective_neurons):
    index_name = selective_neurons.keys()
    if type(index_name[0]) is tuple:
        # this plot only works for 1 selectivity_index
        print 'Msg error: too many indexes to unpack'
        return

    layers_v = selective_neurons.values()[0]

    for k, v in layers_v.items():

        layer_name = k
        neurons = v

        neurons = sorted(neurons, key=lambda x: x[1])

        cols = int(math.sqrt(len(neurons)))
        n_images = len(neurons)

        titles = [round(n[1], 2) for n in neurons]

        fig = plt.figure()
        fig.suptitle('Layer: ' + layer_name + ', Index: ' + index_name[0])

        for i, (n, title) in enumerate(zip(neurons, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), i + 1,)
            plt.imshow(n[0].get_neuron_feature(), interpolation='bicubic')
            plt.axis('off')
            a.set_title(title)
        plt.show()
        fig.clear()


def plot_similarity_idx(neuron_data, sim_neuron, idx_values, rows=2):

    n_images = len(sim_neuron)

    # cols = int(math.sqrt(n_max))
    titles = [round(v, 2) for v in idx_values]

    fig = plt.figure()
    fig.suptitle('Similarity index')

    fig.add_subplot(rows+1, 1, 1)
    plt.imshow(neuron_data.get_neuron_feature(), interpolation='bicubic')
    plt.axis('off')

    for i, (n, title) in enumerate(zip(sim_neuron, titles)):
        c = np.ceil(n_images/float(rows))
        a = fig.add_subplot(rows+1, c, i + c+1)
        plt.imshow(n.get_neuron_feature(), interpolation='bicubic')
        plt.axis('off')
        a.set_title(title)
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    fig.clear()


def plot_neuron_features(layer_data):
    nf = []
    for f in layer_data.get_filters():
        nf.append(f.get_neuron_feature())
    n_images = len(nf)

    cols = int(math.sqrt(n_images))
    fig = plt.figure()
    for n, img in enumerate(zip(nf)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.imshow(nf[n], interpolation='bicubic')
        plt.axis('off')
        a.set_title(str(n))
    # fig.set_size_inches(n_max*3,n_max*3)
    plt.show()
    fig.clear()


def main():
    from keras.applications.vgg16 import VGG16
    from keras.preprocessing.image import load_img
    import pickle
    from image import ImageDataset
    from image import rotate_images


    dataset = '/home/oprades/ImageNet/train/'  # dataset path
    save_path = '/home/oprades/oscar/'
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']

    model = VGG16()

    my_net = pickle.load(open(save_path + 'vgg2_simIdx.obj', 'rb'))
    my_net.model = model
    img_dataset = ImageDataset(dataset, (224, 224))
    my_net.dataset = img_dataset
    my_net.save_path = save_path

    layer1 = my_net.get_layers()[0]
    layer3 = my_net.get_layers()[2]
    # plot_similarity_tsne(layer1)

    # plot_neuron_features(l1)


    # plot_top_scoring_images(my_net, l1, 85, n_max=100)

    # plot_activation_curve(my_net, layer3, 5)

    sel_idx = my_net.selectivity_idx_summary(['symmetry'], layer_names)

    # plot_sel_idx_summary(sel_idx)

    # plot_symmetry_distribution_summary(sel_idx)

    # decomposition
    # img_name = 'n03100240/n03100240_2539.JPEG'
    # act, neurons, loc = my_net.get_max_activations(3, img_name, (130, 190), 10)
    # img = load_img(dataset + img_name, target_size=(224, 224))
    #
    # plot_pixel_decomposition(act, neurons, img, loc)

    # # face example!
    # # act, n, loc, _ = my_net.decomposition(img_name, layer_names[3])
    #
    #
    # act, n, loc, nf = my_net.decomposition([layer_names[4], 67], layer_names[3])
    # print loc
    # plot_decomposition(act, n, loc, nf)




    selective_neurons = my_net.get_selective_neurons(
        layer_names[0:2], 'color', idx2='orientation')
    #
    print selective_neurons
    plot_2d_index(selective_neurons)

    # plot_nf_search(selective_neurons)
    #
    #
    # selective_neurons = my_net.get_selective_neurons(
    #     layer_names[0:2], 'orientation', inf_thr=0.5)
    # print selective_neurons
    # plot_nf_search(selective_neurons)



    # layer1 = my_net.get_layers()[0]
    # f = layer1.get_filters()[45]
    # neuron_data, idx_values = layer1.similar_neurons(45)
    # print len(neuron_data), len(idx_values)
    # plot_similarity_idx(f, neuron_data, idx_values)







if __name__ == '__main__':
    main()