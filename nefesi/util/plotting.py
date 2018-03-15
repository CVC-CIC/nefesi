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
    img = img.copy()
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

def plot_decomposition(activations, neurons, locations, img, plot_nf_list=False):

    nf = []
    img = img.copy()
    for n in neurons:
        nf.append(n.get_neuron_feature())

    w, h = nf[0].size

    rows = 1
    if plot_nf_list:
        rows = 2

    fig = plt.figure()
    fig.add_subplot(rows,2,1)
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

        img.paste(nf[i], (ci, ri, cf, rf))

    fig.add_subplot(rows,2,2)
    plt.imshow(img)
    plt.axis('off')

    if plot_nf_list:
        num_images = len(nf)
        for i in xrange(num_images):
            fig.add_subplot(rows, num_images, i + num_images + 1)
            plt.imshow(nf[i])
            plt.axis('off')

    plt.show()

def test_neuron_sim(layer_data, n=None):
    from scipy import spatial
    # >> > airports = [(10, 10), (20, 20), (30, 30), (40, 40)]
    # >> > tree = spatial.KDTree(airports)
    # >> > tree.query([(21, 21)])

    neurons = layer_data.get_filters()

    idx_neurons = None
    if n is not None:
        idx_neurons = [neurons.index(i) for i in n]
        neurons = n

    num_neurons = len(neurons)

    # x = np.random.rand(num_neurons, num_neurons)
    x = layer_data.get_similarity_idx(neurons_idx=idx_neurons)
    # np.savetxt('sim_l1.txt', x)
    # x = np.array([[0,0.2,1],[0.2,0,0.8],[0.6,0.1,0]])
    x_result = TSNE(n_components=2, metric='euclidean',
                    random_state=0).fit_transform(x)


    x_max = np.max(x_result[:,0])
    x_min = np.min(x_result[:,0])
    y_max = np.max(x_result[:,1])
    y_min = np.min(x_result[:,1])

    # print x_max-abs(x_min), y_max -abs(y_min)
    #
    # print x_result
    # print np.sum(x_result[:, 0])/x_result[:,0].shape

    tree = spatial.KDTree(x_result)
    idx_center = tree.query([x_max-abs(x_min), y_max -abs(y_min)])[1]



    x_max = np.where(x_result[:, 0] == x_max)[0][0]
    x_min = np.where(x_result[:, 0] == x_min)[0][0]
    y_max = np.where(x_result[:, 1] == y_max)[0][0]
    y_min = np.where(x_result[:, 1] == y_min)[0][0]

    # print x_max, x_min, y_max, y_min

    res = []
    res.append(neurons[x_max])
    res.append(neurons[x_min])
    res.append(neurons[y_max])
    res.append(neurons[y_min])
    res.append(neurons[idx_center])



    # nf = [n.get_neuron_feature() for n in neurons]



    return res


def plot_similarity_tsne(layer_data, n=None):


    neurons = layer_data.get_filters()

    idx_neurons = None
    if n is not None:
        idx_neurons = [neurons.index(i) for i in n]
        neurons = n
        print idx_neurons

    num_neurons = len(neurons)

    # x = np.random.rand(num_neurons, num_neurons)
    x = layer_data.get_similarity_idx(neurons_idx=idx_neurons)
    # np.savetxt('sim_l1.txt', x)
    # x = np.array([[0,0.2,1],[0.2,0,0.8],[0.6,0.1,0]])
    print x
    x_result = TSNE(n_components=2, metric='euclidean',
                    random_state=0).fit_transform(x)
    print x_result
    nf = [n.get_neuron_feature() for n in neurons]
    fig, ax = plt.subplots()

    size_fig = fig.get_size_inches()
    nf_size = nf[0].size
    zoom = (size_fig[0] + size_fig[1]) / nf_size[0]

    for i, x, y in zip(range(num_neurons), x_result[:, 0], x_result[:, 1]):
        imscatter(x, y, nf[i], zoom=zoom, ax=ax, labels=str(i))
        ax.plot(x, y)
    plt.axis('off')
    plt.show()


def plot_similarity_circle(layer_data, target_neuron, bins=None):

    neurons = layer_data.get_filters()

    target_neuron_idx = neurons.index(target_neuron)

    fig, ax = plt.subplots()
    size_fig = fig.get_size_inches()
    nf_size = target_neuron.get_neuron_feature().size
    zoom = (size_fig[0] + size_fig[1]) / nf_size[0]

    fig_center = (0.5, 0.5)
    r = [0.15, 0.25, 0.5]
    imscatter(fig_center[0], fig_center[1], target_neuron.get_neuron_feature(), zoom=zoom, ax=ax)
    ax.plot(fig_center[0], fig_center[1])

    if bins is None:
        bins = [0.0, 0.4, 0.8, 1.0]

    for i in xrange(3):
        neuron_data, _ = layer_data.similar_neurons(
            target_neuron_idx, inf_thr=bins[-(i+2)], sup_thr=bins[-(i+1)])

        nf = [n.get_neuron_feature() for n in neuron_data if n is not target_neuron]
        num_neurons = len(nf)

        radius = r[i]
        circle = plt.Circle(fig_center, radius, fill=False)
        degrees = [j*(360/num_neurons) for j in xrange(num_neurons)]

        ax.add_artist(circle)

        # print 0.3 + 0.2*np.sin(20)
        # print 0.3 - 0.2*(1-np.cos(20))

        x1 = fig_center[0]
        y1 = fig_center[1] + radius
        x_coord = [x1 + radius * np.sin(d) for d in degrees]
        y_coord = [y1 - radius * (1 - np.cos(d)) for d in degrees]
        # for d in degrees:
        #     x2 = x1 + r*np.sin(d)
        #     y2 = y1 - r*(1-np.cos(d))
        #     x_coord.append(x2)
        #     y_coord.append(y2)

        for idx, x, y in zip(range(num_neurons), x_coord, y_coord):
            imscatter(x, y, nf[idx], zoom=zoom, ax=ax)
            ax.plot(x, y)

    ax.set_aspect('equal', adjustable='datalim')
    # ax.plot()  # Causes an autoscale update.
    plt.show()


def imscatter(x, y, image, ax=None, zoom=1, labels=None):
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
    if type(index_name) is tuple and len(index_name) == 2:

        layers_v = selectivity_neurons.values()[0]

        for k, v in layers_v.items():
            layer_name = k
            neurons = v

            x_values = [n[1] for n in neurons]
            y_values = [n[2] for n in neurons]
            nf = [n[0].get_neuron_feature() for n in neurons]

            fig, ax = plt.subplots()
            size_fig = fig.get_size_inches()
            nf_size = nf[0].size
            zoom = (size_fig[0]+size_fig[1])/nf_size[0]
            print nf_size
            for i, x, y in zip(range(len(nf)), x_values, y_values):
                imscatter(x, y, nf[i], zoom=zoom, ax=ax)
                ax.plot(x, y)
            plt.title('Layer: ' + layer_name)
            plt.xlabel(index_name[0])
            plt.ylabel(index_name[1])
            plt.show()


def plot_nf_search(selective_neurons, n_max=150):
    index_name = selective_neurons.keys()[0]
    layers_v = selective_neurons.values()[0]

    for k, v in layers_v.items():

        layer_name = k
        neurons = v
        # if len(neurons) > n_max:
        #     neurons = neurons[:n_max]

        neurons = sorted(neurons, key=lambda x: sum(x[1:]))
        num_neurons = len(neurons)
        if num_neurons > n_max:
            neurons = neurons[num_neurons - n_max:]

        cols = int(math.sqrt(len(neurons)))
        n_images = len(neurons)

        titles = [n[1:] for n in neurons]

        fig = plt.figure()
        fig.suptitle('Layer: ' + layer_name + ', Index: ' + str(index_name))

        for i, (n, title) in enumerate(zip(neurons, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), i + 1,)
            plt.imshow(n[0].get_neuron_feature(), interpolation='bicubic')
            plt.axis('off')
            tmp_t = ''
            for t in title:
                tmp_t += str(round(t, 2)) + ','
            a.set_title(tmp_t[:-1])
        # plt.tight_layout()
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


def plot_neuron_features(layer_data, neuron_list=None):
    nf = []
    if neuron_list is None:
        neuron_list = layer_data.get_filters()

    for f in neuron_list:
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
    from keras.applications.vgg16 import VGG16, preprocess_input
    from keras.preprocessing import image
    import pickle
    from image import ImageDataset
    from nefesi.network_data import NetworkData
    from nefesi.neuron_feature import compute_nf


    dataset = '/home/oprades/ImageNet/train/'  # dataset path
    save_path = '/home/oprades/oscar/'
    layer_names = ['block1_conv1', 'block1_conv2',
                   'block2_conv1', 'block2_conv2',
                   'block3_conv1', 'block3_conv2', 'block3_conv3']

    model = VGG16()

    my_net = pickle.load(open(save_path + 'block1_3.obj', 'rb'))

    img_dataset = ImageDataset(dataset, (224, 224), preprocess_input)
    my_net.model = model
    my_net.dataset = img_dataset
    my_net.save_path = save_path

    for l in my_net.get_layers():
        print l

    l = my_net.get_layers()[6]
    plot_neuron_features(l)
    compute_nf(my_net, l, l.get_filters())


    plot_neuron_features(l)
    # nf = l.get_filters()[131].get_neuron_feature()
    #
    # nf_cont = np.asarray(nf).astype(float)
    # minv = np.min(nf_cont.ravel())
    # maxv = np.max(nf_cont.ravel())
    # nf_cont = nf_cont - minv
    # nf_cont = nf_cont / (maxv - minv)
    #
    # nf_cont = image.array_to_img(nf_cont, scale=False)
    #
    # print type(nf_cont)
    #
    # plt.imshow(nf_cont, interpolation='bilinear')
    # plt.show()
    #
    # plt.imshow(nf, interpolation='bilinear')
    # plt.show()







    # layer1 = my_net.get_layers()[0]
    # layer2 = my_net.get_layers()[1]
    # layer3 = my_net.get_layers()[2]

    # print layer3.get_layer_id()
    # plot_neuron_features(layer3)
    #
    # new_layer1 = pickle.load(open(save_path+ 'block3_conv3.obj', 'rb'))
    # # print new_layer3.get_layer_id()
    # # plot_neuron_features(new_layer3)
    #
    #
    #
    # plot_neuron_features(new_layer1)
    #
    #
    # plot_top_scoring_images(my_net, new_layer1, 7, n_max=100)

    # f = layer1.get_filters()[61]
    # n, v = layer1.similar_neurons(61)
    # plot_similarity_idx(f, n, v)
    # plot_similarity_circle(layer1, f)



    # neuron_list = layer1.get_filters()[0:10]
    # plot_similarity_tsne(layer1, neuron_list)

    # first and second conv layer in first block
    # plot_neuron_features(layer1)
    # plot_neuron_features(layer2)

    # plot_top_scoring_images(my_net, l1, 85, n_max=100)

    # plot_activation_curve(my_net, layer3, 5)

    # sel_idx = my_net.selectivity_idx_summary(['symmetry'], layer_names)

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

    # plot_similarity_tsne(layer1)
    # plot_neuron_features(layer1)
    #
    # f = layer1.get_filters()[10]
    # neuron_data, idx_values = layer1.similar_neurons(10)
    # print len(neuron_data), len(idx_values)
    # plot_similarity_idx(f, neuron_data, idx_values, rows=3)


    # selective_neurons_gray = my_net.get_selective_neurons(
    #     layer_names[0], 'color', inf_thr=-1.0, sup_thr=0.2)
    # plot_nf_search(selective_neurons_gray)
    # # res = selective_neurons_gray[('color')][layer_names[0]]
    # # neurons = [n[0] for n in res]
    # # plot_similarity_tsne(layer1, n=neurons)
    #
    # sel_color = my_net.get_selective_neurons(
    #     layer_names[0], 'color', inf_thr=0.2
    # )
    # plot_nf_search(sel_color)
    # res = selective_neurons_gray[('color')][layer_names[0]]
    # neurons = [n[0] for n in res]
    #
    # print 111,  len(neurons)
    # plot_similarity_tsne(layer1, n=neurons)
    # print 222, len(neurons)
    # res = test_neuron_sim(layer1, n=neurons)
    # print 333, len(neurons)
    # plot_neuron_features(layer1, neuron_list=neurons)
    # print 444, len(neurons)
    # for r in res:
    #     neurons.remove(r)
    # #
    # # res2 = [[],[],[],[],[]]
    # # res2_v = [[],[],[],[],[]]
    # # for n in neurons:
    # #     sim = 0
    # #     idx_n = None
    # #     neu = None
    # #     for i in xrange(len(res)):
    # #         idx = layer1.get_filters().index(res[i])
    # #         idx2 = layer1.get_filters().index(n)
    # #         tmp_sim = layer1.similarity_index[idx, idx2]
    # #         if tmp_sim > sim:
    # #             sim = tmp_sim
    # #             neu = n
    # #             idx_n = i
    # #     res2[idx_n].append(neu)
    # #     res2_v[idx_n].append(sim)
    # #
    # #
    # # print res2
    # # print res2_v
    # #
    # # for i in xrange(len(res)):
    # #     res2_v[i] = np.asarray(res2_v[i])
    # #     order = np.argsort(res2_v[i])
    # #     order = order[::-1]
    # #     res2[i] = np.asarray(res2[i])
    # #     res2[i] = res2[i][order]
    # #     res2[i] = list(res2[i])
    # #     res2[i].insert(0, res[i])
    # #
    # #     print res2
    # #     plot_neuron_features(layer1, neuron_list=res2[i])
    #
    #
    # # idx = 20
    # # neuron_data, idx_values = layer1.similar_neurons(idx)
    # # print len(neuron_data), len(idx_values)
    # # plot_similarity_idx(layer1.get_filters()[idx], neuron_data, idx_values)
    #
    # print 555, len(neurons)
    # plot_neuron_features(layer1, neuron_list=neurons)
    # for r in res:
    #
    #     idx = layer1.get_filters().index(r)
    #
    #     # print idx
    #     n, v = layer1.similar_neurons(idx)
    #     # print len(n)
    #     tmp_n = []
    #     tmp_v = []
    #     for i in xrange(len(n)):
    #         if n[i] in neurons:
    #             tmp_n.append(n[i])
    #             tmp_v.append(v[i])
    #     print 666, len(tmp_n)
    #     plot_neuron_features(layer1, neuron_list=n)
    #     plot_similarity_idx(layer1.get_filters()[idx], tmp_n, tmp_v)


    #
    # gray_non_sim = my_net.get_selective_neurons(
    #     selective_neurons_gray, 'symmetry', sup_thr=0.75)
    #
    # gray_sim = my_net.get_selective_neurons(
    #     selective_neurons_gray, 'symmetry', inf_thr=0.75
    # )
    #
    # plot_nf_search(gray_non_sim)
    # plot_nf_search(gray_sim)
    #
    # color_non_sim = my_net.get_selective_neurons(
    #     sel_color, 'symmetry', sup_thr=0.75
    # )
    # color_sim = my_net.get_selective_neurons(
    #     sel_color, 'symmetry', inf_thr=0.75
    # )
    # plot_nf_search(color_non_sim)
    # plot_nf_search(color_sim)
    #
    # res = color_sim[('color','symmetry')][layer_names[0]]
    # neurons = [n[0] for n in res]
    #
    # plot_similarity_tsne(layer1, n=neurons)


    # print selective_neurons.values()[0].keys()
    # plot_2d_index(selective_neurons)
    # plot_neuron_features(layer1)
    #
    # plot_nf_search(selective_neurons)


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