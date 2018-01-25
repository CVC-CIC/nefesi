import numpy as np
import matplotlib.pyplot as plt
import math


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




def main():
    from keras.applications.vgg16 import VGG16
    import pickle
    from image import ImageDataset


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
    layer_names = ['block1_conv2', 'block2_conv2', 'block5_conv3']
    num_max_activations = 100

    model = VGG16()

    my_net = pickle.load(open(save_path + 'vgg16.obj', 'rb'))
    my_net.model = model
    img_dataset = ImageDataset(dataset, (224, 224))
    my_net.dataset = img_dataset
    my_net.save_path = save_path

    # plot_top_scoring_images(my_net, layer_names[2], 5, n_max=100)

    plot_activation_curve(my_net, layer_names[2], 5)


if __name__ == '__main__':
    main()