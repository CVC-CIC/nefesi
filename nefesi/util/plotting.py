
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
from PIL import ImageDraw
from sklearn.manifold import TSNE
from matplotlib.widgets import RadioButtons,Button, Slider,TextBox
from scipy.interpolate import interp1d
from ..class_index import get_concept_labels
from .ColorNaming import colors as color_names
import networkx as nx
from ..interface.popup_windows.combobox_popup_window import ComboboxPopupWindow
from ..interface.popup_windows.special_value_popup_window import SpecialValuePopupWindow

LIST_OF_BUTTONS_TO_NO_DETACH = []
NODES_CROPPED_IN_SUMMARY = 2

def plot_with_cumsum(x, y, leafs, first_level_id, level):
    y = y*100
    colors = {0: 'slategray', 1:'mediumslateblue', 2:'deepskyblue',
              3:'green', 4: 'olive', 5: 'darkorange', 6:'gold',
              7: 'coral'}
    color_list = np.zeros(shape=len(first_level_id), dtype=np.object)
    for i in range(len(first_level_id)):
        color_list[i] = colors[first_level_id[i]]
    #idx_sort = y.argsort()[::-1]
    #y = y[idx_sort]
    #x = x[idx_sort]
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='slategray', lw=4),
                    Line2D([0], [0], color='mediumslateblue', lw=4),
                    Line2D([0], [0], color='deepskyblue', lw=4),
                    Line2D([0], [0], color='green', lw=4),
                    Line2D([0], [0], color='olive', lw=4),
                    Line2D([0], [0], color='darkorange', lw=4),
                    Line2D([0], [0], color='gold', lw=4),
                    Line2D([0], [0], color='coral', lw=4)]
    plt.legend(custom_lines, ['Artifact, Artefact', 'Animal, Fauna', 'Natural Object', 'Geological Form.', 'Misc', 'Fungus', 'Plant, Flora', 'Person, Individual'], prop={'size':8})
    plt.xticks(rotation=90,fontsize=10)
    plt.ylabel("% of images at this node")
    plt.xlabel("WordNet node name")
    plt.title("Level "+str(level)+" frequency [Total of images at this level -> "+str(np.round(np.sum(y),2))+"%]")
    #x[33] = 'dish artifact'
    #x[139] = 'geological bar'
    plt.margins(x=0.005)
    barlist = plt.bar(x, y, color = color_list)
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.show()

def plot_sel_idx_summary(selectivity_idx, bins=10, color_map='jet'):
    """Plots a summary over the selectivity indexes from a group of
    specific layers.
    If selectivity index is "orientation" or "symmetry", plots the
    global index.

    :param selectivity_idx: The output from the function
        `nefesi.network_data.NetworkData.get_selectivty_idx()`.
    :param bins: Integer, number of bins for bars.
    :param color_map: String, name of one color maps accepted by Matplotlib.

    :return:
    """
    cmap = plt.cm.get_cmap(color_map)
    colors = []
    for i in range(bins):
        colors.append(cmap(1.*i/bins))

    for k, v in selectivity_idx.items():

        N = len(v)
        pos = 0
        for l in v:
            if k == 'symmetry' or k == 'orientation' or k == 'class':
                n_idx = len(l[0])
                l = [idx[n_idx - 1] for idx in l]

            for i in range(len(l)):
                if l[i] > 1.0:
                    l[i] = 1.0
                if l[i] < 0.0:
                    l[i] = 0.0

            counts, bins = np.histogram(l, bins=bins, range=(0, 1))
            print(counts, bins)
            num_f = sum(counts)
            prc = np.zeros(len(counts))

            for i in range(len(counts)):
                prc[i] = float(counts[i])/num_f*100.
            y_offset = 0

            bars = []
            for i in range(len(prc)):
                p = plt.bar(pos, prc[i], bottom=y_offset, width=0.35, color=colors[i])
                bars.append(p)
                y_offset = y_offset+prc[i]
            pos += 1

        xticks = []
        for i in range(N):
            xticks.append('Layer ' + str(i + 1))
        plt.xticks(np.arange(N), xticks)
        plt.yticks(np.arange(0, 101, 10))

        labels = [str(bins[i]) + ':' + str(bins[i+1]) for i in range(len(prc))]

        plt.ylabel('% of Neurons')
        plt.title(k + ' selectivity')
        plt.legend(bars, labels, bbox_to_anchor=(1.02, 1.02), loc=2)
        plt.subplots_adjust(right=0.75)
        plt.show()


def plot_symmetry_distribution_summary(selectivity_idx, color_map='jet'):
    """Plots the distribution of index symmetry values among
    the distinct axes of symmetry.

    :param selectivity_idx: The output from the function
        `nefesi.network_data.NetworkData.get_selectivty_idx()`.
    :param color_map: String, name of one color maps accepted by Matplotlib.

    :return:

    :raise:
        ValueError: If index is not "symmetry".
    """
    bins = 4
    cmap = plt.cm.get_cmap(color_map)
    colors = []
    for i in range(bins):
        colors.append(cmap(1. * i / bins))

    for k, v in selectivity_idx.items():
        if k != 'symmetry':
            raise ValueError("This plot only works with symmetry index.")

        N = len(v)
        pos = 0
        for l in v:
            counts = [0, 0, 0, 0]

            for f in l:
                max_idx = max(f)
                counts[f.index(max_idx)] += 1
            num_f = sum(counts)
            prc = np.zeros(len(counts))

            for i in range(len(counts)):
                prc[i] = float(counts[i]) / num_f * 100.
            y_offset = 0

            bars = []
            for i in range(len(prc)):
                p = plt.bar(pos, prc[i], bottom=y_offset, width=0.35, color=colors[i])
                bars.append(p)
                y_offset = y_offset + prc[i]
            pos += 1

        xticks = []
        for i in range(N):
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
    """Plots the TOP scoring images (receptive fields) from a neuron
    with index `neuron_idx`.

    :param network_data: The `nefesi.network_data.NetworkData` instance.
    :param layer_data:  The `nefesi.layer_data.LayerData` instance.
    :param neuron_idx: Integer, index of the neuron.
    :param n_max: Integer, number of images to show.

    :return:

    :raise:
        ValueError: If `layer_data` doesn't match with any layer
        inside `network_data.layers`.
    """
    layers = network_data.layers_data
    layer_name = None
    neuron = None
    for l in layers:
        if l.layer_id == layer_data.layer_id:
            neuron = l.neurons_data[neuron_idx]
            layer_name = l.layer_id

    if layer_name is None:
        raise ValueError("No layer {} in the model.".format(layer_data.layer_id))

    images = neuron.get_patches(network_data, layer_data, as_numpy=False)
    activations = neuron.norm_activations
    images = images[:n_max]
    activations = activations[:n_max]

    color_map = None
    if images[0].mode == 'L':
        color_map = 'gray'

    sumim = np.array(images[0])*0
    for i in images:
        sumim = sumim +np.array(i)
    sumim = sumim/len(images)
    fig = plt.figure()
    plt.imshow(sumim/255.0, cmap='gray')

    cols = int(math.sqrt(len(images)))
    n_images = len(images)
    titles = [round(act, 2) for act in activations]
    fig = plt.figure()
    fig.suptitle('Layer: ' + layer_name + ', Filter index: ' + str(neuron_idx))
    for n, (img, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.imshow(img, interpolation='bicubic', cmap=color_map)
        plt.axis('off')
        a.set_title(title)
    # fig.set_size_inches(n_max*3,n_max*3)
    plt.show()
    #fig.clear()


def plot_activation_curve(network_data, layer_data, neuron_idx, num_images=5):
    """Plots the curve of the activations values from the neuron
    with index `neuron_idx`. Also plots 1 of each `num_images` above
    the graph.

    :param network_data: The `nefesi.network_data.NetworkData` instance.
    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    :param neuron_idx: Integer, index of the neuron.
    :param num_images:Integer, number of TOP scoring images plotted.

    :return:
    """
    neuron = layer_data.neurons_data[neuron_idx]
    images = neuron.get_patches(network_data, layer_data)
    activations = neuron.norm_activations
    color_map = None
    if Image.fromarray(images[0].astype('uint8'), 'RGB').mode == 'L':
        color_map = 'gray'

    idx_images = np.arange(0, len(images), num_images)
    cols = len(idx_images)

    fig = plt.figure()
    for n, img_idx in enumerate(idx_images):
        img = Image.fromarray(images[img_idx].astype('uint8'), 'RGB')
        t = round(activations[img_idx], 2)
        a = fig.add_subplot(2, cols, n + 1)
        plt.imshow(img, interpolation='bicubic', cmap=color_map)
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


def plot_nf_of_entities_in_pc(network_data, master = None, layer_selected = '.*', entity='class'):
    axcolor = 'lightgoldenrodyellow'
    rax = plt.axes([0.005, 0.2, 0.15, 0.55], facecolor=axcolor)
    rax2 = plt.axes([0.16, 0.5, 0.8, 0.05], facecolor=axcolor)
    layers_to_analyze = network_data.get_layer_names_to_analyze()
    try:
        layer_selected = layers_to_analyze.index(layer_selected)
    except:
        layer_selected = 0
    radio = RadioButtons(rax, layers_to_analyze, active=layer_selected)
    if entity == 'class':
        labels = list(np.sort(np.array(list(network_data.default_labels_dict.values()))))
    elif entity == 'object':
        labels = list(np.sort(get_concept_labels(entity)))
    elif entity == 'color':
        labels = list(np.sort(np.array(color_names)))

    def updateslide(val):
        plt.suptitle(labels[int(val)],y=0.7,x=0.5)



    slid= Slider(rax2,'',0, len(labels)-1,valinit=int(len(labels)/2),valfmt='%d')
    slid.on_changed(updateslide)

    def _yes(event):
        entity_name=labels[int(slid.val)]
        plot_pc_of_class(network_data,radio.value_selected,entity_name, master=master, entity=entity)

    def submit(text):
        plt.suptitle(text, y=0.7, x=0.5)
        if text in labels:
            slid.set_val(labels.index(text))
        else:
            opcions=[x for x in labels if x.startswith(text)]
            if len(opcions)>10:
                opcions=opcions[:9]
                opcions.append('...')

            if len(opcions)>1:
                opcions=' ,'.join(opcions)

            plt.suptitle(opcions,y=0.7,x=0.5)

    axbox = plt.axes([0.3, 0.4, 0.45, 0.075])
    text_box = TextBox(axbox, 'Label', initial='')
    text_box.on_submit(submit)

    axcut = plt.axes([0.45, 0.05, 0.1, 0.075])
    bcut = Button(axcut, 'Go', color='red', hovercolor='green')
    bcut.on_clicked(_yes)
    plt.show()


def plot_entity_representation(network_data, layers, entity='class', interface=None, th=0,operation='1/PC'):
    if type(layers) is not list:
        layers = network_data.get_layers_analyzed_that_match_regEx(layers)

    entity_representation, xlabels = network_data.get_entinty_representation_vector(layers=layers,entity=entity,
                                                                                    operation=operation)
    total_entity_representation = np.sum(entity_representation, axis=0)

    if interface is not None:
        entity_non_zero_mask = total_entity_representation > 0.000001
        if entity_non_zero_mask.max():
            entity_non_zero_vector = total_entity_representation[entity_non_zero_mask]
            maxim = round(entity_non_zero_vector.max(), 2)
            non_zero_mean = round(np.mean(entity_non_zero_vector), 2)
            minim = round(entity_non_zero_vector.min(), 2)
            values, counts = np.unique(entity_non_zero_vector, return_counts=True)
            mode = round(values[np.argmax(counts)], 2)
            std = round(np.std(entity_non_zero_vector), 2)
            text = "Set the threshold for consider a significant " + entity + " .\n " \
                 "min = " + str(minim) + " max = " + str(maxim) + "\n" \
                " mode = " + str(mode) + "mean = " + str(non_zero_mean) + " std = " + str(std)
            start = round(min(non_zero_mean + std, maxim - minim), 2)
            th = interface.get_value_from_popup(index='entity', text=text, max=maxim, start=start)
            if th == -1:
                return

    entity_mask = total_entity_representation > th
    total_entity_representation = total_entity_representation[entity_mask]
    xlabels = xlabels[entity_mask]

    args_sorted = total_entity_representation.argsort()[::-1]

    entity_representation = entity_representation[:, entity_mask]
    entity_representation, xlabels = entity_representation[:,args_sorted], xlabels[args_sorted]
    bars = []
    color_map = plt.cm.get_cmap('autumn')
    if len(layers) == 1:
        color_map = [color_map(0.0)]
    else:
        color_map = [color_map(i/(len(layers)-1)) for i in range(len(layers))][::-1]
    for layer in range(len(layers)):
        floor_of_bar = np.sum(entity_representation[:layer,...], axis=0)
        bars.append(plt.bar(xlabels, entity_representation[layer],
                              bottom=floor_of_bar, color=color_map[layer])[0])

    plt.xticks(rotation=90, fontsize=10)
    plt.ylabel("Total Representation ")
    plt.margins(x=0.005)
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.legend(bars[::-1], layers[::-1])

    non_represented_entities = len(entity_non_zero_mask)-np.count_nonzero(entity_non_zero_mask)
    represented_below_th_entities = len(entity_mask)-(np.count_nonzero(entity_mask)+non_represented_entities)
    plt.title(entity.capitalize() +" Representation (Total "+entity+": "
              +str(len(entity_mask))+") [th="+str(round(th,2))+"]  \n"
                "Non-represented "+entity+": "+str(non_represented_entities)+" - Represented "+entity
              +" below th: "+ str(represented_below_th_entities))

    plt.show()

def neurons_by_object_vs_ocurrences_in_imagenet(network_data, layers='.*', entity='class', interface=None, th=0,operation='1/PC'):
    if type(layers) is not list:
        layers = network_data.get_layers_analyzed_that_match_regEx(layers)

    entity_representation, xlabels = network_data.get_entinty_representation_vector(layers=layers, entity=entity,
                                                                                    operation=operation)
    total_entity_representation = np.sum(entity_representation, axis=0)

    if interface is not None:
        entity_non_zero_mask = total_entity_representation > 0.000001
        if entity_non_zero_mask.max():
            entity_non_zero_vector = total_entity_representation[entity_non_zero_mask]
            maxim = round(entity_non_zero_vector.max(), 2)
            non_zero_mean = round(np.mean(entity_non_zero_vector), 2)
            minim = round(entity_non_zero_vector.min(), 2)
            values, counts = np.unique(entity_non_zero_vector, return_counts=True)
            mode = round(values[np.argmax(counts)], 2)
            std = round(np.std(entity_non_zero_vector), 2)
            text = "Set the threshold for consider a significant " + entity + " .\n " \
                                                                              "min = " + str(minim) + " max = " + str(
                maxim) + "\n" \
                         " mode = " + str(mode) + "mean = " + str(non_zero_mean) + " std = " + str(std)
            start = round(min(non_zero_mean + std, maxim - minim), 2)
            th = interface.get_value_from_popup(index='entity', text=text, max=maxim, start=start)
            if th == -1:
                return

    entity_mask = total_entity_representation > th
    total_entity_representation = total_entity_representation[entity_mask]
    xlabels = xlabels[entity_mask]

    args_sorted = total_entity_representation.argsort()[::-1]
    object_ocurrence_vector = np.load('ObjectOcurrenceVector.npy')
    object_ocurrence_vector = object_ocurrence_vector[entity_mask]
    object_ocurrence_vector = object_ocurrence_vector[args_sorted]
    object_ocurrence_vector = object_ocurrence_vector.astype(np.float)/(np.max(object_ocurrence_vector.astype(np.float))/np.max(total_entity_representation))
    entity_representation = entity_representation[:, entity_mask]
    entity_representation, xlabels = entity_representation[:, args_sorted], xlabels[args_sorted]
    bars = []
    color_map = plt.cm.get_cmap('autumn')
    if len(layers) == 1:
        color_map = [color_map(0.0)]
    else:
        color_map = [color_map(i / (len(layers) - 1)) for i in range(len(layers))][::-1]
    for layer in range(len(layers)):
        floor_of_bar = np.sum(entity_representation[:layer, ...], axis=0)
        bars.append(plt.bar(xlabels, entity_representation[layer],
                            bottom=floor_of_bar, color=color_map[layer])[0])
    plt.plot(xlabels, object_ocurrence_vector, '-')

    plt.xticks(rotation=90, fontsize=10)
    plt.ylabel("Total Representation ")
    plt.margins(x=0.005)
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.legend(bars[::-1], layers[::-1])

    non_represented_entities = 0#len(entity_non_zero_mask) - np.count_nonzero(entity_non_zero_mask)
    represented_below_th_entities = len(entity_mask) - (np.count_nonzero(entity_mask) + non_represented_entities)
    plt.title(entity.capitalize() + " Representation (Total " + entity + ": "
              + str(len(entity_mask)) + ") [th=" + str(round(th, 2)) + "]  \n"
                                                                       "Non-represented " + entity + ": " + str(
        non_represented_entities) + " - Represented " + entity
              + " below th: " + str(represented_below_th_entities))

    plt.show()

def plot_coocurrence_graph(network_data, layers=None, entity='class', interface=None, th_low=0, max_degree=None,
                           operation='1/PC', th_superior = None):
    class_matrix, labels = network_data.get_entinty_co_ocurrence_matrix(layers=layers,entity=entity,operation=operation)
    #Axis 0 = Layers
    class_matrix = np.sum(class_matrix, axis=0)
    # Make 0's the diagonal for make the matrix a graph adyacecy matrix
    diag = class_matrix[range(class_matrix.shape[0]), range(class_matrix.shape[1])]
    class_matrix[range(class_matrix.shape[0]), range(class_matrix.shape[1])] = 0
    if interface is not None:
        non_zero_matrix = class_matrix[class_matrix > 0.0001]
        if len(non_zero_matrix) != 0:
            maxim = round(non_zero_matrix.max(),2)
            non_zero_mean = round(np.mean(non_zero_matrix),2)
            minim = round(non_zero_matrix.min(),2)
            values,counts = np.unique(non_zero_matrix,return_counts=True)
            mode = round(values[np.argmax(counts)],2)
            std = round(np.std(non_zero_matrix), 2)
            text = "Set the low threshold for consider a significant "+ entity + " relation.\n " \
                    "min = "+str(minim)+" max = "+str(maxim)+"\n" \
                    "mean = "+str(non_zero_mean)+ " mode = "+str(mode) + " std = "+str(std)
            start = round(min(non_zero_mean+std, maxim-minim),2)
            th_low = interface.get_value_from_popup(index='entity', text=text, max=maxim, start=start)
            if th_low == -1:
                return
            text = "Set the superior threshold for show " + entity + " relation.\n " \
                                                                                  "min = " + str(
                minim) + " max = " + str(maxim) + "\n" \
                                                  "mean = " + str(non_zero_mean) + " mode = " + str(
                mode) + " std = " + str(std)
            th_superior = interface.get_value_from_popup(index='entity', text=text, max=maxim, start=maxim)
            if th_superior == -1:
                return
    # strange dict with keys equals to his index. Sames that is the one that needs 'relabel_nodes'
    label_names = {key: value for key, value in enumerate(labels)}
    class_matrix[class_matrix < 0.0001] = 0

    entitys_without_relations = np.count_nonzero(np.max(class_matrix,axis=0) < 0.001)
    class_matrix[class_matrix < th_low] = 0
    if th_superior is not None:
        class_matrix[class_matrix > th_superior] = 0

    entitys_with_relations_below_th = np.count_nonzero(np.max(class_matrix, axis=0) < 0.001)-entitys_without_relations

    G = nx.DiGraph(class_matrix)
    G = nx.relabel_nodes(G, label_names)
    G.remove_nodes_from(list(nx.isolates(G)))

    outdeg = np.array(G.out_degree(), dtype=[('name', np.object),('degree',np.int)])

    #The user select the max degree for clean
    if interface is not None:
        degrees = outdeg['degree']
        if len(degrees) != 0:
            maxim = int(np.max(degrees))
            mean = round(np.mean(degrees),2)
            minim = degrees.min()
            values,counts = np.unique(degrees,return_counts=True)
            mode = values[np.argmax(counts)]
            std = round(np.std(degrees), 2)
            text = "Set the max degree of node for be represented.\n " \
                    "min = "+str(minim)+" max = "+str(maxim)+"\n" \
                    "mean = "+str(mean)+ " mode = "+str(mode) + " std = "+str(std)+".\n" \
                    "[Value included (max for show all nodes)]"
            max_degree = interface.get_value_from_popup(index='entity', text=text, max=maxim, start=maxim)
            if max_degree == -1:
                return
    elif max_degree is None:
        max_degree = np.max(outdeg['degree'])

    #Remove the nodes with degree over the threshold
    G.remove_nodes_from(outdeg[outdeg['degree']>max_degree]['name'])

    #graphs = list(nx.strongly_connected_component_subgraphs(G))
    #label_names = list(graphs[0])
    # import xml.etree.ElementTree
    # tree = xml.etree.ElementTree.parse('/home/guillem/Nefesi/nefesi_old/nefesi/imagenet_structure.xml').getroot()

    # gf.get_hierarchy_of_label(labels=label_names, freqs=freqs, xml='/home/guillem/Nefesi/nefesi_old/nefesi/imagenet_structure.xml',population_code=len(label_names))


    # ---------- Plot the graph ---------------
    nodes_in_order = list(G.degree._nodes.keys())

    #set the list of node color
    labels = list(labels)
    node_representation = [float(diag[labels.index(node)]) for node in nodes_in_order]

    # An ugly fake for plot the bar
    nodes = nx.draw_networkx_nodes(G, nx.spring_layout(G), with_labels=True, node_color=node_representation,
                                   cmap=plt.cm.cool)
    plt.close()

    #set the list of edge weight
    edges_weight = np.array(list(nx.get_edge_attributes(G, 'weight').values()))
    interpolator = interp1d ([th_low, edges_weight.max()], [1., 4.])
    edges_weight = interpolator(edges_weight)


    title = entity.capitalize() + ' correlation in Network [th_low=' + str(round(th_low, 2)) + "] \n " + \
            entity.capitalize() +" without relations: " + str(entitys_without_relations) +" - " + \
            entity.capitalize() +" with all relations below th_low: " + str(entitys_with_relations_below_th)
    #append a little summary of the cropped nodes
    if max_degree != np.max(outdeg['degree']):
        nodes_cropped = np.count_nonzero(outdeg['degree']>max_degree)
        title += "\n Nodes with deg>"+str(int(max_degree))+" (not plotted): "+str(nodes_cropped)
        if nodes_cropped != 0:
            cropped_nodes = outdeg[outdeg['degree'] > max_degree]
            cropped_nodes = np.sort(cropped_nodes,order = 'degree')[::-1]
            title+= " ["
            for i, (node, degree) in enumerate(cropped_nodes):
                title+=node+"("+str(degree)+")"
                if i>=NODES_CROPPED_IN_SUMMARY-1:
                    title+="..."
                    break
                else:
                    title+=", "
            else:
                title = title[:-len(', ')] #erase the last ", "
            title+="]"
    #plot
    plt.title(title)
    cbr = plt.colorbar(nodes, pad=0.04)
    cbr.ax.get_yaxis().labelpad = 15
    cbr.ax.set_ylabel('Neurons with PC = 1', rotation=270)
    nx.draw(G, with_labels=True, node_color=node_representation,
            cmap=plt.cm.cool, alpha=0.95, width=edges_weight)
    plt.show()


def plot_similarity_graph(network_data, layer, interface=None, th=0, max_degree=None, entity = 'class'):
    similiarity_matrix = network_data.similarity_idx(layer)[0]
    #Axis 0 = Layers
    # Make 0's the diagonal for make the matrix a graph adyacecy matrix
    similiarity_matrix[range(similiarity_matrix.shape[0]), range(similiarity_matrix.shape[1])] = 0
    if interface is not None:
        non_zero_matrix = similiarity_matrix[similiarity_matrix > 0.0001]
        if len(non_zero_matrix) != 0:
            maxim = round(non_zero_matrix.max(),2)
            non_zero_mean = round(np.mean(non_zero_matrix),2)
            minim = round(non_zero_matrix.min(),2)
            values,counts = np.unique(non_zero_matrix,return_counts=True)
            mode = round(values[np.argmax(counts)],2)
            std = round(np.std(non_zero_matrix), 2)
            text = "Set the threshold for consider a significant similarity relation.\n " \
                    "min = "+str(minim)+" max = "+str(maxim)+"\n" \
                    "mean = "+str(non_zero_mean)+ " mode = "+str(mode) + " std = "+str(std)
            start = round(min(non_zero_mean+std, maxim-minim),2)
            th = interface.get_value_from_popup(index='entity', text=text, max=maxim, start=start)
            if th == -1:
                return


    # strange dict with keys equals to his index. Sames that is the one that needs 'relabel_nodes'
    layer_data = network_data.get_layer_by_name(layer[0])
    label_names = {}
    for i in range(len(similiarity_matrix)):
        neuron = layer_data.neurons_data[i]
        if entity == 'class':
            ents = neuron.class_selectivity_idx(labels=network_data.default_labels_dict)
        elif entity == 'object':
            ents = neuron.concept_selectivity_idx(layer_data=layer_data,
                                                  network_data=network_data, neuron_idx=i, concept=entity)
        elif entity == 'color':
            ents = neuron.color_selectivity_idx(network_data=network_data, layer_name=layer_data.layer_id, neuron_idx=i)
        name = str(i)
        if ents[0]['label'] != 'None':
            name += ' ('+ents[0]['label'] +' - PC:'+str(len(ents))+')'
        label_names[i] = name
    similiarity_matrix[similiarity_matrix < 0.0001] = 0
    entitys_without_relations = np.count_nonzero(np.max(similiarity_matrix,axis=0) < 0.001)
    similiarity_matrix[similiarity_matrix < th] = 0
    entitys_with_relations_below_th = np.count_nonzero(np.max(similiarity_matrix, axis=0) < 0.001)-entitys_without_relations

    G = nx.DiGraph(similiarity_matrix)
    G = nx.relabel_nodes(G, label_names)
    G.remove_nodes_from(list(nx.isolates(G)))

    outdeg = np.array(G.out_degree(), dtype=[('name', np.object),('degree',np.int)])

    #The user select the max degree for clean
    if interface is not None:
        degrees = outdeg['degree']
        if len(degrees) != 0:
            maxim = int(np.max(degrees))
            mean = round(np.mean(degrees),2)
            minim = degrees.min()
            values,counts = np.unique(degrees,return_counts=True)
            mode = values[np.argmax(counts)]
            std = round(np.std(degrees), 2)
            text = "Set the max degree of node for be represented.\n " \
                    "min = "+str(minim)+" max = "+str(maxim)+"\n" \
                    "mean = "+str(mean)+ " mode = "+str(mode) + " std = "+str(std)+".\n" \
                    "[Value included (max for show all nodes)]"
            max_degree = interface.get_value_from_popup(index='entity', text=text, max=maxim, start=maxim)
            if max_degree == -1:
                return
    elif max_degree is None:
        max_degree = np.max(outdeg['degree'])

    #Remove the nodes with degree over the threshold
    G.remove_nodes_from(outdeg[outdeg['degree']>max_degree]['name'])

    #graphs = list(nx.strongly_connected_component_subgraphs(G))
    #label_names = list(graphs[0])
    # import xml.etree.ElementTree
    # tree = xml.etree.ElementTree.parse('/home/guillem/Nefesi/nefesi_old/nefesi/imagenet_structure.xml').getroot()

    # gf.get_hierarchy_of_label(labels=label_names, freqs=freqs, xml='/home/guillem/Nefesi/nefesi_old/nefesi/imagenet_structure.xml',population_code=len(label_names))


    # ---------- Plot the graph ---------------
    nodes_in_order = list(G.degree._nodes.keys())

    #set the list of edge weight
    edges_weight = np.array(list(nx.get_edge_attributes(G, 'weight').values()))
    interpolator = interp1d ([th,edges_weight.max()], [1.,4.])
    edges_weight = interpolator(edges_weight)


    title = 'Symilarity in Network [th='+str(round(th,2))+"] \n "+\
            "Neurons without relations: "+str(entitys_without_relations)+" - "+\
            "Neurons with all relations below th: "+str(entitys_with_relations_below_th)
    #append a little summary of the cropped nodes
    if max_degree != np.max(outdeg['degree']):
        nodes_cropped = np.count_nonzero(outdeg['degree']>max_degree)
        title += "\n Nodes with deg>"+str(int(max_degree))+" (not plotted): "+str(nodes_cropped)
        if nodes_cropped != 0:
            cropped_nodes = outdeg[outdeg['degree'] > max_degree]
            cropped_nodes = np.sort(cropped_nodes,order = 'degree')[::-1]
            title+= " ["
            for i, (node, degree) in enumerate(cropped_nodes):
                title+=node+"("+str(degree)+")"
                if i>=NODES_CROPPED_IN_SUMMARY-1:
                    title+="..."
                    break
                else:
                    title+=", "
            else:
                title = title[:-len(', ')] #erase the last ", "
            title+="]"
    #plot
    plt.title(title)
    nx.draw(G, with_labels=True,
            cmap=plt.cm.cool, alpha=0.95, width=edges_weight)
    plt.show()

def plot_pc_of_class(network_data, layer_name, entity_name, master = None, entity='class'):
    plt.figure()
    #given a nefesimodel, a layer, and a dictionary of the population codes for a given class, plots the different neuron_features were the class appears
    # first we create a dictionary with all the classes above a threshold (here 0.1)
    entity_info = {}
    #Detect all neurons with this entity_name in his population code (and sabes idx, entities and value of entity_name)
    for layer in network_data.layers_data:
        layer_entity = []
        for i, neuron in enumerate(layer.neurons_data):
            if entity == 'class':
                ents = neuron.class_selectivity_idx(labels = network_data.default_labels_dict)
            elif entity == 'object':
                ents = neuron.concept_selectivity_idx(layer_data=layer_name,
                                                      network_data=network_data, neuron_idx=i, concept=entity)
            elif entity == 'color':
                ents = neuron.color_selectivity_idx(network_data=network_data, layer_name=layer_name, neuron_idx=i)

            if entity_name in ents['label']:
                value = ents[ents['label'] == entity_name]['value'][0]
                layer_entity.append((i, tuple(ents['label']),value))

        entity_info[layer.layer_id] = layer_entity



# then we create the dictionary with the layers and pc where class_name appears
    pc_dict = {}
    for layername in entity_info.keys():
        pc_dict[layername] = {}
        for i, neuron in enumerate(entity_info[layername]):
            pc_number = len(neuron[1])
            if pc_number not in pc_dict[layername]:
                pc_dict[layername][pc_number] = [neuron]
            else:
                pc_dict[layername][pc_number].append(neuron)
        #transform the layer info from list to an structured array, for make easier the sort by value
        if layername == layer_name and len(list(pc_dict[layername].keys())) > 0:
            for pc_number in pc_dict[layername].keys():
                pc_dict[layername][pc_number] = np.array(pc_dict[layername][pc_number], dtype=[('idx', np.int),
                                                                                               ('labels', np.object),
                                                                                               ('value', np.float)])





#finally we plot the result (neuron features of neurons of layer_name, activated by class_name)

    nf_size = network_data.get_neuron_of_layer(layer_name, 1).neuron_feature.size[0]

    pcs_of_layer = list(pc_dict[layer_name].keys())
    pcs_of_layer.sort()
    image_axes = np.zeros(len(pcs_of_layer), np.object)
    for k, j in enumerate(pcs_of_layer):
        if len(pc_dict[layer_name][j])>1:
            x = network_data.get_layer_by_name(layer_name).get_similarity_idx(model = network_data.model,
                                                                        neurons_idx=pc_dict[layer_name][j]['idx'],
                                                                              dataset=network_data.dataset)
            x_result = TSNE(n_components=1, metric='euclidean',
                            random_state=0).fit_transform(x)
            pc_dict[layer_name][j] = pc_dict[layer_name][j][np.argsort(x_result[::-1,0])]

        neurons_num=len(pc_dict[layer_name][j])

        neuronfeature= Image.new('RGB',(nf_size*neurons_num,nf_size))
        for i in range(neurons_num):
            neuron_num= pc_dict[layer_name][j][i][0]
            neuronfeature.paste(network_data.get_neuron_of_layer(layer_name, neuron_num).neuron_feature,(i*nf_size,0))

            sub_axis = plt.subplot(len(pcs_of_layer),1,k+1)

            image_axes[k] = (pc_dict[layer_name][j][i][0], sub_axis, pc_dict[layer_name][j])
            plt.imshow(neuronfeature,shape=(200,200))

            plt.subplots_adjust(hspace=.001)

            plt.xticks([])
            plt.yticks([])


            label=''
            for name in pc_dict[layer_name][j][i][1]:
                label+=str(name).split(',')[0]+'\n'

            font_size = int(interp1d([1,3],[12,6], bounds_error=False, fill_value=6)(len(pcs_of_layer)))
            plt.text(0.01+i/neurons_num, -0.1, label, {'fontsize': font_size},
                     horizontalalignment='left',
                     verticalalignment='top',
                     rotation=0,
                     clip_on=False,
                     transform=plt.gca().transAxes)

            plt.text(0.01 + i / neurons_num, 0.9,pc_dict[layer_name][j][i][0], {'fontsize': font_size},
                     horizontalalignment='left',
                     verticalalignment='top',
                     rotation=0,
                     clip_on=False,
                     transform=plt.gca().transAxes)

            plt.ylabel('PC = %d' %(j))
            plt.gcf().subplots_adjust(bottom=0.3)


    plt.gcf().canvas.mpl_connect('button_press_event', lambda event: _on_click_image(event,master,network_data,
                                                                                     layer_name, image_axes, entity))
    plt.suptitle(layer_name, y=0.96)

    layers_neurons = []

    for lay, content in pc_dict.items():
        if not len(content) == 0:
            quantity = 0
            for v in content.values():
                quantity += len(v)
            layers_neurons.append((lay,quantity))


    # for centering
    r = len(layers_neurons)/4
    lines = int(r)
    excedent = int((r-lines)*4)
    # Adds the buttons
    horitzontal_start, vertical_start, width, height  = 0.1, 0.8, 0.2, 0.04
    axes_list = []
    for i in range(len(layers_neurons)):
        line = int(i / 4)
        if line == lines and excedent==2:
            h = horitzontal_start + (width * ((i+1) % 4))
        elif line == lines and excedent == 1:
            h = 0.4
        elif line == lines and excedent == 3:
            h = horitzontal_start+ (width/2) + (width * (i % 4))
        else:
            h = horitzontal_start+(width*(i%4))
        v = 0.01+(height*(lines-line))
        ax = plt.axes([h, v, 0.195, 0.035])
        LIST_OF_BUTTONS_TO_NO_DETACH.append(Button(ax, layers_neurons[i][0]+'('+str(layers_neurons[i][1])+')',
                    color='lemonchiffon', hovercolor='green'))
        axes_list.append((ax, layers_neurons[i][0]))
        LIST_OF_BUTTONS_TO_NO_DETACH[-1].on_clicked(lambda event: _on_click_another_layer(event, network_data,
                                                                                          entity_name, axes_list, master,
                                                                                          entity))


    plt.subplots_adjust(wspace=0.5, hspace=0.8)
    plt.show()

def plot_enitity_one_repetition(network_data, layers=None, entity='class'):
    class_matrix, labels = network_data.get_entinty_co_ocurrence_matrix(layers=layers,entity=entity,operation='1/2')
    #Axis 0 = Layers
    class_matrix[:, range(class_matrix.shape[1]), range(class_matrix.shape[2])] = 0
    class_matrix[class_matrix < 0.0001] = 0
    one_rep = []
    more_than_one_rep = []
    for i in range(len(class_matrix)):

        one_rep.append(float(np.sum((class_matrix[i]<0.6) & (class_matrix[i]>0.1)))
                       / len(network_data.get_layer_by_name(layer=layers[i]).neurons_data))
        more_than_one_rep.append(np.sum(class_matrix[i] > 0.75)
                                 / len(network_data.get_layer_by_name(layer=layers[i]).neurons_data))

    plt.title('Pairs PC/#Neurons of layer on '+network_data.model.name)
    plt.plot(layers,one_rep, label='Only one repeated pair')
    plt.plot(layers, more_than_one_rep, label='More than one repeated pairs')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()




def plot_relevant_nf(network_data, layer_name, neuron_idx, layer_to_ablate, master = None, entity='class', th = 0.5):
    plt.figure()
    #given a nefesimodel, a layer, and a dictionary of the population codes for a given class, plots the different neuron_features were the class appears
    # first we create a dictionary with all the classes above a threshold (here 0.1)
    entity_info = {}
    #Detect all neurons with this entity_name in his population code (and sabes idx, entities and value of entity_name)
    objective_neuron = network_data.get_neuron_of_layer(layer=layer_name, neuron_idx=neuron_idx)
    relevance_idx = objective_neuron.get_relevance_idx(network_data=network_data, layer_name=layer_name,
                                             neuron_idx=neuron_idx, layer_to_ablate=layer_to_ablate)

    relevant_neurons = np.where(relevance_idx >= np.max(relevance_idx)*th)[0]
    relevant_neurons = relevant_neurons[relevance_idx[relevant_neurons].argsort()][::-1]

    relevant_neurons_data = []
    for relevant_neuron_idx in relevant_neurons:
        neuron = network_data.get_neuron_of_layer(layer=layer_to_ablate, neuron_idx=relevant_neuron_idx)
        if entity == 'class':
            ents = neuron.class_selectivity_idx(labels=network_data.default_labels_dict)
        elif entity == 'object':
            ents = neuron.concept_selectivity_idx(layer_data=layer_to_ablate,
                                                  network_data=network_data, neuron_idx=relevant_neuron_idx, concept=entity)
        elif entity == 'color':
            ents = neuron.color_selectivity_idx(network_data=network_data, layer_name=layer_to_ablate,
                                                neuron_idx=relevant_neuron_idx)

        relevant_neurons_data.append((relevant_neuron_idx, tuple(ents['label']), relevance_idx[relevant_neuron_idx]))



# then we create the dictionary with the layers and pc where class_name appears
    pc_dict = {}
    for i, neuron in enumerate(relevant_neurons_data):
        pc_number = 0 if neuron[1][0] == 'None' else len(neuron[1])
        if pc_number not in pc_dict:
            pc_dict[pc_number] = [neuron]
        else:
            pc_dict[pc_number].append(neuron)






#finally we plot the result (neuron features of neurons of layer_name, activated by class_name)

    nf_size = network_data.get_neuron_of_layer(layer_to_ablate, 1).neuron_feature.size[0]

    pcs_of_layer = list(pc_dict.keys())
    pcs_of_layer.sort()
    image_axes = np.zeros(len(pcs_of_layer), np.object)
    for k, j in enumerate(pcs_of_layer):
        neurons_num=len(pc_dict[j])

        neuronfeature= Image.new('RGB',(nf_size*neurons_num,nf_size))
        for i in range(neurons_num):
            neuron_num= pc_dict[j][i][0]
            neuronfeature.paste(network_data.get_neuron_of_layer(layer_to_ablate, neuron_num).neuron_feature,(i*nf_size,0))

            sub_axis = plt.subplot(len(pcs_of_layer),1,k+1)

            image_axes[k] = (pc_dict[j][i][0], sub_axis, pc_dict[j])
            plt.imshow(neuronfeature,shape=(200,200))

            plt.subplots_adjust(hspace=.001)

            plt.xticks([])
            plt.yticks([])

            label=''
            for name in pc_dict[j][i][1]:
                label+=str(name).split(',')[0]+'\n'
            label += 'Rel: '+str(round(pc_dict[j][i][2],2))

            font_size = int(interp1d([1,3],[12,6], bounds_error=False, fill_value=6)(len(pcs_of_layer)))
            plt.text(0.01+i/neurons_num, -0.1, label, {'fontsize': font_size},
                     horizontalalignment='left',
                     verticalalignment='top',
                     rotation=0,
                     clip_on=False,
                     transform=plt.gca().transAxes)

            plt.text(0.01 + i / neurons_num, 0.9,pc_dict[j][i][0], {'fontsize': font_size},
                     horizontalalignment='left',
                     verticalalignment='top',
                     rotation=0,
                     clip_on=False,
                     transform=plt.gca().transAxes)

            plt.ylabel(' PC = %d ' %(j))
            plt.gcf().subplots_adjust(bottom=0.3)


    plt.gcf().canvas.mpl_connect('button_press_event', lambda event: _on_click_image(event,master,network_data,
                                                                                     layer_to_ablate, image_axes, entity))
    plt.suptitle('Relevant neurons for '+ layer_name+'-'+str(neuron_idx)+' in '+layer_to_ablate+' [th = '+str(th)+']',
                 y=0.96)

    plt.subplots_adjust(wspace=0.5, hspace=0.8)
    plt.show()


def _on_click_another_layer(event, network_data, entity_name, axes_list, master=None, entity='class'):
    for axe, layer_name in axes_list:
        if axe == event.inaxes:
            plot_pc_of_class(network_data, layer_name, entity_name, master=master, entity=entity)

def _on_click_image(event,master, network_data, layer_name, axes, entity):
    from ..interface.popup_windows.neuron_window import NeuronWindow
    if event.xdata != None and event.ydata != None and event.dblclick:
        for ax in axes:
            if event.inaxes == ax[1]:
                max = ax[1].dataLim.x1
                images = len(ax[2])
                each_image_width = max/images
                for i in range(images):
                    if each_image_width*i<event.xdata<each_image_width*(i+1):
                        neuron_idx = ax[2][i][0]
                        possible_plots = ['Neuron Info', 'Relevant Neurons']
                        to_plot = possible_plots[0] if layer_name == network_data.layers_data[0].layer_id else \
                            get_value_from_popup_combobox(master = master, values = possible_plots,
                                                          text = 'Select the plot to open')
                        if to_plot == possible_plots[0]:
                            print('Opening Neuron ' + str(neuron_idx) + ' of layer ' + layer_name)
                            neuron_window = NeuronWindow(master, network_data=network_data, layer_to_evaluate=layer_name,
                                     neuron_idx=neuron_idx)
                            master.wait_window(neuron_window.window) #Because of this is blocked when you open one
                        elif to_plot == possible_plots[1]:
                            ablatable_layers = network_data.get_ablatable_layers(layer_name)
                            text = 'Select the layer where want relevant neurons'
                            layer_to_ablate = get_value_from_popup_combobox(master, values = ablatable_layers,
                                                                            text = text, default = ablatable_layers[-1])
                            if layer_to_ablate != -1:
                                th = get_value_from_popup_entry(master=master, network_data=network_data,
                                                                index='relevance',maxim=1.0, start=0.75,
                                        text= '% over max relevance to consider relevant')
                                if th != -1:
                                    plot_relevant_nf(network_data=network_data, layer_name=layer_name,
                                                 neuron_idx=neuron_idx, layer_to_ablate=layer_to_ablate,
                                                 master=master, entity=entity,th=th)


                        break
                break

def get_value_from_popup_combobox(master, values, text, default= None):
    popup_window = ComboboxPopupWindow(master, values=values, text=text, default=default)
    master.wait_window(popup_window.top)
    return popup_window.value

def get_value_from_popup_entry(master,network_data,index,maxim=1.0,start=0.75,text='Entry value'):
    popup_window = SpecialValuePopupWindow(master=master, network_data=network_data, index=index,max=maxim,
                                           start=start,text=text)
    master.wait_window(popup_window.top)
    return popup_window.value
def plot_pixel_decomposition(activations, neurons, img, loc, rows=1):
    """Plots the neuron features that provokes the maximum
    activations on specific pixel from an image.

    :param activations: First output value of the function `nefesi.network_data.
        NetworkData.get_max_activations()`.
    :param neurons: Second output value of the function `nefesi.network_data.
        NetworkData.get_max_activations()`.
    :param img: A PIL image instance, the same image used in `nefesi.network_data.
        NetworkData.get_max_activations()`.
    :param loc: Third output value of the function `nefesi.network_data.
        NetworkData.get_max_activations()`.
    :param rows: Integer, number of rows where to show the neuron features.

    :return:
    """
    nf = []
    img = img.copy()
    for n in neurons:
        nf.append(n.neuron_feature)

    color_map = 'gray'
    if nf[0].mode == 'L':
        color_map = 'gray'

    n_images = len(nf)
    ri, rf, ci, cf = loc

    dr = ImageDraw.Draw(img)
    dr.rectangle([(ci, ri), (cf, rf)], outline='red')
    del dr

    fig = plt.figure()
    fig.add_subplot(rows+1, 1, 1)
    plt.imshow(img)
    plt.axis('off')

    for n in range(n_images):
        img = nf[n]
        c = np.ceil(n_images/float(rows))
        a = fig.add_subplot(rows+1, c, n + c + 1)
        plt.imshow(img, interpolation='bicubic', cmap=color_map)
        plt.axis('off')
        t = round(activations[n], 2)
        a.set_title(t)

    plt.show()


def plot_decomposition(activations, neurons, locations, img, plot_nf_list=False):
    """Plots a composition of a neuron feature or image, with the neuron features
    that provokes the maximum activations in each pixel.

    :param activations: First output value of the function `nefesi.network_data.
        NetworkData.get_max_activations()`.
    :param neurons: Second output value of the function `nefesi.network_data.
        NetworkData.get_max_activations()`.
    :param locations: Third output value of the function `nefesi.network_data.
        NetworkData.get_max_activations()`.
    :param img: A PIL image instance. Should be the same image used in
        `nefesi.network_data.NetworkData.get_max_activations()`.
        If the decomposition is about a neuron feature, this argument should be
        the fourth output value of the function `nefesi.network_data.
        NetworkData.get_max_activations()`.
    :param plot_nf_list: Boolean

    :return:
    """
    nf = []
    img = img.copy()
    color_map = None
    if img.mode == 'L':
        color_map = 'gray'
    for n in neurons:
        nf.append(n.neuron_feature)

    w, h = nf[0].size
    rows = 1
    if plot_nf_list:
        rows = 2

    fig = plt.figure()
    fig.add_subplot(rows,2,1)
    plt.imshow(img, cmap=color_map)
    plt.axis('off')

    for i in range(len(activations)-1, -1, -1):
        ri, rf, ci, cf = locations[i]
        print(locations[i])
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
    plt.imshow(img, cmap=color_map)
    plt.axis('off')

    if plot_nf_list:
        if nf[0].mode == 'L':
            color_map = 'gray'

        num_images = len(nf)
        for i in range(num_images):
            fig.add_subplot(rows, num_images, i + num_images + 1)
            plt.imshow(nf[i], cmap=color_map)
            plt.axis('off')
    plt.show()


def plot_similarity_tsne(layer_data, n=None):
    """Plots the neuron feature on the layer `layer_data`
    applying the TSNE algorithm to the similarity index of this layer.

    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    :param n: Optional, list of `nefesi.neuron_data.NeuronData` instances.
        If `n` is not None, plots only the neuron features in `n`.

    :return:

    :raise: If layer `layer_data` has not similarity index.

    """
    if layer_data.similarity_index is None:
        raise ValueError("The similarity index in layer {},"
                         " is not yet calculated.".format(layer_data.layer_id))

    neurons = layer_data.neurons_data
    idx_neurons = None
    if n is not None:
        idx_neurons = [neurons.index(i) for i in n]
        neurons = n
        print(idx_neurons)

    num_neurons = len(neurons)

    x = layer_data.get_similarity_idx(neurons_idx=idx_neurons)
    x_result = TSNE(n_components=2, metric='euclidean',
                    random_state=0).fit_transform(x)
    nf = [n.neuron_feature for n in neurons]
    fig, ax = plt.subplots()

    size_fig = fig.get_size_inches()
    nf_size = nf[0].size
    zoom = (size_fig[0] + size_fig[1])*2 / nf_size[0]

    for i, x, y in zip(range(num_neurons), x_result[:, 0], x_result[:, 1]):
        imscatter(x, y, nf[i], zoom=zoom, ax=ax, label=str(i))
        ax.plot(x, y)
    plt.axis('off')
    plt.show()


def plot_similarity_circle(layer_data, target_neuron, bins=None):
    """Plots the neuron features in circle from a layer,
     based on the similarity index distance between the
     `target_neuron` and the rest of neurons in that layer.

    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    :param target_neuron: The `nefesi.neuron_data.NeuronData` instance.
    :param bins: List of Integers. Bins applied to similarity indexes
        thresholds.

    :return:
    """
    if bins is None:
        bins = [0.0, 0.4, 0.8, 1.0]

    neurons = layer_data.neurons_data

    target_neuron_idx = np.where(neurons == target_neuron)[0][0]

    fig, ax = plt.subplots()
    size_fig = fig.get_size_inches()
    nf_size = target_neuron.neuron_feature.size
    zoom = (size_fig[0] + size_fig[1]) / nf_size[0]

    fig_center = (0.5, 0.5)
    r = [0.15, 0.25, 0.5]
    imscatter(fig_center[0], fig_center[1],
              target_neuron.neuron_feature, zoom=zoom, ax=ax)
    ax.plot(fig_center[0], fig_center[1])

    for i in range(3):
        neuron_data, _ = layer_data.similar_neurons(
            target_neuron_idx, inf_thr=bins[-(i+2)], sup_thr=bins[-(i+1)])

        nf = [n.neuron_feature for n in neuron_data if n is not target_neuron]
        num_neurons = len(nf)

        radius = r[i]
        circle = plt.Circle(fig_center, radius, fill=False)
        degrees = [j*(360/num_neurons) for j in range(num_neurons)]
        ax.add_artist(circle)

        x1 = fig_center[0]
        y1 = fig_center[1] + radius
        x_coord = [x1 + radius * np.sin(d) for d in degrees]
        y_coord = [y1 - radius * (1 - np.cos(d)) for d in degrees]

        for idx, x, y in zip(range(num_neurons), x_coord, y_coord):
            imscatter(x, y, nf[idx], zoom=zoom, ax=ax)
            ax.plot(x, y)

    ax.set_aspect('equal', adjustable='datalim')
    # ax.plot()  # Causes an autoscale update.
    plt.axis('off')
    plt.show()


def imscatter(x, y, image, ax=None, zoom=1, label=None):
    """Plots an image on the `x` and `y` coordinates inside
    another plot graph.
    """
    if ax is None:
        ax = plt.gca()

    color_map = None
    if image.mode == 'L':
        color_map = 'gray'

    im = OffsetImage(image, zoom=zoom, interpolation='bicubic', cmap=color_map)
    x, y = np.atleast_1d(x, y)
    ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
    #ax.text(x+2, y+2, label) # TODO: put labels (index of the neuron)
    ax.add_artist(ab)
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


def plot_2d_index(selective_neurons):
    """Plots the neuron features from a layer comparing two selectivity
    indexes in the two axis of the plot.

    :param selective_neurons: Output value of function
        `nefesi.network_data.NetworkData.get_selective_neurons()`.

    :return:

    :raise:
        ValueError: If the number of indexes in `selectivity neurons`
         are not two.
    """
    index_name = selective_neurons.keys()[0]
    if type(index_name) is tuple and len(index_name) == 2:
        layers_v = selective_neurons.values()[0]

        for k, v in layers_v.items():
            layer_name = k
            neurons = v

            x_values = [n[1] for n in neurons]
            y_values = [n[2] for n in neurons]
            nf = [n[0].neuron_feature for n in neurons]

            fig, ax = plt.subplots()
            size_fig = fig.get_size_inches()
            nf_size = nf[0].size
            zoom = (size_fig[0]+size_fig[1])/nf_size[0]
            for i, x, y in zip(range(len(nf)), x_values, y_values):
                imscatter(x, y, nf[i], zoom=zoom, ax=ax)
                ax.plot(x, y)
            plt.title("Layer: " + layer_name)
            plt.xlabel(index_name[0] + " index")
            plt.ylabel(index_name[1] + " index")
            plt.show()
    else:
        if len(index_name) != 2:
            raise ValueError("This function only can plot 2 indexes.")


def plot_nf_search(selective_neurons, n_max=150):
    """Plots the neuron features in `selective_neurons`, order by their
    index values.

    :param selective_neurons: Output value of function
        `nefesi.network_data.NetworkData.get_selective_neurons()`.
    :param n_max: Integer, max number of neuron features to plot.

    :return:
    """
    index_name = selective_neurons.keys()[0]
    layers_v = selective_neurons.values()[0]

    for k, v in layers_v.items():

        layer_name = k
        neurons = v
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

            color_map = None
            if n[0].neuron_feature.mode == 'L':
                color_map = 'gray'

            plt.imshow(n[0].neuron_feature, interpolation='bicubic', cmap=color_map)
            plt.axis('off')
            tmp_t = ''
            for t in title:
                tmp_t += str(round(t, 2)) + ','
            a.set_title(tmp_t[:-1])
        # plt.tight_layout()
        plt.show()
        fig.clear()


def plot_similarity_idx(neuron_data, sim_neuron, idx_values, rows=2):
    """Plots the list of neuron features in `sim_neuron`, similars to
    the `neuron_data` and their respective similarity index values.

    :param neuron_data: The `nefesi.network_data.NetworkData` instance.
    :param sim_neuron: First output value from the function
        `nefesi.layer_data.LayerData.similar_neurons()`.
    :param idx_values: Second output value from the function
        `nefesi.layer_data.LayerData.similar_neurons()`.
    :param rows: Integer, number of rows where to show the neuron features
        from `sim_neurons`.

    :return:
    """
    # removes to itself from the list
    sim_neuron.remove(neuron_data)

    n_images = len(sim_neuron)
    titles = [round(v, 2) for v in idx_values]

    fig = plt.figure()
    fig.suptitle('Similarity index')

    fig.add_subplot(rows+1, 1, 1)

    color_map = None
    if neuron_data.neuron_feature.mode == 'L':
        color_map = 'gray'

    plt.imshow(neuron_data.neuron_feature, interpolation='bicubic', cmap=color_map)
    plt.axis('off')

    for i, (n, title) in enumerate(zip(sim_neuron, titles)):
        c = np.ceil(n_images/float(rows))
        a = fig.add_subplot(rows+1, c, i + c+1)
        plt.imshow(n.neuron_feature, interpolation='bicubic', cmap=color_map)
        plt.axis('off')
        a.set_title(title)
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    fig.clear()


def plot_neuron_features(layer_data, neuron_list=None):
    """Plot the neuron features from `layer_data` or the neuron features
    in `neuron_list`.

    :param layer_data: The `nefesi.layer_data.LayerData` instance.
    :param neuron_list: List of `nefesi.neuron_data.NeuronData` instances.

    :return:
    """
    nf = []
    if neuron_list is None:
        neuron_list = layer_data.neurons_data

    for f in neuron_list:
        nf.append(f.neuron_feature)

    color_map = None
    if nf[0].mode == 'L':
        color_map = 'gray'

    n_images = len(nf)
    cols = int(math.sqrt(n_images))
    fig = plt.figure()
    fig.suptitle('NF from layer: ' + layer_data.layer_id)
    for n, img in enumerate(zip(nf)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.imshow(nf[n], interpolation='bicubic', cmap=color_map)
        plt.axis('off')
        a.set_title(str(n))
    # fig.set_size_inches(n_max*3,n_max*3)
    plt.show()
    fig.clear()


def main():

    from nefesi.network_data import NetworkData

    t = NetworkData.load_from_disk('/home/oprades/oscar/vgg16_new.obj',
                                   model_file='/home/oprades/oscar/vgg16.h5')


    # sel_idx = t.get_selectivity_idx(['class'], ['fc1', 'fc2'])
    #
    # print sel_idx
    # plot_sel_idx_summary(sel_idx)

    # plot_symmetry_distribution_summary(sel_idx)
    # plot_similarity_circle(t.layers[1], t.layers[1].filters[45])
    # plot_activation_curve(t, t.layers[1], 34)
    # compute_nf(my_net, l, l.get_filters())
    # plot_neuron_features(l)
    # print selective_neurons.values()[0].keys()

    selective_neurons = t.get_selective_neurons('block1_conv1', 'color')
    print(selective_neurons)
    # plot_2d_index(selective_neurons)
    #
    plot_nf_search(selective_neurons)
    # # plot_neuron_features(layer1)
    #
    # plot_nf_search(selective_neurons)

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

    # sel_idx = my_net.get_selectivity_idx(['symmetry'], layer_names)

    # plot_sel_idx_summary(sel_idx)

    # plot_symmetry_distribution_summary(sel_idx)

    # decomposition
    # img_name = 'n03100240/n03100240_2539.JPEG'
    # act, neurons, loc = my_net.get_max_activations(3, img_name, (130, 190), 10)
    #img = load_img(dataset + img_name, target_size=(224, 224))
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