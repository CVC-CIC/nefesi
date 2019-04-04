
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
from PIL import ImageDraw, Image
from PIL.Image import ANTIALIAS
from sklearn.manifold import TSNE
from matplotlib import gridspec
from ..symmetry_index import SYMMETRY_AXES
from .general_functions import get_n_circles_well_distributed, get_n_circles_TSNE
from ..class_index import get_ntop_population_code
import matplotlib.cbook as cbook

FONTSIZE_BY_LAYERS = [None, 17, 15, 12, 10, 8]
APPENDIX_FONT_SIZE = 8
NEURON_FONT_SIZE = 11
INVALID_NEURON_IDX = -1
ORDER = ['tops','bottoms']
CONDITIONS = {'<': (lambda x, y: x < y), '<=': (lambda x, y: x <= y), '==': (lambda x, y: x == y),
              '>': (lambda x, y: x > y), '>=': (lambda x, y: x >= y)}


def get_one_neuron_plot(network_data, layer, neuron_idx, chart='default'):
    chart = chart.lower()
    figure = plt.figure(figsize=(12, 18))
    subplot = figure.add_subplot(gridspec.GridSpec(12, 18)[:-1, :-2])
    if chart == 'activation curve':
        figure = activation_curve(network_data, layer, neuron_idx)
    elif chart == 'similar neurons':
        figure = plot_similar_neurons(network_data, layer, neuron_idx)
    return figure

def plot_similar_neurons(network_data, layer, neuron_idx,min=0., max=1., condition1='>=', condition2=None,
                         order=ORDER[0], max_neurons=15):
    figure = plt.figure(figsize=(12, 18))
    subplot = figure.add_subplot(gridspec.GridSpec(12, 18)[:-1, :-2])
    # Review is this is correct or needs to take the column
    similarity_row = network_data.similarity_idx(layer_name=layer)[0][neuron_idx,:]
    font_size = FONTSIZE_BY_LAYERS[1]

    hidden_annotations, neurons_that_pass_filter = \
        neuron_with_all_indexes_plot(network_data=network_data, sel_idx=similarity_row, sel_idx_to_calcs=similarity_row, subplot=subplot,
                                     font_size=font_size + 2, layer_name=layer,
                                     min=min, max=max, condition1=condition1, condition2=condition2,
                                     max_neurons=max_neurons, order=order, neuron_idx=neuron_idx)
    set_texts_of_one_layer_plot(condition1, condition2, hidden_annotations, 'similarity', layer, max, min,
                                network_data, neurons_that_pass_filter, order, subplot, neuron=neuron_idx)
    return figure, hidden_annotations

def plot_relevant_neurons(network_data, layer, layer_to_ablate, neuron_idx,min=0., max=1., condition1='>=', condition2=None,
                         order=ORDER[0], max_neurons=15):
    figure = plt.figure(figsize=(12, 18))
    subplot = figure.add_subplot(gridspec.GridSpec(12, 18)[:-1, :-2])
    # Review is this is correct or needs to take the column
    relevant_row = network_data.get_neuron_of_layer(layer=layer, neuron_idx=neuron_idx).\
        get_relevance_idx(network_data=network_data, layer_name=layer,neuron_idx=neuron_idx,
                          layer_to_ablate=layer_to_ablate)

    font_size = FONTSIZE_BY_LAYERS[1]

    hidden_annotations, neurons_that_pass_filter = \
        neuron_with_all_indexes_plot(network_data=network_data, sel_idx=relevant_row, sel_idx_to_calcs=relevant_row, subplot=subplot,
                                     font_size=font_size + 2, layer_name=layer,
                                     min=min, max=max, condition1=condition1, condition2=condition2,
                                     max_neurons=max_neurons, order=order, neuron_idx=neuron_idx,
                                     layer_to_ablate=layer_to_ablate)
    set_texts_of_one_layer_plot(condition1, condition2, hidden_annotations, 'relevance', layer, max, min,
                                network_data, neurons_that_pass_filter, order, subplot, neuron=neuron_idx,
                                layer_to_ablate=layer_to_ablate)
    return figure, hidden_annotations


def neuron_with_all_indexes_plot(network_data, neuron_idx, sel_idx, sel_idx_to_calcs, subplot, font_size, layer_name='default',
                                 min =0, max=1, condition1='<=', condition2=None, max_neurons=15, order=ORDER[0], color_map='jet',
                                 annotate_index = False, similarity_idx = None, layer_to_ablate=None):

    circles, hidden_annotations, layer_name, neurons_that_pass_filter, valids_ids, valids_idx = make_one_layer_base_subplot(
        color_map, condition1, condition2, layer_name, max, max_neurons, min, order, sel_idx, sel_idx_to_calcs, subplot,
        neuron_to_non_count=neuron_idx, similarity_matrix=similarity_idx)
    original_neuron_indexes = network_data.get_all_index_of_neuron(layer=layer_name, neuron_idx=neuron_idx)
    layer_name = layer_name if layer_to_ablate is None else layer_to_ablate
    hidden_annotations = np.zeros((len(hidden_annotations),2), dtype=hidden_annotations.dtype)
    for i in range(len(circles)):
        text = 'Neuron '+str(neuron_idx)+' vs Neuron '+str(valids_ids[i])
        indexes = network_data.get_all_index_of_neuron(layer=layer_name, neuron_idx=valids_ids[i])
        for key in original_neuron_indexes.keys():
            if key == 'color':
                text += '\n Color: ' + str(round(np.sum(original_neuron_indexes[key]['value']), ndigits=2))+' vs ' \
                        ''+str(round(np.sum(indexes[key]['value']), ndigits=2))
            elif key == 'orientation':
                text += '\n Orientation(' + str(network_data.default_degrees_orientation_idx) + 'º): ' \
                        ''+str(round(original_neuron_indexes[key][-1], ndigits=2))+' vs ' \
                        ''+str(round(indexes[key][-1], ndigits=2))
            elif key == 'symmetry':
                text += '\n Symmetry: ' + str(round(original_neuron_indexes[key][-1], ndigits=3))+' vs ' \
                        ''+str(round(indexes[key][-1], ndigits=2))
            elif key == 'class':
                text += '\n First class: ' +str(original_neuron_indexes[key][0])+' vs '+str(indexes[key][0])
            elif key == 'population code':
                text += '\n Pop. code (thr=' + str(network_data.default_thr_pc) + '): ' + str(indexes[key])+' vs ' \
                    ''+ str(indexes[key])
            elif key == 'object':
                text += '\n First object: ' +str(original_neuron_indexes[key][0])+' vs '+str(indexes[key][0])

        hidden_annotations[i,0] = set_neuron_annotation(subplot=subplot, text=text,
                                                      position=circles[i],
                                                      layer_name=layer_name, neuron_idx=valids_ids[i],xytext_pos=(10, -5))
        layer = network_data.get_layer_by_name(layer_name)
        neuron = layer.neurons_data[valids_ids[i]]
        neuron_feature = neuron._neuron_feature.resize((25, 25), ANTIALIAS)
        imagebox = OffsetImage(neuron_feature)
        imagebox.image.axes = subplot
        hidden_annotations[i,1] = set_neuron_img_annotation(subplot=subplot, img=imagebox,
                                                          position=circles[i],
                                                          layer_name=layer_name, neuron_idx=valids_ids[i],xybox_pos=(-11,10))
    return hidden_annotations, neurons_that_pass_filter


def activation_curve(network_data, layer, neuron_idx, num_images=10):
    """Plots the curve of the activations values from the neuron
	with index `neuron_idx`. Also plots 1 of each `num_images` above
	the graph.

	:param network_data: The `nefesi.network_data.NetworkData` instance.
	:param layer: The `nefesi.layer_data.LayerData` instance.
	:param neuron_idx: Integer, index of the neuron.
	:param num_images:Integer, number of TOP scoring images plotted.

	:return:
	"""
    layer_data = network_data.get_layer_by_name(layer=layer)
    neuron = layer_data.neurons_data[neuron_idx]
    images = neuron.get_patches(network_data, layer_data)
    activations = neuron.norm_activations

    color_map = None
    if Image.fromarray(images[0].astype('uint8'), 'RGB').mode == 'L':
        color_map = 'gray'
    idx_images = np.arange(0, len(images), len(images)//num_images)
    idx_images[-1] = len(activations)-1
    cols = len(idx_images)

    fig = plt.figure(figsize=(6, cols))
    for n, img_idx in enumerate(idx_images):
        img = Image.fromarray(images[img_idx].astype('uint8'), 'RGB')
        t = round(activations[img_idx], 2)
        a = fig.add_subplot(gridspec.GridSpec(6, cols)[:2, n])
        plt.imshow(img, interpolation='bicubic', cmap=color_map)
        plt.axis('off')
        a.set_title(t)

    fig.add_subplot(gridspec.GridSpec(6, 1)[2:,:])
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

    return fig


def get_one_layer_plot(index, network_data, layer_to_evaluate, special_value=45,
                            color_map='jet',
                       min=0.0, condition1 = '>=',max='1.00', condition2='>=', order=ORDER[0],max_neurons=15):
    index = index.lower()
    font_size = FONTSIZE_BY_LAYERS[1]
    figure = plt.figure(figsize=(12, 18))
    subplot = figure.add_subplot(gridspec.GridSpec(12, 18)[:-1, :])
    # -----------------------------------CALCULATE THE SELECTIVITY INDEX----------------------------------------
    sel_idx = network_data.get_selectivity_idx(sel_index=index, layer_name=layer_to_evaluate,
                                               degrees_orientation_idx=special_value,thr_pc=special_value)[index][0]

    similarity_idx = network_data.get_layer_by_name(layer_to_evaluate[0]).similarity_index
    if index == 'class':
        hidden_annotations, neurons_that_pass_filter = \
            class_neurons_plot(sel_idx, sel_idx_to_calcs=sel_idx['value'], subplot=subplot,layer_name=layer_to_evaluate,
                               font_size=font_size + 2, min=min, max=max, condition1=condition1, condition2=condition2,
                                max_neurons=max_neurons, order=order, network_data=network_data,similarity_idx=similarity_idx)
    elif index == 'color':
        hidden_annotations, neurons_that_pass_filter = \
            color_neurons_plot(sel_idx,sel_idx_to_calcs=sel_idx, network_data=network_data, subplot=subplot,
                               layer_name=layer_to_evaluate, font_size=font_size + 2, min=min, max=max,
                               condition1=condition1, condition2=condition2, max_neurons=max_neurons, order=order,similarity_idx=similarity_idx)

    elif index == 'orientation':
        hidden_annotations, neurons_that_pass_filter = \
            orientation_neurons_plot(sel_idx=sel_idx, sel_idx_to_calcs=sel_idx[:,-1],degrees_rotation=special_value, subplot=subplot,
                                     layer_name=layer_to_evaluate, font_size=font_size + 2,
                               min=min, max=max, condition1=condition1, condition2=condition2,
                               max_neurons=max_neurons, order=order,similarity_idx=similarity_idx)

    elif index == 'symmetry':
        hidden_annotations, neurons_that_pass_filter = \
            symmetry_neurons_plot(sel_idx=sel_idx, sel_idx_to_calcs=sel_idx[:,-1], subplot=subplot,
                                     layer_name=layer_to_evaluate, font_size=font_size + 2,
                               min=min, max=max, condition1=condition1, condition2=condition2,
                               max_neurons=max_neurons, order=order,similarity_idx=similarity_idx)
    elif index == 'population code':
        hidden_annotations, neurons_that_pass_filter = \
            population_code_neurons_plot(sel_idx=sel_idx, sel_idx_to_calcs=sel_idx, network_data=network_data,
                                         thr_pc=special_value, subplot=subplot,
                                      layer_name=layer_to_evaluate, font_size=font_size + 2,
                                      min=min, max=max, condition1=condition1, condition2=condition2,
                                      max_neurons=max_neurons, order=order,similarity_idx=similarity_idx)
    elif index == 'object':
        hidden_annotations, neurons_that_pass_filter = \
            object_neurons_plot(sel_idx, sel_idx_to_calcs=sel_idx['value'], subplot=subplot,layer_name=layer_to_evaluate,
                               font_size=font_size + 2, min=min, max=max, condition1=condition1, condition2=condition2,
                                max_neurons=max_neurons, order=order, network_data=network_data,similarity_idx=similarity_idx)


    set_texts_of_one_layer_plot(condition1, condition2, hidden_annotations, index, layer_to_evaluate, max, min,
                                network_data, neurons_that_pass_filter, order, subplot,
                                special_value=special_value)
    return figure, hidden_annotations

def class_neurons_plot(sel_idx, sel_idx_to_calcs, subplot, font_size, layer_name='default',
                    min =0,max=1,condition1='<=', condition2=None, max_neurons=15, order=ORDER[0], color_map='jet',
                       network_data=None, max_concept_labels=2, similarity_idx = None):

    circles, hidden_annotations, layer_name, neurons_that_pass_filter, valids_ids, valids_idx = make_one_layer_base_subplot(
        color_map, condition1, condition2, layer_name, max, max_neurons, min, order, sel_idx, sel_idx_to_calcs, subplot,
        similarity_matrix=similarity_idx)
    concept_selectivity = None
    """
    if network_data.addmits_concept_selectivity():
        layer_data = network_data.get_layer_by_name(layer=layer_name)
        concept_selectivity = [layer_data.neurons_data[id].concept_selectivity_idx(layer_data=layer_data,
                                                                network_data=network_data,
                                                                neuron_idx=id) for id in valids_ids]
    """
    hidden_annotations = np.zeros((len(hidden_annotations), 2), dtype=hidden_annotations.dtype)
    for i in range(len(circles)):
        text = str(valids_idx[i]['label'])
        if concept_selectivity is not None:
            for level_idx, level in enumerate(concept_selectivity[i]):
                text+='\n Level '+str(level_idx)+' concepts:\n'
                for concept_idx,concept in enumerate(level):
                    if concept_idx>=max_concept_labels:
                        break
                    elif concept_idx>0:
                        text+= ', '
                    else:
                        text+='  '
                    text+= concept['class']+'('+str(round(concept['count'],ndigits=2))+')'
        hidden_annotations[i,0] = set_neuron_annotation(subplot=subplot, text=text,
                                                      position=circles[i],
                                                      layer_name=layer_name, neuron_idx=valids_ids[i])
        layer = network_data.get_layer_by_name(layer_name)
        neuron = layer.neurons_data[valids_ids[i]]
        neuron_feature = neuron._neuron_feature.resize((25, 25), ANTIALIAS)
        imagebox = OffsetImage(neuron_feature)
        imagebox.image.axes = subplot
        hidden_annotations[i, 1] = set_neuron_img_annotation(subplot=subplot, img=imagebox,
                                                             position=circles[i],
                                                             layer_name=layer_name, neuron_idx=valids_ids[i],
                                                             xybox_pos=(-11, 10))

    return hidden_annotations, neurons_that_pass_filter

def object_neurons_plot(sel_idx, sel_idx_to_calcs, subplot, font_size, layer_name='default',
                    min =0,max=1,condition1='<=', condition2=None, max_neurons=15, order=ORDER[0], color_map='jet',
                       network_data=None, max_concept_labels=2, similarity_idx = None):

    circles, hidden_annotations, layer_name, neurons_that_pass_filter, valids_ids, valids_idx = make_one_layer_base_subplot(
        color_map, condition1, condition2, layer_name, max, max_neurons, min, order, sel_idx, sel_idx_to_calcs, subplot,
        similarity_matrix=similarity_idx)
    hidden_annotations = np.zeros((len(hidden_annotations), 2), dtype=hidden_annotations.dtype)
    for i in range(len(circles)):
        text = str(valids_idx[i]['label'])
        hidden_annotations[i,0] = set_neuron_annotation(subplot=subplot, text=text,
                                                      position=circles[i],
                                                      layer_name=layer_name, neuron_idx=valids_ids[i])
        layer = network_data.get_layer_by_name(layer_name)
        neuron = layer.neurons_data[valids_ids[i]]
        neuron_feature = neuron._neuron_feature.resize((25, 25), ANTIALIAS)
        imagebox = OffsetImage(neuron_feature)
        imagebox.image.axes = subplot
        hidden_annotations[i, 1] = set_neuron_img_annotation(subplot=subplot, img=imagebox,
                                                             position=circles[i],
                                                             layer_name=layer_name, neuron_idx=valids_ids[i],
                                                             xybox_pos=(-11, 10))

    return hidden_annotations, neurons_that_pass_filter

"""
def concept_neurons_plot(sel_idx, sel_idx_to_calcs, subplot, font_size, layer_name='default',
                    min =0,max=1,condition1='<=', condition2=None, max_neurons=15, order=ORDER[0], color_map='jet',
                       network_data=None, max_concept_labels=2):
    circles, hidden_annotations, layer_name, neurons_that_pass_filter, valids_ids, valids_idx = make_one_layer_base_subplot(
        color_map, condition1, condition2, layer_name, max, max_neurons, min, order, sel_idx, sel_idx_to_calcs, subplot)
    layer_data = network_data.get_layer_by_name(layer=layer_name)
    class_selectivity = [layer_data.neurons_data[id].class_selectivity_idx() for id in valids_ids]
    hidden_annotations = np.zeros((len(hidden_annotations), 2), dtype=hidden_annotations.dtype)
    for i in range(len(circles)):
        text = class_selectivity[i][0]+'('+str(class_selectivity[i][1])+')'
        for level_idx, level in enumerate(valids_idx[i]):
            text+='\n Level '+str(level_idx)+' concepts:\n'
            for concept_idx,concept in enumerate(level):
                if concept_idx>=max_concept_labels:
                    break
                elif concept_idx>0:
                    text+= ', '
                else:
                    text+='  '
                text+= concept['class']+'('+str(round(concept['count'],ndigits=2))+')'
        hidden_annotations[i,0] = set_neuron_annotation(subplot=subplot, text=text,
                                                      position=circles[i],
                                                      layer_name=layer_name, neuron_idx=valids_ids[i])
        layer = network_data.get_layer_by_name(layer_name)
        neuron = layer.neurons_data[valids_ids[i]]
        neuron_feature = neuron._neuron_feature.resize((25, 25), ANTIALIAS)
        imagebox = OffsetImage(neuron_feature)
        imagebox.image.axes = subplot
        hidden_annotations[i, 1] = set_neuron_img_annotation(subplot=subplot, img=imagebox,
                                                             position=circles[i],
                                                             layer_name=layer_name, neuron_idx=valids_ids[i],
                                                             xybox_pos=(-11, 10))

    return hidden_annotations, neurons_that_pass_filter
"""

def color_neurons_plot(sel_idx, sel_idx_to_calcs, network_data, subplot, font_size, layer_name='default',
                    min =0,max=1,condition1='<=', condition2=None, max_neurons=15, order=ORDER[0], color_map='jet',
                       similarity_idx = None):

    circles, hidden_annotations, layer_name, neurons_that_pass_filter, valids_ids, valids_idx = make_one_layer_base_subplot(
        color_map, condition1, condition2, layer_name, max, max_neurons, min, order, sel_idx, sel_idx_to_calcs, subplot,
        similarity_matrix=similarity_idx)

    for i in range(len(circles)):
        neuron = network_data.get_neuron_of_layer(layer=layer_name, neuron_idx=valids_ids[i])
        neuron_feature = neuron._neuron_feature.resize((25,25), ANTIALIAS)
        imagebox = OffsetImage(neuron_feature)
        imagebox.image.axes = subplot
        hidden_annotations[i] = set_neuron_img_annotation(subplot=subplot, img=imagebox,
                                                      position=circles[i],
                                                      layer_name=layer_name, neuron_idx=valids_ids[i])

    return hidden_annotations, neurons_that_pass_filter

def orientation_neurons_plot(sel_idx, sel_idx_to_calcs, degrees_rotation,  subplot, font_size, layer_name='default',
                    min =0,max=1,condition1='<=', condition2=None, max_neurons=15, order=ORDER[0], color_map='jet',
                             similarity_idx = None):
    """

    :param valids_idx:
    :param subplot:
    :param colors:
    :param font_size:
    :param max:
    :param min:
    :param max_neurons:
    :param order: 'tops' or 'bottoms'
    :return:
    """
    circles, hidden_annotations, layer_name, neurons_that_pass_filter, valids_ids, valids_idx = make_one_layer_base_subplot(
        color_map, condition1, condition2, layer_name, max, max_neurons, min, order, sel_idx,sel_idx_to_calcs, subplot,
        similarity_matrix=similarity_idx)

    std_between_rotations = np.round(np.std(valids_idx,axis=1),decimals=3)
    for i in range(len(circles)):
        max_selectivity_arg = np.argmax(valids_idx[i,:])
        min_selectivity_arg = np.argmin(valids_idx[i, :])
        label = 'σ\'= '+str(std_between_rotations[i])+'\n' \
                ' max = '+str(round(valids_idx[i,max_selectivity_arg],3))+' at ('+str((max_selectivity_arg+1)*degrees_rotation)+'º)\n' \
                ' min = '+str(round(valids_idx[i,min_selectivity_arg],3))+' at ('+str((min_selectivity_arg+1)*degrees_rotation)+'º)'
        hidden_annotations[i] = set_neuron_annotation(subplot=subplot, text=label, position=circles[i],
                                                  layer_name=layer_name, neuron_idx=valids_ids[i])

    return hidden_annotations, neurons_that_pass_filter

def population_code_neurons_plot(sel_idx, sel_idx_to_calcs, network_data, thr_pc, subplot, font_size, layer_name='default',
                    min =0,max=1,condition1='<=', condition2=None, max_neurons=15, order=ORDER[0], color_map='jet', similarity_idx = None):

    circles, hidden_annotations, layer_name, neurons_that_pass_filter, valids_ids, valids_idx = make_one_layer_base_subplot(
        color_map, condition1, condition2, layer_name, max, max_neurons, min, order, sel_idx,sel_idx_to_calcs, subplot,
        similarity_matrix=similarity_idx)
    n=5
    hidden_annotations = np.zeros((len(hidden_annotations), 2), dtype=hidden_annotations.dtype)
    for i in range(len(circles)):
        neuron = network_data.get_neuron_of_layer(layer=layer_name, neuron_idx=valids_ids[i])
        top = get_ntop_population_code(neuron_data=neuron,threshold_pc=thr_pc,labels=network_data.default_labels_dict)
        top_len = len(top)
        ntop = top[:np.minimum(n,top_len)]
        text = ''
        if top_len == 0:
            text = 'No classes with sel. >= '+str(thr_pc)
        elif top_len>n:
            text+= 'Top '+str(len(ntop))+' of '+str(top_len)+' classes:'
        else:
            text+= 'Selective classes:'
        for label, rel_freq in ntop:
            text+= '\n '+label+' '+str(round(rel_freq*100,ndigits=1))+'%'

        hidden_annotations[i,0] = set_neuron_annotation(subplot=subplot, text=text, position=circles[i],
                                                  layer_name=layer_name, neuron_idx=valids_ids[i])
        layer = network_data.get_layer_by_name(layer_name)
        neuron = layer.neurons_data[valids_ids[i]]
        neuron_feature = neuron._neuron_feature.resize((25, 25), ANTIALIAS)
        imagebox = OffsetImage(neuron_feature)
        imagebox.image.axes = subplot
        hidden_annotations[i, 1] = set_neuron_img_annotation(subplot=subplot, img=imagebox,
                                                             position=circles[i],
                                                             layer_name=layer_name, neuron_idx=valids_ids[i],
                                                             xybox_pos=(-11, 10))

    return hidden_annotations, neurons_that_pass_filter


def symmetry_neurons_plot(sel_idx, sel_idx_to_calcs, subplot, font_size, layer_name='default',
                    min =0,max=1,condition1='<=', condition2=None, max_neurons=15, order=ORDER[0], color_map='jet',
                          similarity_idx = None):
    """

    :param valids_idx:
    :param subplot:
    :param colors:
    :param font_size:
    :param max:
    :param min:
    :param max_neurons:
    :param order: 'tops' or 'bottoms'
    :return:
    """
    circles, hidden_annotations, layer_name, neurons_that_pass_filter, valids_ids, valids_idx = make_one_layer_base_subplot(
        color_map, condition1, condition2, layer_name, max, max_neurons, min, order, sel_idx,sel_idx_to_calcs, subplot,
        similarity_matrix=similarity_idx)

    valids_idx = np.round(valids_idx,decimals=3)
    for i in range(len(circles)):
        text_of_annotation = 'Symmetry Axis:'
        for axis_idx, axis in enumerate(SYMMETRY_AXES):
            text_of_annotation += '\n '+str(axis) + 'º: '+str(valids_idx[i, axis_idx])
        hidden_annotations[i] = set_neuron_annotation(subplot=subplot, text=text_of_annotation, position=circles[i],
                                                  layer_name=layer_name, neuron_idx=valids_ids[i])

    return hidden_annotations, neurons_that_pass_filter



def make_one_layer_base_subplot(color_map, condition1, condition2, layer_name, max, max_neurons, min, order, sel_idx,
                                sel_idx_to_calcs, subplot, neuron_to_non_count = None, similarity_matrix = None):
    if type(layer_name) is list:
        if len(layer_name) == 1:
            layer_name = layer_name[0]
        else:
            raise ValueError("Invalid layer_name" + str(layer_name))

    valids_ids, valids_idx, valids_idx_values, neurons_that_pass_filter = \
        get_neurons_from_constraint(sel_idx_complete=sel_idx, sel_idx_to_use=sel_idx_to_calcs, min=min, max=max,
                                    condition_1=condition1, condition2=condition2, order=order, max_neurons=max_neurons,
                                    neuron_to_non_count = neuron_to_non_count)

    if similarity_matrix is None:
        circles = get_n_circles_well_distributed(valids_idx_values, color_map)
    else:
        similarity_matrix = similarity_matrix[valids_ids][:,valids_ids]
        circles = get_n_circles_TSNE(similarity_matrix, valids_idx_values, valids_ids, color_map)
    circles = circles[np.argsort(circles['y_center'])]
    for i in range(len(circles)):
        subplot.add_patch(circles[i]['circle'])
        subplot.text(circles[i]['x_center'], circles[i]['y_center'], str(round(valids_idx_values[i], 2)),
                     fontdict={'size': NEURON_FONT_SIZE, 'weight': 'bold'},horizontalalignment='center',
                     verticalalignment='center')

    hidden_annotations = np.zeros(len(valids_idx),
                                  dtype=np.dtype(
                                      [('layer_name', 'U64'), ('neuron_idx', np.int), ('annotation', np.object),
                                       ('x0', np.float), ('x1', np.float), ('y0', np.float), ('y1', np.float)]))

    subplot.axis('equal')
    return circles, hidden_annotations, layer_name, neurons_that_pass_filter, valids_ids, valids_idx


def get_neurons_from_constraint(sel_idx_complete, sel_idx_to_use, min, max=1.0, condition_1='<=', condition2=None,
                                order=ORDER[0], max_neurons=15, neuron_to_non_count = None):

    if condition2 is None:
        valids_neurons = CONDITIONS[condition_1](sel_idx_to_use,min)
        if neuron_to_non_count is not None:
            valids_neurons[neuron_to_non_count] = False
        neurons_in_decision = np.where(valids_neurons)[0]
    else:
        valids_neurons_condition_1 = CONDITIONS[condition_1](min, sel_idx_to_use)
        if neuron_to_non_count is not None:
            valids_neurons_condition_1[neuron_to_non_count] = False
        valids_neurons_condition_2 = CONDITIONS[condition2](sel_idx_to_use,max)
        if neuron_to_non_count is not None:
            valids_neurons_condition_2[neuron_to_non_count] = False
        neurons_in_decision = np.where(valids_neurons_condition_1 & valids_neurons_condition_2)[0]
    valids_idx = sel_idx_complete[neurons_in_decision]
    valids_idx_to_use = sel_idx_to_use[neurons_in_decision]
    args_sorted = np.argsort(valids_idx_to_use)
    if order.lower() == ORDER[0]:
        args_sorted = args_sorted[::-1]
    neurons_to_show = np.minimum(len(valids_idx), max_neurons)
    args_sorted = args_sorted[:neurons_to_show]
    if order.lower() == ORDER[0]:
        args_sorted = args_sorted[::-1]
    valids_idx = valids_idx[args_sorted]
    valids_ids = neurons_in_decision[args_sorted]
    valids_idx_values = valids_idx_to_use[args_sorted]
    return valids_ids, valids_idx, valids_idx_values, len(neurons_in_decision)


def set_neuron_img_annotation(subplot, img, position, layer_name=None, neuron_idx=-1, xybox_pos=(10,10)):



    annotation = AnnotationBbox(img, xy=(position['x_center'], position['y_center']),
                                xybox=xybox_pos,
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.3)

    subplot.add_artist(annotation)
    annotation.set_visible(False)

    return (layer_name, neuron_idx, annotation,position['x0'],position['x1'],position['y0'],position['y1'])



def set_neuron_annotation(subplot, text, position, layer_name=None, neuron_idx=-1, xytext_pos =(10, 5)):

    annotation = subplot.annotate(text, xy=(position['x_center'], position['y_center']),
                                  xytext=xytext_pos, textcoords='offset points',
                                  bbox=dict(boxstyle="round", fc="w"))
    annotation.set_visible(False)

    return (layer_name, neuron_idx, annotation,position['x0'],position['x1'],position['y0'],position['y1'])


def set_texts_of_one_layer_plot(condition1, condition2, hidden_annotations, index, layer_to_evaluate, max, min,
                                network_data, neurons_that_pass_filter, order, subplot, special_value=None, neuron=None,
                                layer_to_ablate = 'layer_ablated'):
    subplot.set_xticklabels([])
    subplot.set_yticklabels([])
    sel_idx_label = 'Sel. Index'
    if type(layer_to_evaluate) is list:
        layer_to_evaluate = layer_to_evaluate[0]
    title = index.title() + " Selectivity of layer " + layer_to_evaluate
    if index == 'orientation':
        title+= ' ('+str(special_value) + 'º by rot.)'
    elif index == 'population code':
        title+= ' (thr: '+str(special_value)+')'
        min = int(min)
        if max is not None:
            max = int(max)
        sel_idx_label = 'Popul. code'
    elif index == 'similarity':
        title = 'Neurons Similar to '+layer_to_evaluate+' Neuron '+str(neuron)
        sel_idx_label = 'Similarity'
    elif index == 'relevance':
        title = 'Relevant Neurons for ' + layer_to_evaluate + ' Neuron ' + str(neuron) +' in '+layer_to_ablate
        sel_idx_label = 'Relevance'

    subplot.set_title(title, fontdict={'size': 12, 'weight': 'bold'})
    if condition2 is None:
        constraint_text = 'Constraint : '+sel_idx_label+' '+ condition1 + ' ' + str(min) + ''
    else:
        constraint_text = 'Constraint : ' + str(min) + ' ' + condition1 + ' '+sel_idx_label+' '\
                          + condition2 + ' ' + str(max)
    if neurons_that_pass_filter > len(hidden_annotations):
        constraint_text += ' *'
        total_neurons = network_data.get_len_neurons_of_layer(layer_to_evaluate)
        if index == 'similarity':
            total_neurons-=1
        appendix = '* showing ' + r"$\bf{" +str(len(hidden_annotations)) + "}$ " + order[:-1] + ' neurons (from ' \
                   + r"$\bf{"+str(neurons_that_pass_filter) + "}$/" \
                   +  r"$\bf{"+str(total_neurons)+ "}$ " \
                    " that satisfy the constraint)"
        subplot.figure.text(x=0.17, y=0.05, s=appendix, fontdict={'size': APPENDIX_FONT_SIZE + 2, 'style': 'italic'})
    subplot.set_xlabel(constraint_text, fontdict={'size': 10, 'weight': 'bold'})




def get_plot_net_summary_figure(index, network_data, layersToEvaluate=".*", special_value=None,
                                bins=10, color_map='jet'):
    #-------------------------------------------INITIALIZATIONS---------------------------------------
    index = index.lower()
    #Set the names of layersToEvaluate (in order to plot it if user used RegEx)
    if type(layersToEvaluate) is not list:
        layersToEvaluate = network_data.get_layers_analyzed_that_match_regEx(layersToEvaluate)
    #Initiate the font_size to looks goods with the quantity of layers to show
    font_size = FONTSIZE_BY_LAYERS[-1] if len(layersToEvaluate) >= len(FONTSIZE_BY_LAYERS) else \
                FONTSIZE_BY_LAYERS[len(layersToEvaluate)]
    #Initiate the figure to plot and de color map
    figure = plt.figure(figsize=(12, 18))
    subplot = figure.add_subplot(gridspec.GridSpec(12, 18)[:-1,:-2])
    cmap = plt.cm.get_cmap(color_map)
    colors = [cmap(float(i) / bins) for i in range(bins)]

    #Initate auxiliar arrays that will be usefull in order to plot more beautiful result
     #This array will be filled in function as parameter. Used for take count of bars that appears along the chart
    different_bars = np.zeros(bins+1, dtype=np.dtype([('bar',np.object),('label','U64'),('used',np.bool)]))
     #The labels of axis X, filled in each iteration with some information
    x_axis_labels= np.zeros(len(layersToEvaluate), dtype='U128')
     #The hidden annotations that will appears only when user hover the correspondant bar with the mouse. (Correspondant
      #bar is the rectangle with edges x0,x1,y0,y1)
    hidden_annotations = np.zeros(len(layersToEvaluate),
            dtype=np.dtype([('layer_name','U64'), ('neuron_idx',np.int), ('annotation',np.object),
                            ('x0',np.float), ('x1',np.float), ('y0',np.float), ('y1',np.float)]))

    #-----------------------------------CALCULATE THE SELECTIVITY INDEX----------------------------------------
    sel_idx = network_data.get_selectivity_idx(sel_index=index, layer_name=layersToEvaluate,
                                               degrees_orientation_idx=special_value, thr_pc=special_value)[index]

    #------------------------------------------MAKE PLOTS-------------------------------------------------------
    for pos,(layer_name, sel_idx_of_layer) in enumerate(zip(layersToEvaluate,sel_idx)):
        if index == 'class':
            mean, std, hidden_annotations[pos] = class_layer_bars(sel_idx_of_layer,pos,subplot,colors,different_bars,
                                                                  layer_name=layer_name, font_size=font_size+2)
            x_axis_labels[pos] = layer_name + " \n" \
                                           "μ=" + str(mean) + " σ=" + str(std)
        elif index == 'object':
            mean, std, hidden_annotations[pos] = object_layer_bars(sel_idx_of_layer, pos, subplot, colors,
                                                                  different_bars,
                                                                  layer_name=layer_name, font_size=font_size + 2)
            x_axis_labels[pos] = layer_name + " \n" \
                                              "μ=" + str(mean) + " σ=" + str(std)
        elif index == 'color':
            mean, std, hidden_annotations[pos] = color_layer_bars(sel_idx_of_layer, pos, subplot, colors,
                                                                  different_bars,
                                                                  layer_name=layer_name, font_size=font_size + 2)
            x_axis_labels[pos] = layer_name + " \n" \
                                              "μ=" + str(mean) + " σ=" + str(std)
        elif index == 'orientation':
            mean, std, mean_std_between_rotations,hidden_annotations[pos] = \
                orientation_layer_bars(sel_idx_of_layer, pos, subplot, colors, different_bars,
                                       layer_name=layer_name,font_size=font_size+2)
            if len(layersToEvaluate)>2:
                x_axis_labels[pos] = layer_name + "\n" \
                                           "μ=" + str(mean) + " σ=" + str(std)+"\n" \
                                            "(σ'=" +str(mean_std_between_rotations)+")*"
            else:
                x_axis_labels[pos] = layer_name + "\n" \
                    "μ=" + str(mean) + " σ=" + str(std) + " (σ'=" + str(mean_std_between_rotations) + ")*"
        elif index == 'symmetry':
            mean, std, mean_std_between_axys,hidden_annotations[pos] = \
                symmetry_layer_bars(sel_idx_of_layer, pos, subplot, colors,different_bars,
                                    layer_name=layer_name, font_size=font_size + 2)
            if len(layersToEvaluate) > 2:
                x_axis_labels[pos] = layer_name + "\n" \
                                               "μ=" + str(mean) + " σ=" + str(std) + "\n" \
                                                "(σ'=" + str(mean_std_between_axys) + ")*"
            else:
                x_axis_labels[pos] = layer_name + "\n" \
                                "μ=" + str(mean) + " σ=" + str(std) + " (σ'=" + str(mean_std_between_axys) + ")*"

        elif index == 'population code':
            mean, std, hidden_annotations[pos] = population_code_layer_bars(sel_idx_of_layer, pos, subplot, colors,
                                                                  different_bars,
                                                                  layer_name=layer_name, font_size=font_size + 2)
            x_axis_labels[pos] = layer_name + " \n" \
                                              "μ=" + str(mean) + " σ=" + str(std)
        elif index == 'concept':
            mean, std, hidden_annotations[pos] = concept_layer_bars(sel_idx_of_layer,pos,subplot,colors,different_bars,
                                                                  layer_name=layer_name, font_size=font_size+2)
            x_axis_labels[pos] = layer_name + " \n" \
                                              "μ=" + str(mean) + " σ=" + str(std)
    #-------------------------------SET THE PLOT ADDITIONAL INFORMATION--------------------------------------------

    set_aditional_general_plot_information(index, bins, different_bars, subplot,
                                           x_axis_labels, special_value, font_size)

    return figure,hidden_annotations


def set_aditional_general_plot_information(index, bins, different_bars, subplot,
                                           x_axis_labels, special_value=None, font_size=12):
    """
    Sets the additional plot information on general plots (Tittles, labels of axis, legend...)
    :param index: Str, the name of the index that represents
    :param bins: The number of bins from the histogram
    :param different_bars: The different_bars auxiliar array with the bars used
    :param subplot:
    :param x_axis_labels:
    :param special_value:
    :param font_size:
    :return:
    """
    xticks_fontsize = int(font_size / 1.15) if len(x_axis_labels)>2 and (index=='orientation' or index == 'symmetry') \
                else font_size
    subplot.set_xticks(range(len(x_axis_labels)))
    subplot.set_xticklabels(x_axis_labels, fontsize=xticks_fontsize)
    subplot.set_yticks(range(0, 101, bins))
    subplot.set_yticklabels([str(i) + '%' for i in range(0, 101, bins)])
    subplot.set_ylabel('% of Neurons')
    title = index.title() + ' Selectivity'
    if index == 'orientation':
        title += ' (' + str(special_value) + 'º by rotation)'
    elif index == 'population code':
        title+= ' (thr: '+str(special_value)+')'
    subplot.set_title(title, fontdict={'size': 12, 'weight': 'bold'})

    subplot.figure.legend(different_bars['bar'][different_bars['used']][::-1], different_bars['label'][different_bars['used']][::-1],
                  bbox_to_anchor=(1, 0.85), loc="upper right", title='Selectivity Range')
    if index == 'orientation' or index == 'symmetry':
        appendix = '* σ\' is the mean of std beetween the specific index of each rotation'
        subplot.figure.text(x=0.2, y=0.0025, s=appendix, fontdict={'size': APPENDIX_FONT_SIZE, 'style': 'italic'})


def symmetry_layer_bars(layer_idx, layer_pos, subplot, colors, different_bars, bins=10,layer_name='default',
                        font_size = 12):
    layer_idx_values = layer_idx[:,-1]
    last_bar, mean_selectivity, std_selectivity = plot_bars_in_general_figure(bins, colors, different_bars, font_size,
                                                                         layer_idx_values, layer_pos, subplot)
    mean_std_between_axys = np.mean(np.std(layer_idx[:,:-1], axis=0))
    each_axis_mean = np.round(a=np.mean(layer_idx[:,:-1],axis=0)*100,decimals=1)
    each_axis_std = np.round(a=np.std(layer_idx[:,:-1],axis=0)*100,decimals=1)

    text_of_annotation = 'Symmetry Axis:\n'
    for axis_idx, axis in enumerate(SYMMETRY_AXES):
        text_of_annotation += str(axis) + 'º:\n' \
        ' μ=' + str(each_axis_mean[axis_idx]) + ' σ=' + str(each_axis_std[axis_idx]) + '\n'

    hidden_annotations = get_annotation_for_event(subplot=subplot, different_bars=different_bars, layer_pos=layer_pos,
                                                  actual_bar=last_bar,text=text_of_annotation, y_pos=90-15*len(SYMMETRY_AXES),
                                                  layer_name=layer_name)
    return round(mean_selectivity * 100, 1), round(std_selectivity * 100, 1), round(mean_std_between_axys * 100, 1), \
           hidden_annotations

def orientation_layer_bars(layer_idx, layer_pos, subplot, colors, different_bars, bins=10,
                           layer_name='default', font_size = 12):
    layer_idx_values = layer_idx[:, -1]
    last_bar, mean_selectivity, std_selectivity = plot_bars_in_general_figure(bins, colors, different_bars, font_size,
                                                                         layer_idx_values, layer_pos, subplot)
    mean_std_between_rotations = np.mean(np.std(layer_idx[:,:-1], axis=0))

    hidden_annotation = get_annotation_for_event(subplot=subplot,different_bars=different_bars, layer_pos=layer_pos,
                                                 actual_bar = last_bar, layer_name=layer_name,
                                                 text="TEXTO POR DECIDIR")

    return round(mean_selectivity * 100, 1), round(std_selectivity * 100, 1), round(mean_std_between_rotations * 100, 1),\
           hidden_annotation


def class_layer_bars(layer_idx, layer_pos, subplot, colors, different_bars, bins=10,
                     layer_name='default', font_size = 12):
    layer_idx_values = layer_idx['value']
    bar, mean_selectivity, std_selectivity = plot_bars_in_general_figure(bins, colors, different_bars, font_size,
                                                                         layer_idx_values, layer_pos, subplot)
    hidden_annotation = get_annotation_for_event(subplot=subplot,different_bars=different_bars, layer_pos=layer_pos,
                                                 actual_bar = bar, text="TEXTO POR DECIDIR",
                                                 layer_name=layer_name)

    return round(mean_selectivity*100,1), round(std_selectivity*100,1), hidden_annotation

def object_layer_bars(layer_idx, layer_pos, subplot, colors, different_bars, bins=10,
                     layer_name='default', font_size = 12):
    layer_idx_values = layer_idx['value']
    bar, mean_selectivity, std_selectivity = plot_bars_in_general_figure(bins, colors, different_bars, font_size,
                                                                         layer_idx_values, layer_pos, subplot)
    hidden_annotation = get_annotation_for_event(subplot=subplot,different_bars=different_bars, layer_pos=layer_pos,
                                                 actual_bar = bar, text="TEXTO POR DECIDIR",
                                                 layer_name=layer_name)

    return round(mean_selectivity*100,1), round(std_selectivity*100,1), hidden_annotation

"""
def concept_layer_bars(layer_idx, layer_pos, subplot, colors, different_bars, bins=10,
                     layer_name='default', font_size = 12):

    layer_idx_values = [neuron_concept[0]['count'][0] for neuron_concept in layer_idx]
    bar, mean_selectivity, std_selectivity = plot_bars_in_general_figure(bins, colors, different_bars, font_size,
                                                                         layer_idx_values, layer_pos, subplot)
    hidden_annotation = get_annotation_for_event(subplot=subplot,different_bars=different_bars, layer_pos=layer_pos,
                                                 actual_bar = bar, text="TEXTIÑU RICO",
                                                 layer_name=layer_name)

    return round(mean_selectivity*100,1), round(std_selectivity*100,1), hidden_annotation
"""

def population_code_layer_bars(layer_idx, layer_pos, subplot, colors, different_bars, bins=10,
                     layer_name='default', font_size = 12):
    bar, mean_selectivity, std_selectivity = plot_population_code_bars_in_general_figure(bins, colors, different_bars, font_size,
                                                                         layer_idx, layer_pos, subplot)
    hidden_annotation = get_annotation_for_event(subplot=subplot,different_bars=different_bars, layer_pos=layer_pos,
                                                 actual_bar = bar, text="TEXTO POR DECIDIR",
                                                 layer_name=layer_name)

    return round(mean_selectivity,1), round(std_selectivity,1), hidden_annotation

def color_layer_bars(layer_idx, layer_pos, subplot, colors, different_bars, bins=10,
                     layer_name='default', font_size = 12):
    bar, mean_selectivity, std_selectivity = plot_bars_in_general_figure(bins, colors, different_bars, font_size,
                                                                         layer_idx, layer_pos, subplot)
    hidden_annotation = get_annotation_for_event(subplot=subplot,different_bars=different_bars, layer_pos=layer_pos,
                                                 actual_bar = bar, text="TEXTO POR DECIDIR",
                                                 layer_name=layer_name)

    return round(mean_selectivity*100,1), round(std_selectivity*100,1), hidden_annotation



def plot_bars_in_general_figure(bins, colors, different_bars, font_size, layer_idx_values, layer_pos, subplot):
    counts, bins = np.histogram(layer_idx_values, bins=bins, range=(0., 1.001))
    freq_percent = (counts / len(layer_idx_values)) * 100
    y_offset = 0
    for i in range(len(freq_percent)):
        if not np.isclose(freq_percent[i], 0.):
            bar = subplot.bar(layer_pos, freq_percent[i], bottom=y_offset, width=0.45, color=colors[i])
            if freq_percent[i] > 15:
                digits_to_print = len(str(counts[i]))
                x_offset = (font_size * (digits_to_print - 1) / (100 * digits_to_print))
                subplot.text(layer_pos, y_offset + (freq_percent[i] / 2),
                             str(int((counts[i]/sum(counts))*100))+'%', fontdict={'size': font_size, 'weight': 'bold'}, horizontalalignment='center',
                     verticalalignment='center')
            y_offset += freq_percent[i]
            # if is the first bar of this bin encountered
            if not different_bars[i][2]:
                different_bars[i][0] = bar
                different_bars[i][1] = str(int(bins[i] * 100)) + '% - ' + str(int(bins[i + 1] * 100)) + '%'
                different_bars[i][2] = True
    mean_selectivity = np.mean(layer_idx_values)
    std_selectivity = np.std(layer_idx_values)
    return bar, mean_selectivity, std_selectivity

def plot_population_code_bars_in_general_figure(bins, colors, different_bars, font_size, layer_idx_values, layer_pos, subplot):
    unique_bins, counts = np.unique(layer_idx_values, return_counts=True)#np.histogram(layer_idx_values, unique_bins=unique_bins, range=(0., 1.001))
    unique_bins = np.clip(unique_bins, 0, bins-1)
    first_last = None
    for i in range(len(unique_bins)):
        if unique_bins[i]==(bins-1):
            if first_last is None:
                first_last = i
            else:
                counts[first_last]+= counts[i]
    if first_last is not None:
        unique_bins = unique_bins[:first_last+1]
        counts = counts[:first_last+1]
    freq_percent = (counts / len(layer_idx_values)) * 100
    y_offset = 0
    for i in range(len(freq_percent)):
        if not np.isclose(freq_percent[i], 0.):
            bar = subplot.bar(layer_pos, freq_percent[i], bottom=y_offset, width=0.45, color=colors[unique_bins[i]])
            if freq_percent[i] > 15:
                digits_to_print = len(str(counts[i]))
                x_offset = (font_size * (digits_to_print - 1) / (100 * digits_to_print))
                subplot.text(layer_pos, y_offset + (freq_percent[i] / 2),
                             str(int((counts[i]/sum(counts))*100))+'%', fontdict={'size': font_size, 'weight': 'bold'}, horizontalalignment='center',
                     verticalalignment='center')
            y_offset += freq_percent[i]
            # if is the first bar of this bin encountered
            if str(unique_bins[i]) not in different_bars['label']:
                different_bars[unique_bins[i]][0] = bar
                label = '>= ' if unique_bins[i]==(bins-1) else ''
                different_bars[unique_bins[i]][1] = label+str(unique_bins[i])
                different_bars[unique_bins[i]][2] = True
    mean_selectivity = np.mean(layer_idx_values)
    std_selectivity = np.std(layer_idx_values)
    return bar, mean_selectivity, std_selectivity



def get_annotation_for_event(subplot, different_bars,layer_pos, actual_bar, text, y_pos=70, layer_name='default'):
    annotation = subplot.annotate(text, xy=(layer_pos, 0),
                                  xytext=(layer_pos-0.1, y_pos),
                                  bbox=dict(boxstyle="round", fc="w"))
    annotation.set_visible(False)
    actual_bar = actual_bar[0]
    first_bar = different_bars['bar'][different_bars['used']][0][0] #the bottom (0)
    last_bar = different_bars['bar'][different_bars['used']][-1][0] #the top (100)
    rect_x0, rect_x1, rect_y0, rect_y1 = actual_bar._x0, actual_bar._x1, first_bar._y0, last_bar._y1
    #           layer_name, neuron_idx, annotation,rect_x0,rect_x1,rect_y0,rect_y1
    return (layer_name, INVALID_NEURON_IDX, annotation,rect_x0,rect_x1,rect_y0,rect_y1)

def check_layer_to_evaluate_validity(network_data,layer_to_evaluate):
    if type(layer_to_evaluate) is str:
        layer_to_evaluate = network_data.get_layers_analyzed_that_match_regEx(layer_to_evaluate)
    if type(layer_to_evaluate) is list:
        if len(layer_to_evaluate)==0:
            raise ValueError("Imposible to have a plot of a length-0 layer list")
        elif len(layer_to_evaluate)>1:
            return False
    else:
        raise ValueError("Non valid layer_to_evaluate value: "+str(layer_to_evaluate))
    return True






























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

    images = neuron.get_patches(network_data, layer_data)
    activations = neuron.norm_activations
    images = images[:n_max]
    activations = activations[:n_max]

    color_map = None
    if images[0].mode == 'L':
        color_map = 'gray'

    cols = int(math.sqrt(len(images)))
    n_images = len(images)
    titles = [round(act, 2) for act in activations]
    fig = plt.figure()
    fig.suptitle('Layer: ' + layer_name + ', Filter index: ' + str(neuron_idx))
    for n, (img, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.imshow(img, interpolation='bicubic', cmap=color_map)
        plt.axis('off')
        # a.set_title(title)
    # fig.set_size_inches(n_max*3,n_max*3)
    plt.show()
    fig.clear()


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
    if images[0].mode == 'L':
        color_map = 'gray'

    idx_images = np.arange(0, len(images), num_images)
    cols = len(idx_images)

    fig = plt.figure()
    for n, img_idx in enumerate(idx_images):
        img = images[img_idx]
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
    zoom = (size_fig[0] + size_fig[1]) / nf_size[0]

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

    target_neuron_idx = neurons.index(target_neuron)

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
    # plot_neuron_features(text_label)
    # print selective_neurons.values()[0].keys()

    selective_neurons = t.get_selective_neurons('block1_conv1', 'color')
    print(selective_neurons)
    # plot_2d_index(selective_neurons)
    #
    plot_nf_search(selective_neurons)
    # # plot_neuron_features(layer1)
    #
    # plot_nf_search(selective_neurons)

    # nf = text_label.get_filters()[131].get_neuron_feature()
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