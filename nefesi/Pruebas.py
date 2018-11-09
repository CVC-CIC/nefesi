import os
import pickle
from nefesi.class_index import get_hierarchical_population_code_idx,get_class_selectivity_idx, get_population_code_idx
from nefesi.util.general_functions import get_hierarchy_of_label
import numpy as np
from anytree import PreOrderIter,LevelOrderGroupIter
import pandas as pd

def fuse_imageNet(imagenet_base_path,dataset_base_path):
    def _addapt_dataset(dataset_base_path, first_base, file_name=None):
        dirs = os.listdir(dataset_base_path)
        for file in dirs:
            if os.path.isdir(dataset_base_path+'/'+file):
                _addapt_dataset(dataset_base_path+'/'+file,first_base, file_name=file)
            else:
                    dst_dir = first_base + '/' + file_name
                    print(dst_dir)
                    if not os.path.exists(dst_dir):
                        os.mkdir(dst_dir)
                    os.symlink(dataset_base_path+'/'+file, dst_dir+'/'+file)
    _addapt_dataset(imagenet_base_path, dataset_base_path)


def script_classes():
    model = pickle.load(open('../Data/WithImageNet/vgg16_with_fc.obj','rb'))

    final_table = []
    for layer in model.layers_data:
        layer_name = layer.layer_id
        print (layer_name)
        for id, neuron in enumerate(layer.neurons_data):
            class_selectivity = neuron.class_selectivity_idx(labels=model.default_labels_dict)
            pc = neuron.population_code_idx(labels=model.default_labels_dict,threshold=0.1)
            tree = get_hierarchical_population_code_idx(neuron, xml='imagenet_structure.xml', threshold_pc=0.1, population_code=pc,
                                                 class_sel=class_selectivity)
            children = tree.children
            if len(children)>0:
                for child in children:
                    level = 1
                    first_level = child.name
                    best_rep = child.rep
                    last_name = child.name
                    sub_childs = child.children
                    while(len(sub_childs)>0):
                        if len(sub_childs)==1:
                            sub_child = sub_childs[0]
                            last_name = sub_child.name
                            if type(last_name) is tuple:
                                last_name = last_name[0]
                            sub_childs = sub_child.children
                            level+=1
                        else:
                            break
                    leafs = [[node.name,node.freq] for node in PreOrderIter(child) if node.is_leaf]
                    for i in range(len(leafs)):
                        if type(leafs[i][0]) is tuple:
                            leafs[i][0] = leafs[i][0][0]

                    for leaf_name, leaf_freq in leafs:
                        final_table.append((layer_name, id, class_selectivity[1], pc, leaf_name,np.round(leaf_freq,5), first_level,last_name,level,best_rep))

            else:
                final_table.append((layer_name, id, class_selectivity[1], pc, '', 0.0, '', '',
                                   0, 0))

    array = np.array(final_table, dtype=(
        [('layer', 'U32'), ('neuron', np.int), ('class_selectivity_idx', np.float), ('population_code', np.int),
         ('class_id', 'U64'), ('class_freq', np.float), ('first_wordnet_level', 'U128'),
         ('last_common_level', 'U128'),
         ('deep_level', np.int), ('included_categories', np.int)]))
    pd.DataFrame(array).to_csv('ExperimentPC_Th01.csv')
    pickle.dump(model, open('../Data/WithImageNet/vgg16_with_fc_and_index.obj', 'wb'))

def tree_of_ImageNet():
    labels = np.array(os.listdir('/data/local/datasets/ImageNet/train'))
    freq = np.zeros(len(labels), np.float)
    for i, label in enumerate(labels):
        freq[i] = len(os.listdir('/data/local/datasets/ImageNet/train/'+label))
    freq/=np.sum(freq)

    tree = get_hierarchy_of_label(labels=labels,freqs=freq, xml='imagenet_structure.xml', population_code=1000,class_sel=1)
    array = []
    for level, children in enumerate(LevelOrderGroupIter(tree)):
        for node in children:
            name = node.name
            if type(name) is tuple:
                name = name[0]
            leafs = len([0 for child in PreOrderIter(node) if child.is_leaf])
            array.append((name,node.freq,level,node.is_leaf, leafs))

    array = np.array(array, dtype=(
        [('label', 'U128'), ('frequency', np.float), ('level', np.int), ('is_leaf', np.bool), ('leafs_contained', np.int)]))
    pd.DataFrame(array).to_csv('FrequencyOfImageNetClasses.csv')



if __name__ == '__main__':
    #fuse_imageNet(imagenet_base_path='./',dataset_base_path='./')
    #script_classes()
    tree_of_ImageNet()