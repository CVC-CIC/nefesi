import os
import pickle
from nefesi.class_index import get_hierarchical_population_code_idx,get_class_selectivity_idx, get_population_code_idx
import numpy as np
from anytree import PreOrderIter
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
            if 'class' not in neuron.selectivity_idx:
                class_selectivity = get_class_selectivity_idx(neuron,labels=model.default_labels_dict)
            else:
                class_selectivity = neuron.selectivity_idx['class']
            if 'population code0.1' not in neuron.selectivity_idx:
                pc = get_population_code_idx(neuron,labels=model.default_labels_dict,threshold_pc=0.1)
            else:
                pc = neuron.selectivity_idx['population code0.1']
            tree = get_hierarchical_population_code_idx(neuron, xml='imagenet_structure.xml', threshold_pc=0.1, population_code=pc,
                                                 class_sel=0)
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


if __name__ == '__main__':
    #fuse_imageNet(imagenet_base_path='./',dataset_base_path='./')
    script_classes()