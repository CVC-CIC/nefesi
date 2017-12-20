
from read_activations import get_sorted_activations
from neuron_feature import compute_nf

class LayerData(object):

    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.filters = None


    def get_layer_id(self):
        return self.layer_id

    def set_max_activations(self):
        for f in self.filters:
            f.set_max_activations()

    def evaluate_activations(self, file_names, images, model, num_max_activations, batch_size):
        self.filters = get_sorted_activations(file_names, images, model,
                                              self.layer_id, self.filters, num_max_activations, batch_size)

    def build_neuron_feature(self, dataset_path, model):
        self.filters = compute_nf(dataset_path, model, self.layer_id, self.filters)

    def get_filters(self):
        return self.filters

    def get_selectivity_idx(self, model, index_name, dataset_path, **kwargs):
        sel_idx = []
        for f in self.filters:
            if index_name == 'color':
                res = f.color_selectivity_idx(model, self.layer_id, self.filters.index(f), dataset_path)
                sel_idx.append(res)
            elif index_name == 'orientation':
                degrees = kwargs.get('degrees')
                n_rotations = kwargs.get('n_rotations')
                res = f.orientation_selectivity_idx(model, self.layer_id, self.filters.index(f), dataset_path,
                                                    degrees, n_rotations)
                sel_idx.append(res)
            elif index_name == 'symmetry':
                res = f.symmetry_selectivity_idx(model, self.layer_id, self.filters.index(f), dataset_path)
                sel_idx.append(res)
            elif index_name == 'class':
                pass
        return sel_idx
