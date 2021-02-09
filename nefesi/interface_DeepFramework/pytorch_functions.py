
class LoadModel():
    """

    """
    def __init__(self, model_name):
        pass

    @property
    def name(self):
        pass

    @property
    def layers(self):
        pass

    @property
    def input(self):
        pass

    def get_layer(self, layer_id):
        pass

    def save(self, model_name):
        pass


class DataBatchGenerator():
    def __init__(self, preprocessing_function, src_dataset, target_size,
                         batch_size, color_mode):
        pass

    # the attributes that are called
    @property
    def iterator(self):
        pass

    @property
    def samples(self):
        pass

    @property
    def batch_index(self):
        pass

    @property
    def batch_size(self):
        pass

    @property
    def filenames(self):
        pass

    @property
    def index_array(self):
        pass


def get_preprocess_function(model_name):
    pass


def intermediate_funcs(inp, outputs):
    pass
