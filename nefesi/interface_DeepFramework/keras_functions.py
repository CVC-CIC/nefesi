from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K


class LoadModel():
    """

    """
    def __init__(self, model_name):
        self.kerasmodel = load_model(model_name)

    @property
    def name(self):
        return self.kerasmodel.name

    @property
    def layers(self):
        return self.kerasmodel.layers

    @property
    def input(self):
        return self.kerasmodel.input

    def get_layer(self, layer_id):
        """
        Retrieves a layer based on either its name (unique) or index.
        :return:
        """
        return self.kerasmodel.get_layer(layer_id)

    def save(self, model_name):
        """
        save model
        :param model_name:
        :return:
        """
        self.kerasmodel.model.save(model_name)


class DataBatchGenerator():
    def __init__(self, preprocessing_function, src_dataset, target_size,
                         batch_size, color_mode):
        datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
        self.keras_data_batch = datagen.flow_from_directory(
            src_dataset,
            target_size=target_size,
            batch_size=batch_size,
            shuffle=True,
            color_mode=color_mode
        )

    # the attributes that are called
    @property
    def iterator(self):
        return self.keras_data_batch

    @property
    def samples(self):
        return self.keras_data_batch.samples

    @property
    def batch_index(self):
        return self.keras_data_batch.batch_index

    @property
    def batch_size(self):
        return self.keras_data_batch.batch_size

    @property
    def filenames(self):
        return self.keras_data_batch.filenames

    @property
    def index_array(self):
        return self.keras_data_batch.index_array


def get_preprocess_function(model_name):
    if model_name.lower() == 'vgg16':
        from keras.applications.vgg16 import preprocess_input
    elif model_name.lower() == 'resnet50':
        from keras.applications.resnet50 import preprocess_input
    elif model_name.lower() == 'vgg19':
        from keras.applications.vgg19 import preprocess_input
    elif model_name.lower() == 'xception':
        from keras.applications.xception import preprocess_input
    elif model_name.lower() == 'mobilenetv2_1.00_224':
        from keras.applications.mobilenetv2 import preprocess_input
    else:
        preprocess_input = None
    return preprocess_input


def intermediate_funcs(inp, outputs):
    K.learning_phase = 0
    funcs = K.function(inp, outputs)
    return funcs
