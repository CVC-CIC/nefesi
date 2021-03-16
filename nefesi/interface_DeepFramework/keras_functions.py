from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import keras.backend as K
from keras.preprocessing import image
import numpy as np


class DeepModel():
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

    @property
    def input_shape(self):
        return self.kerasmodel.input_shape

    @property
    def _network_nodes(self):
        return self.kerasmodel._network_nodes

    def get_layer(self, layer_name, get_index=False):
        """
        Retrieves a layer based on either its name (unique) or index.
        :return:
        """
        for index, layer in enumerate(self.layers):
            if layer.name == layer_name:
                if get_index:
                    return layer, index
                else:
                    return layer

        raise ValueError('No such layer: ' + layer_name)

    def neurons_of_layer(self, layer_name):
        return self.kerasmodel.get_layer(layer_name).output_shape[-1]

    def save(self, model_name):
        """
        save model
        :param model_name:
        :return:
        """
        self.kerasmodel.save(model_name)

    def calculate_activations(self, layers_name, model_inputs):
        inp = self.kerasmodel.input
        if type(inp) is not list:
            inp = [inp]
        if isinstance(layers_name, str):
            layers_name = [layers_name]

        # uses .get_output_at() instead of .output. In case a layer is
        # connected to multiple inputs. Assumes the input at node index=0
        # is the input where the model inputs come from.
        outputs = [self.kerasmodel.get_layer(layer).output for layer in layers_name]
        # evaluation functions
        # K.learning_phase flag = 1 (train mode)
        # funcs = K.function(inp+ [K.learning_phase()], outputs) #modifies learning parameters
        # layer_outputs = funcs([model_inputs, 1])
        K.learning_phase = 0
        funcs = K.function(inp, outputs)
        layer_outputs = funcs([model_inputs])
        return layer_outputs


class ImageWithNames(DirectoryIterator):
    def __init__(self, *args, **kwargs):
        super(ImageWithNames, self).__init__(*args, **kwargs)
        self.filenames_np = np.array(self.filenames)
        # self.filenames_np = np.array(self.filepaths)

    def _get_batches_of_transformed_samples(self, index_array):
        original_tuple = super(ImageWithNames, self)._get_batches_of_transformed_samples(index_array)
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (self.filenames_np[index_array],))
        return tuple_with_path


class DataBatchGenerator():
    def __init__(self, preprocessing_function, src_dataset, target_size,
                         batch_size, color_mode):
        datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
        # self.keras_data_batch = datagen.flow_from_directory(
        #     src_dataset,
        #     target_size=target_size,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     color_mode=color_mode
        # )
        shuffle = False
        self.keras_data_batch = ImageWithNames(
            src_dataset,
            datagen,
            target_size=target_size,
            batch_size=batch_size,
            shuffle=shuffle,
            color_mode=color_mode
        )

    # the attributes that are called
    @property
    def iterator(self):
        # the batch iterator
        return self.keras_data_batch

    @property
    def samples(self):
        # number of images
        return self.keras_data_batch.samples

    @property
    def batch_index(self):
        return self.keras_data_batch.batch_index

    @property
    def batch_size(self):
        # batch size
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


def _load_multiple_images(src_dataset, img_list, color_mode, target_size, preprocessing_function=None,
                          prep_function=True):
    """Returns a list of images after applying the
     corresponding transformations.

    :param image_names: List of strings, name of the images.
    :param prep_function: Boolean.

    :return: Numpy array that contains the images (1+N dimension where N is the dimension of an image).
    """
    # Have the first to generalize channel shapes (in order to don't need to recode if new color_modes will be accepted)
    img = _load_single_image(src_dataset, img_list[0], color_mode, target_size)
    # Gets the output shape in order to assing a shape to images matrix
    outputShape = [len(img_list)]
    outputShape.extend(list(img.shape))
    # Declare the numpy where all images will be saved
    images = np.zeros(shape=tuple(outputShape), dtype=img.dtype)
    images[0] = img  # assign de first
    for i in range(1, len(img_list)):
        img = _load_single_image(src_dataset, img_list[i], color_mode, target_size)
        images[i] = image.img_to_array(img)

    if preprocessing_function is not None and prep_function is True:
        # dtype = images.dtype
        # also for problems with the keras backend
        images = images.astype(np.float32)
        images = preprocessing_function(images)  # np.asarray(images)) #Now are array right since the beginning
        # NEEDS TO BE TESTED IF REALLY CONTINUE WORKING FINE
    return images


def _load_single_image(src_dataset, img_name, color_mode, target_size, preprocessing_function=None,
                       prep_function=False):
    """Loads an image into PIL format.

    :param img_name: String, name of the image.

    :return: PIL image instance
    """
    grayscale = color_mode == 'grayscale'

    img = image.load_img(src_dataset + img_name,
                         grayscale=grayscale,
                         target_size=target_size)
    img = np.array(img)

    if preprocessing_function is not None and prep_function:
        img = preprocessing_function(img)
    return img
