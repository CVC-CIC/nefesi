"""
This file is to define the deep framework used in the analysis.
Recommended version:
keras.__version__ = '2.2.4'
torch.__version__ = '1.6.0'
"""

Type_Framework = "Pytorch"   # "Keras" or "Pytorch"

if Type_Framework == "Keras":
    from .keras_functions import DeepModel as ModelType
    from .keras_functions import DataBatchGenerator, get_preprocess_function
    from .keras_functions import _load_multiple_images, _load_single_image
    model_file_extension = 'h5'
    channel_type = "channel_last"
elif Type_Framework == "Pytorch":
    from .pytorch_functions import DeepModel as ModelType
    from .pytorch_functions import DataBatchGenerator, get_preprocess_function
    from .pytorch_functions import _load_multiple_images, _load_single_image
    model_file_extension = 'pkl'
    channel_type = "channel_first"
else:
    raise Exception("Type_Framework error. ")


def deep_model(model_name):
    return ModelType(model_name)


def data_batch_generator(preprocessing_function, src_dataset, target_size,
                         batch_size, color_mode):
    return DataBatchGenerator(preprocessing_function, src_dataset, target_size,
                              batch_size, color_mode)


def preprocess_function(model_name):
    return get_preprocess_function(model_name)


def load_multiple_images(src_dataset, img_list, color_mode, target_size, preprocessing_function=None,
                         prep_function=True):
    """
    This will be used in the calculation of deep framework. The output should be based on the type of
    deep framework.
    """
    return _load_multiple_images(src_dataset, img_list, color_mode, target_size,
                                 preprocessing_function=preprocessing_function,
                                 prep_function=prep_function)


def load_single_image(src_dataset, img_name, color_mode, target_size, preprocessing_function=None,
                      prep_function=False):
    """
    This will be used in the calculation of features. The output should be in numpy.
    """
    return _load_single_image(src_dataset, img_name, color_mode, target_size,
                              preprocessing_function=preprocessing_function,
                              prep_function=prep_function)
