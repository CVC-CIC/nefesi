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
    model_file_extension = 'h5'
elif Type_Framework == "Pytorch":
    from .pytorch_functions import DeepModel as ModelType
    from .pytorch_functions import DataBatchGenerator, get_preprocess_function
    model_file_extension = 'pkl'
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


