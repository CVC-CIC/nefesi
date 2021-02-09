Type_Framework = "Keras"   # or "Pytorch"

if Type_Framework == "Keras":
    from .keras_functions import LoadModel as ModelType, DataBatchGenerator, intermediate_funcs, get_preprocess_function
elif Type_Framework == "Pytorch":
    from .pytorch_functions import LoadModel as ModelType, DataBatchGenerator, intermediate_funcs
else:
    raise Exception("Type_Framework error. ")

def load_model(model_name):
    return ModelType(model_name)


def data_batch_generator(preprocessing_function, src_dataset, target_size,
                         batch_size, color_mode):
    return DataBatchGenerator(preprocessing_function, src_dataset, target_size,
                              batch_size, color_mode)


def create_intermediate_funcs(inp, outputs):
    return intermediate_funcs(inp, outputs)


def preprocess_function(model_name):
    return get_preprocess_function(model_name)


