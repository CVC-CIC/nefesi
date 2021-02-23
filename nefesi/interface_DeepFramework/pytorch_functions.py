"""
The vgg model is from: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
"""
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os.path
import numpy as np
import types

# pytorch model has no input shape. Attention: pytorch is channel first, keras is channel last.
input_shape = (None, 224, 224, 3)

class DeepModel():
    """

    """
    def __init__(self, model_name):
        self.pytorchmodel = torch.load(model_name)
        self.gpu_ids = [0]
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        self.pytorchmodel.to(self.device)
        self.pytorchmodel.eval()
        self.name = os.path.basename(model_name).split(".")[0]
        self.flag_set_layer = False   # Whether attribute "layers" is ready

    def set_layers(self):
        """
        The pytorch has no layers attributes like keras. So they must be set before using.
        :return:
        """
        self.all_layers = []
        layers_setting_hook = []
        for name, layer in self.pytorchmodel.named_modules():
            if not isinstance(layer, torch.nn.Sequential) and name != '':
                # add new attributes
                layer.name = '{}({})'.format(name, layer._get_name())
                layer.get_config = types.MethodType(get_config, layer)
                self.all_layers.append(layer)
                layers_setting_hook.append(LayerAttributes(layer))
        x = torch.rand(100, self.input_shape[3], self.input_shape[1], self.input_shape[2]).to(self.device)
        y = self.pytorchmodel(x)
        for setting_hook in layers_setting_hook:
            setting_hook.remove()

    @property
    def layers(self):
        """
        This function gets the list of layers of the model.
        Some new attributes are added for the compatibility of the keras code.
        i.e.: layer.name,
        :return:
        """
        if not self.flag_set_layer:
            self.set_layers()
            self.flag_set_layer = True

        return self.all_layers

    @property
    def input_shape(self):
        return input_shape

    def get_layer(self, layer_name, get_index=False):
        for index, layer in enumerate(self.layers):
            if layer.name == layer_name:
                if get_index:
                    return layer, index
                else:
                    return layer

        raise ValueError('No such layer: ' + layer_name)

    def neurons_of_layer(self, layer_name):
        # Pytorch has the separate RELU layer, we need to return the channels of the last layer.
        last_layer = None
        current_layer = None
        for layer in self.layers:
            if layer.name == layer_name:
                current_layer = layer
                break
            last_layer = layer
        if hasattr(current_layer, 'out_channels'):
            return current_layer.out_channels
        else:
            return last_layer.out_channels

    def save(self, model_name):
        pass

    def calculate_activations(self, layers_name, model_inputs):
        model_inputs = model_inputs.to(self.device)
        outputs = [LayerActivations(self.get_layer(layer)) for layer in layers_name]
        self.pytorchmodel.forward(model_inputs)
        layer_outputs = [output.features for output in outputs]
        # for index, name in enumerate(layers_name):
        #     layer = self.get_layer(name)
        #     if not hasattr(layer, "output_shape"):
        #         layer.output_shape = layer_outputs[index].shape
        return layer_outputs


class LayerActivations:
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        feature_np = output.cpu().detach().numpy()
        self.features = np.transpose(feature_np, (0, 2, 3, 1))

    def remove(self):
        self.hook.remove()


class LayerAttributes:
    """
    Hook for setting the attributes of layers.
    """
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # add some attributes to layers
        if isinstance(input, tuple):
            layer_input_shape = input[0].size()
        else:
            layer_input_shape = input.size()
        layer_output_shape = output.size()
        module.input_shape = self.transform_shape_format(layer_input_shape)
        module.output_shape = self.transform_shape_format(layer_output_shape)

    def transform_shape_format(self, shape_in_tensor):
        if shape_in_tensor.__len__() == 4:
            shape_in_tuple = (None, shape_in_tensor[2], shape_in_tensor[3], shape_in_tensor[1])
        else:
            shape_in_list = list(shape_in_tensor)
            shape_in_list[0] = None
            shape_in_tuple = tuple(shape_in_list)
        return shape_in_tuple

    def remove(self):
        self.hook.remove()


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class DataBatchGenerator():
    def __init__(self, preprocessing_function, src_dataset, target_size,
                         batch_size, color_mode):
        if color_mode == "grayscale":
            preprocessing_function.transforms.append(
                    transforms.Grayscale(num_output_channels=1))
        dataset = ImageFolderWithPaths(src_dataset, transform=preprocessing_function)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        pass

    # the attributes that are called
    @property
    def iterator(self):
        # the batch iterator
        return self.dataloader

    @property
    def samples(self):
        # number of images
        return self.dataloader.dataset.__len__()

    @property
    def batch_index(self):
        # index of batch
        return 0

    @property
    def batch_size(self):
        # batch size
        return self.dataloader.batch_size

    @property
    def filenames(self):
        return self.dataloader.dataset.samples

    @property
    def index_array(self):
        pass


def get_preprocess_function(model_name):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess


def get_config(self):
    config = {}
    attr_list = ["kernel_size", "stride", "padding"]
    target_attr_list = ["kernel_size", "strides", "padding"]
    for index, attr in enumerate(attr_list):
        if hasattr(self, attr):
            config[target_attr_list[index]] = getattr(self, attr)
    return config


if __name__ == "__main__":
    model = DeepModel("/home/yixiong/PycharmProjects/nefesi/vgg16_pytorch.pkl")
    pass

