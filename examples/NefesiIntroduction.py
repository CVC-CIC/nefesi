"""
This file contains a toy examples to have a first contact with Nefesi, and Keras.
This file has been created with tensorflow (and tensorflow-gpu) 1.8.0, keras 2.2.0, and python 3.6 (with anaconda3 interpreter)
"""

from keras.applications.vgg16 import VGG16
from keras.models import load_model #For load local keras models (h5 files)
from nefesi.network_data import NetworkData
from nefesi.util.image import ImageDataset
#from nefesi.util.plotting import plot_nf_search

import os
import pickle

def main():
	#example1SaveModel() #Charge a standard model and save it locally
	#example2ChargeModel() #Charge a model locally
	#example3NefesiInstance()
	example4NetworkEvaluation()

"""
Evaluate a Network. By the nature of the analisi, this will be based on an Image Dataset (Charged Locally).
"""
def example4NetworkEvaluation():
	"""
	Nefesi analisys is based on neuron activations produced by a dataset of example images. This analisis produces a results
	based in the input images (Mean of activations, N-Top Images...) for this reason the NetworkData object needs to be
	associated with a dataset, that is from where Nefesi take this images
	"""
	#Take the instance (of previous example)
	nefesiModel = example3NefesiInstance()
	#Set the dataset that will be use in analisis
	nefesiModel.dataset = chargeNefesiImageDataset()
	print("Dataset saved correctly and assigned to Nefesi object (NetworkData) correctly")
	#save_path atttribute save the path where results will be saved. This attribute (same as dataset) is optional, because
	#can be initialized in function nefesiModel.eval_network(...) that will see in next example
	nefesiModel.save_path = "../Data"
	print("Path to save results saved correctly --> "+nefesiModel.save_path)
	"""
	Nefesi analysis is selected by layers. The param layer_data indicates the layers that will be analyzed.
	For example in VGG16 this are the list of layers that contains:
	['input_1', 'block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', 'block2_pool', 'block3_conv1',
	'block3_conv2', 'block3_conv3', 'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
	'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool', 'flatten', 'fc1', 'fc2', 'predictions'].
	For select the layers to analyze nefesi offers 2 ways: By regular expresion or by Name.
	An example of regular expression can be... ".*conv*" to select all convolutional layers (by default the value is ".*"
	(all layers).
	An example of name list can be... ['block1_conv1', 'block2_pool', 'block5_conv3'] for analyze only thats 3 layers
	"""
	#Select to analyze all convolutional layers
	nefesiModel.layers_data = ".*conv*"
	print("Layers "+str(nefesiModel.getLayerNamesToAnalyze())+" selected to analyze\n"
															  "NetworkData object is full configured now")
	return nefesiModel

"""
Instantiate a NefesiImageDataset instance.
"""
def chargeNefesiImageDataset():
	"""
	Nefesi uses a ImageDataset to evaluate the features of the network. This Dataset is specified as an object, and have
	 the preproces of each image (resize, crop... or specific function), in order to give to an heterogeneus dataset a list
	 of well known caractheristics (As size, color space...).
	Imports needed: from nefesi.util.image import ImageDataset
	:return: ImageDataset instance, that represents de Dataset that will be used to evaluate the network
	"""
	#the path from where images will be taken must to have the next architecture:
	"""
	ClassAFolder -> Img1, Img2, Img3...
	ClassBFolder -> Img1, Img2, Img3...
	"""
	path = '../Datasets/TinyImagenet/train/'
	#target_size is the size of the images will be resized and cropped before to put in the net, in this case the best
	#option is to set as (224 (height), 224 (width)) cause this is the input size of VGG16.
	targetSize = (224,224)
	#the color mode selected ('rgb' or 'grayscale') is the color mode to READ the images, not the internal treatment colorMode.
	#In the most cases it will be 'rgb', cause is the common input of the nets and have more info than 'grayscale'.
	colorMode = 'rgb'
	#Calls to constructor
	dataset = ImageDataset(src_dataset=path, target_size=targetSize,preprocessing_function=None, color_mode=colorMode)

	return dataset

"""
Charges a model locally and instance a NetworkData object (The main class of Nefesi package)
"""
def example3NefesiInstance():
	"""
	Nefesi is an useful library for analize a CNN. The main Nefesi class is NetworkData that receives as a constructor
	 parameter only the model (keras.models.Model instance).
	Imports needed: from nefesi.network_data import NetworkData
	"""
	model = example2ChargeModel() #Charges the model from a local file
	nefesiModel = NetworkData(model=model) #Instantiate the NetworkData object

	print("Nefesi object (NetworkData) instantiated")
	return nefesiModel
"""
Charge a model (keras.models.Model instace) from a local .h5 file
"""
def example2ChargeModel():
	"""
		If you have a model (saved in a .h5 file) Keras allows you to recharge it with a simple load_model('filepath')
		 one instruction. It is usefull to combine with example1, to open, modify, save and charge your own models.
		Imports needed: from keras.models import load_model
	"""
	print("Loading VGG16.h5 model")

	model = load_model('../Data/VGG16.h5')

	print("Model loaded. \n Many times, when model is loaded, 'UserWarning: No training configuration found in save file' can be raised. "
		  "This is because the model saved was not compiled (model.compile(...)). This warning is not rellevant if you "
		  "don't want to train the model further.")
	return model

"""
Charge a standard model (keras.models.Model instance) from Keras library and save it locally
"""
def example1SaveModel():
	"""
	Keras have some famous models in the library that can be charged. NOTE: This models will be downloaded from
	his GitHub source when constructor is called. That call needs an Internet connection. Another way to charge
	pretrained model is locally with .h5 file.
	Imports needed: from keras.applications.vgg16 import VGG16
	"""
	# Charge VGG16 model (downloads from github Source -->
	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5)
	print("Charging VGG16 model")

	model = VGG16()

	print("Model charge, saving model at ./VGG16.h5")
	# Save the model locally on path+name.h5 file.
	saveModel(model=model, path='../Data', name='VGG16')  # Save it locally

"""
Save model in Keras is so easy one instruction. A model object type has the method "save('fileName.h5')" that
saves all the model in a local file
"""
def saveModel(model,path='', name='myModel'):
	#The file format of keras models is '.h5'
	if not name.endswith('.h5'):
		name = name+'.h5'
	# Save the model (model) locally
	model.save(path+name)

if __name__ == '__main__':
    # print os.path.dirname(os.path.abspath(__file__))
    # print os.getcwd()

    main()
