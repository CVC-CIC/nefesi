"""
This file contains a toy examples to have a first contact with Nefesi, and Keras.
This file has been created with tensorflow (and tensorflow-gpu) 1.8.0, keras 2.2.0, and python 3.6 (with anaconda3 interpreter)
"""

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model #For load local keras models (h5 files)

from nefesi.network_data import NetworkData
from nefesi.util.image import ImageDataset
import numpy as np
import time
import pickle

import nefesi.util.GPUtil as gpu
gpu.assignGPU()

def main():
	start = time.time()
	#example1SaveModel(VGG16) #Charge a standard model and save it locally
	#example2ChargeModel('../Data/VGG16.h5') #Charge a model locally
	#example3NefesiInstance('../Data/VGG16.h5')
	#example4FullFillNefesiInstance('/home/eric/Nefesi/Data/VGG16.h5', '/home/eric/Nefesi/Datasets/TinyImagenet/trainSubset/', '/home/ramon/work/nefesi/Data2/')
	example5NetworkEvaluation(      '/home/eric/Nefesi/Data/VGG16.h5', '/home/eric/Nefesi/Datasets/TinyImagenet/trainSubset/', '/home/eric/Nefesi/Data/Vgg16Efficiency/')
	#example6LoadingResults()
	#example7AnalyzingResults()
	print("TIME ELAPSED: "+str(time.time()-start))

"""
Analyze the results of the evaluation
"""
def example7AnalyzingResults():
	"""
	The truth kernel part of Nefesi, the analysis of the evaluation maded in example5. Nefesi analysis is based on index,
	and allows user to inspect the neurons that maximize this index, minimize... Index that allows are: "color", "orientation",
	 "symmetry", "class" or "population code". Let's to see it in a example
	"""
	nefesiModel = example6LoadingResults()
	start = time.time()
	#colorSelectivity(nefesiModel)
	symmetrySelectivity(nefesiModel)
	#classSelectivity(nefesiModel)
	#orientationSelectivity(nefesiModel)
	#populationCode(nefesiModel)
	end = time.time()
	print("TIME ELAPSED: ")
	print(end - start)

def populationCode(nefesiModel):
	layersToEvaluate = 'block1_conv1'
	# degrees_orientation_idx 180 will be only one rotation
	selIdx = nefesiModel.get_selectivity_idx(sel_index="population code", layer_name=layersToEvaluate)
	print("a")

def orientationSelectivity(nefesiModel):
	"""
	Orientation selectivity is an index that specifies, how much the neuron is resistent to image rotations.
	(experimentally we determined that is so usefull to detect, neurons that are selectives to circular things).
	In order to evaluate it, this neuron takes his N-Top images (the N images that more activation produced to it)
	and rotate each image degrees_orientation_idx degrees (15 by default) n times, until complete the 360 degrees.
	This index will be in range [0.0,1.0] and an 1.0 idx will indicate that the activitation was the same for no-rotated
	image, and rotated image. The return of orientation selectivity index will be a numpy for each neuron with m positions.
	Where m is ceil(360/degrees_orientation_idx). (A position for each rotation tried (with his the index obtained) and the
	last position won't be the 360 degrees rotation, will be the mean of all rotations tried (on this neuron)). For example
	for degrees orientation 90 and block1_conv1 layer will be:
	selIdx['orientation']--->[list of orientation_idx values] --> idx for 90 degrees
															  --> idx for 180 degrees
															  --> idx for 270 degrees
															  --> mean of last 3 values
	"""
	#Let's evaluate first layer (64 neurons)
	layersToEvaluate = 'block5_conv1'
	print("Let's Evaluate the orientation selectivity index of layers: "+
		  str(nefesiModel.get_layers_analyzed_that_match_regEx(layersToEvaluate))+ " (This operation will take minutes).")
	#degrees_orientation_idx 180 will be only one rotation
	selIdx = nefesiModel.get_selectivity_idx(sel_index="orientation", layer_name=layersToEvaluate, degrees_orientation_idx=180)
	print("Orientation selectivity index calculated for each neuron of layer 'block1_conv1'\n"
		  "Max value of color selectivity encountered in first layer analyzed"+ str(np.max(selIdx['orientation'][0][:,-1]))+
		  " in neuron: " +str(np.argmax(selIdx['orientation'][0][:,-1]))+". \n"
		  "Neurons with more than 60% of mean orientation selectivity: "+str(len(np.where(selIdx['orientation'][0][:,-1]>0.6)[0]))+
		  ".\n Mean of Color Selectivity in first layer: "+str(np.mean(selIdx['orientation'][0][:,-1]))+".")

def classSelectivity(nefesiModel):
	"""
		Class  selectivity is an index that specifies what classt the neuron is more selective. In order to evaluate it,
		 this neuron takes his N-Top images (the N images that more activation produced to it) and evaluate what is the
		 classe that more accumulated activation produces. This index will be in range [0.0,1.0] and an 1.0 idx will indicate
		 that all the images that produces an activation in this neuron are of the specified class. The return of class
		selectivity index will be a tuple for each neuron that contains: (humanLabelNameOfClass(String),
		SelectivityIndexForThisClass(float)). This index only take a seconds to calculate.
	"""
	#Let's to evaluate layers 1 and 3
	layersToEvaluate = 'block(5)_conv1'
	"""
	This selectivity index accepts a dictionary, that translate from real label names (often not human readable as n03794056)
	to another label names (that can be human readable as 'mousetrap'). This dictionary can be specified as a none if this
	parameter (labels) is not specified, real label names will be used
	"""
	#Charge a pyhton dict that translate from imageNet labels (like n03794056) to human readable labels (like 'mousetrap')
	with open("../nefesi/external/labels_imagenet.obj", "rb") as f:
		labelsDict = pickle.load(f)
	print("Let's Evaluate the class selectivity index of layers:"+
		  str(nefesiModel.get_layers_analyzed_that_match_regEx(layersToEvaluate)) +" (This operation will take seconds)")
	selIdx = nefesiModel.get_selectivity_idx(sel_index="class", layer_name=layersToEvaluate, labels=labelsDict)
	"""
	The selIdx that is returned for index 'class' has a little different property. It's an heterogeneus array, that contains
	a for each position a tuple ('label','value'). Cause his content is a tuple you can access in the next forms:
	selIdx = {'class': [list of tuples for layer 1] --> ('goldfish',0.25)
													--> ('mousetrap',0.41)
													--> ('goldfish', 0.009)
					   [list of tuples for layer 2] --> ('chair',0.72)
													--> ('cherry',0.08)
													--> ('goldfish', 0.95)
	print(selIdx['class'][0][1]) ----> ('mousetrap',0.41)
	print(selIdx['class'][0][1][1]) ----> 0.41
	print(selIdx['class'][0]['label']) ----> ['goldfish','mousetrap','goldfish]
	print(selIdx['class'][0]['value']) ----> [0.25,0.41,0.009]

	NOTE: 'label', and 'value' are the important names to remember. (Working like pandas)
	"""
	print("Class selectivity index contains an structured tuples ('label','value') you can acces to all only labels or all"
		  "only value attributes. Keys: "+str(selIdx['class'][0].dtype.names))

	for layer_idx, layer_name in enumerate(nefesiModel.get_layers_analyzed_that_match_regEx(layersToEvaluate)):
		print("---------------- LAYER "+layer_name.upper()+" ----------------\n"
			  "Class with higher selectivity index and his index: "+
			  str(selIdx['class'][layer_idx][np.argmax(selIdx['class'][layer_idx]['value'])])+
			  "\n Mean idx of class selecitivity: "+ str(np.mean(selIdx['class'][layer_idx]['value'])))

"""
Looks each neuron symmetry selectivity on the network
"""
def symmetrySelectivity(nefesiModel):
	"""
	Simmetry selectivity is an index that specifies how much a specific neuron is selective to simmerty. In order to evaluate it,
	 this neuron takes his N-Top images (the N images that more activation produced to it) and evaluate the diference between
	 the activation of the neuron to the image_i and the activation with the same image mirrored and repeat the result rotating
	 the image 45º, 90º and 135º. The result will be a list (numpy) of 5 floats per each neuron in range [0.0,1.0] (index for
	 0º rotated image, 45º rotated image, 90º rotated image, 135º rotated image, the mean) that specifies the area between
	 two graphics (activations with images vs activations with mirrored images)
	"""
	layersToEvaluate = ["block5_conv1"]
	print("Let's Evaluate the symmetry selectivity index of " + str(len(layersToEvaluate)) + " layers: " + str(layersToEvaluate)+
		  " (This operation can take minutes)")
	"""
	selIdx will be a dictionary that contains per each key in sel_index a list (numpy) that contains a entrance per each layer
	in layer_name, that contains list of float per each neuron that is his symmetry selectivity index(list of len 5).
	For example in the case sel_index="simmetry", layer_name=["block1_conv1","block3_conv1"] selIdx will be:

	selIdx-->selIdx['simmetry']-->[numpy with the simmetry_idx of each neuron in layer "block1_conv1"]-->[0º,45º,90º,135º,mean] of neuron 1
										 															  -->[0º,45º,90º,135º,mean] of neuron 2
																									  ...
																									  -->[0º,45º,90º,135º,mean] of neuron n
							   -->[numpy with the simmetry_idx of each neuron in layer "block3_conv1"]-->[0º,45º,90º,135º,mean] of neuron 1
								   																      -->[0º,45º,90º,135º,mean] of neuron 2
																								      ...
																								      -->[0º,45º,90º,135º,mean] of neuron n
	"""
	# Calculate the color index of layer block1_conv1 (this process can take more than ten minutes)
	selIdx = nefesiModel.get_selectivity_idx(sel_index="symmetry", layer_name=layersToEvaluate)
	print("Symmetry selectivity index calculated for each neuron of layer 'block1_conv1'\n"
		  "Max value of mean simmetry selectivity encountered in this layer: {} in neuron: {}  (symmetryIndex in "
		  "[0º,45º,90º,135º,mean]: {}).\n"
		  "Neurons with more than 60% of symmetry selectivity: {}".format(np.max(selIdx['symmetry'][0][:,4]),
																	   np.argmax(selIdx['symmetry'][0][:,4]),
		  											selIdx['symmetry'][0][np.argmax(selIdx['symmetry'][0][:, 4]), :],
		  											len(np.where(selIdx['symmetry'][0][:, 4] > 0.6)[0])))


"""
Looks each neuron color selectivity on the network
"""
def colorSelectivity(nefesiModel):
	"""
	Color selectivity is an index that specifies how much a specific neuron is selective to color. In order to evaluate it,
	this neuron takes his N-Top images (the N images that more activation produced to it) and evaluate the diference between
	the activation of the neuron to the colorImage_i and the activation with the same image in grayScale. The result will
	be a float per each neuron in range [0.0,1.0], that specifies the area between two graphics (activations with color vs
	activations with grayscale)

	More info at: https://arxiv.org/abs/1702.00382v1
	"""
	layersToEvaluate = ["block5_conv1"]
	print("Let's Evaluate the color selectivity index of "+str(len(layersToEvaluate))+" layers: "+str(layersToEvaluate))
	"""
	selIdx will be a dictionary that contains per each key in sel_index a list that contains a entrance per each layer
	in layer_name, that contains a float per each neuron that is his selectivity index. For example in the case
	sel_index="color", layer_name=["block1_conv1","block3_conv1"] selIdx will be:

	selIdx-->selIdx['color']-->[list with the color_idx of each neuron in layer "block1_conv1"]
							-->[list with the color_idx of each neuron in layer "block3_conv1"]
	"""
	#Calculate the color index of layer block1_conv1 (this process can take more than a minute)
	selIdx = nefesiModel.get_selectivity_idx(sel_index="color", layer_name=layersToEvaluate)
	print("Color selectivity index calculated for each neuron of layer 'block1_conv1'\n"
		  "Max value of color selectivity encountered in this layer: "+str(np.max(selIdx['color'][0]))+" in neuron: "
		  ""+str(np.argmax(selIdx['color'][0]))+". \n"
		  "Neurons with more than 60% of color selectivity: "+str(len(np.where(selIdx['color'][0]>0.6)[0]))+ ". Mean of "
		  "Color Selectivity in layer 'block1_conv1': "+str(np.mean(selIdx['color'][0]))+".")


"""
Charges one of the last files saved, in order to start analyzing it
"""
def example6LoadingResults():
	"""
	Charges a model with analysis data. In order to have it a sthatic function is used (NetworkData.load_from_disk(...)
	and it takes the files generated in last example. In order to charge results of more than one layer, is needed to charge
	the .obj with the name of the last layer that you wants to analyze (For example: if you have block1_conv1.obj, block2_conv1.obj,
	block3_conv2.obj and vgg16.obj, and you wants to analyze block1_conv1, block2_conv1 layers you need to set file_name=block2_conv1.obj,
	or file_name=vgg16.obj (vgg16.obj==block2_conv1.obj (in this case)) if you want to charge all layer results)

	"""
	nefesiModel = NetworkData.load_from_disk(file_name="../../Data/Vgg16Efficiency/vgg16.obj", model_file="../../Data/VGG16.h5")
	return nefesiModel

"""
Makes the network evaluation. This is a kernel part of Nefesi package, and will generate the files that after will be
used to analyze the network.
"""
def example5NetworkEvaluation(model_file_name, dataset_folder, save_folder):
	"""
	Nefesi will eval the network, and save results in a directory nefesiModel.save_path (setted at last example as
	"../Data/" . In this directory will appears a list of '.obj' files. One per layer evaluated and all with the name
	of corresponding layer. Each file will have the information of all previous evaluated layers (for example: if you
	eval block1_conv1 and block2_conv1, in the directory will appears block1_conv1.obj, and block2_conv1.obj. The last
	one with the content of block1_conv1.obj and his own content (block2_conv2). The reason of it is because when you
	will charge a .obj in order to analyze results, you will only charge one file to analyze all layers).
	"""
	nefesiModel = example4FullFillNefesiInstance(model_file_name, dataset_folder, save_folder)
	#This function will have the evaluation. All parameters setted in example4 are also parameters of this function. If
	#user don't set it in .evalNetwork(...) takes by default the values of nefesiModel object
	print("Let's start to evaluate network. This process is based on dataset and the time that spent is depenent of the "
		  "size of it. Is recommended to have a visible GPU for this process "
		  "(os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"1\" \n"
		  "GO TO EVALUATE! :)")
	start = time.time()
	nefesiModel.eval_network(verbose=True, batch_size=250)
	print("TIME ELAPSED: "+str(time.time()-start))

	print("Evaluation finished, nefesiObject have now the info of analysis and results can be reloaded with files (.obj)"
		  "in dir --> "+nefesiModel.save_path)

"""
Full fills the NetworkData object. By the nature of the analisi, this will be based on an Image Dataset (Charged Locally),
the specification of layers to be analyzed (beacause not all layers will have sense to analyze (like input layer), and
the path to save the results. In this example is explained how to set this parameters correctly.
"""
def example4FullFillNefesiInstance(model_file_name, dataset_folder, save_folder):
	"""
	Nefesi analisys is based on neuron activations produced by a dataset of example images. This analisis produces a results
	based in the input images (Mean of activations, N-Top Images...) for this reason the NetworkData object needs to be
	associated with a dataset, that is from where Nefesi take this images
	"""
	#Take the instance (of previous example)
	nefesiModel = example3NefesiInstance(model_file_name)
	#Set the dataset that will be use in analisis
	nefesiModel.dataset = chargeNefesiImageDataset(dataset_folder)
	print("Dataset saved correctly and assigned to Nefesi object (NetworkData) correctly")
	#save_path atttribute save the path where results will be saved. This attribute (same as dataset) is optional, because
	#can be initialized in function nefesiModel.eval_network(...) that will see in next example
	nefesiModel.save_path = save_folder
	print("Path to save results saved correctly --> "+nefesiModel.save_path)
	layer_names = [l.name for l in nefesiModel.model.layers]
	print (layer_names)
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
	#Select to analyze first conv of block 1, 3 and 5 (init, middle & end)
	nefesiModel.layers_data = "block(3|5)_conv(1)"
	#nefesiModel.layers_data = "fc2"
	print("Layers "+str(nefesiModel.get_layer_names_to_analyze())+" selected to analyze\n"
															  "NetworkData object is full configured now")
	return nefesiModel

"""
Instantiate a NefesiImageDataset instance.
"""
def chargeNefesiImageDataset(dataset_folder):
	"""
	Nefesi uses a ImageDataset to evaluate the features of the network. This Dataset is specified as an object, and have
	 the preproces of each image (resize, crop... or specific function), in order to give to an heterogeneus dataset a list
	 of well known caractheristics (As size, color space...).
	Imports needed: from nefesi.util.image import ImageDataset, preprocess_input
	:return: ImageDataset instance, that represents de Dataset that will be used to evaluate the network
	"""
	#the path from where images will be taken must to have the next architecture:
	"""
	ClassAFolder -> Img1, Img2, Img3...
	ClassBFolder -> Img1, Img2, Img3...
	"""
	path = '../Datasets/TinyImagenet/trainSubset/'

	#target_size is the size of the images will be resized and cropped before to put in the net, in this case the best
	#option is to set as (224 (height), 224 (width)) cause this is the input size of VGG16.
	targetSize = (224,224)
	#the color mode selected ('rgb' or 'grayscale') is the color mode to READ the images, not the internal treatment colorMode.
	#In the most cases it will be 'rgb', cause is the common input of the nets and have more info than 'grayscale'.
	colorMode = 'rgb'
	#Calls to constructor. Preprocess_input is the function that applies the preprocess with the VGG16 was trained.
	dataset = ImageDataset(src_dataset=dataset_folder, target_size=targetSize, preprocessing_function=preprocess_input, color_mode=colorMode)

	return dataset

"""
Charges a model locally and instance a NetworkData object (The main class of Nefesi package)
"""
def example3NefesiInstance(file_name):
	"""
	Nefesi is an useful library for analize a CNN. The main Nefesi class is NetworkData that receives as a constructor
	 parameter only the model (keras.models.Model instance).
	Imports needed: from nefesi.network_data import NetworkData
	"""
	model = example2ChargeModel(file_name) #Charges the model from a local file
	nefesiModel = NetworkData(model=model) #Instantiate the NetworkData object
	print("Nefesi object (NetworkData) instantiated")
	return nefesiModel
"""
Charge a model (keras.models.Model instace) from a local .h5 file
"""
def example2ChargeModel(file_name):
	"""
		If you have a model (saved in a .h5 file) Keras allows you to recharge it with a simple load_model('filepath')
		 one instruction. It is usefull to combine with example1, to open, modify, save and charge your own models.
		Imports needed: from keras.models import load_model
	"""
	print(f"Loading {file_name} model")

	model = load_model(file_name)

	print("Model loaded. \n Many times, when model is loaded, 'UserWarning: No training configuration found in save file' can be raised. "
		  "This is because the model saved was not compiled (model.compile(...)). This warning is not rellevant if you "
		  "don't want to train the model further.")
	return model

"""
Charge a standard model (keras.models.Model instance) from Keras library and save it locally
"""
def example1SaveModel(model_function, path_name):
	"""
	Keras have some famous models in the library that can be charged. NOTE: This models will be downloaded from
	his GitHub source when constructor is called. That call needs an Internet connection. Another way to charge
	pretrained model is locally with .h5 file.
	Imports needed: from keras.applications.vgg16 import VGG16
	"""
	# Charge VGG16 model (downloads from github Source -->
	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5)
	print(f"Charging model {model_function.__name__}")

	model = model_function()


	# Save the model locally on path+name.h5 file.
	import os
	file_name = os.path.join(path_name, model_function.__name__)
	if not file_name.endswith('.h5'):
		file_name = file_name+'.h5'
	print(f"Model charged, saving model at {file_name}")
	"""
	Save model in Keras is so easy one instruction. A model object type has the method "save('fileName.h5')" that
	saves all the model in a local file
	"""
	# Save the model (model) locally
	model.save(file_name)

main()
