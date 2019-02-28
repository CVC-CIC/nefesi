"""
This file contains a toy examples to have a first contact with Nefesi, and Keras.
This file has been created with tensorflow (and tensorflow-gpu) 1.8.0, keras 2.2.0, and python 3.6 (with anaconda3 interpreter)
"""

#from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from .calculate_indexes import run_calculs
import warnings
import os
import dill as pickle

import nefesi.util.GPUtil as gpu
gpu.assignGPU()
BATCH_SIZE = 100

class EvaluationWithConfig:
	def __init__(self, network_data, model_file, evaluate_index=False, verbose=True):
		self.network_data = network_data
		self.model_file = model_file
		self.evaluate_index = evaluate_index
		self.verbose = verbose

	def run_evaluation(self):
		self.network_data.model = load_model(self.model_file)
		self.network_data.save_changes = True
		# Change it for use a new preprocessing function
		self.set_preprocess_function()
		self.network_data.eval_network(verbose=self.verbose,batch_size=BATCH_SIZE)
		if self.evaluate_index:
			run_calculs(network_data=self.network_data,verbose=self.verbose)

	def set_preprocess_function(self):
		if self.network_data.model.name == 'vgg16':
			from keras.applications.vgg16 import preprocess_input
		elif self.network_data.model.name == 'resnet50':
			from keras.applications.resnet50 import preprocess_input
		elif self.network_data.model.name == 'vgg19':
			from keras.applications.vgg19 import preprocess_input
		elif self.network_data.model.name == 'xception':
			from keras.applications.xception import preprocess_input
		else:
			preprocess_input = None
			warnings.warn("Preprocess function is "+str(self.network_data.dataset.preprocessing_function)+". Please be sure "
							"that you don't need a specific preprocess_input function for this network")
		self.network_data.dataset.preprocessing_function = preprocess_input


def main():
	with open("../nefesi/evaluation_scripts/evaluation_config.cfg", "rb") as f:
		evaluation = pickle.load(f)
	evaluation.run_evaluation()
