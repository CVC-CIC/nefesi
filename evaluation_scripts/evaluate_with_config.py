"""
This file contains a toy examples to have a first contact with Nefesi, and Keras.
This file has been created with tensorflow (and tensorflow-gpu) 1.8.0, keras 2.2.0, and python 3.6 (with anaconda3 interpreter)
"""

#from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

from evaluation_scripts.calculate_indexs import run_calculs

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle



class EvaluationWithConfig:
	def __init__(self, network_data, model_file, evaluate_index=False, verbose=True):
		self.network_data = network_data
		self.model_file = model_file
		self.evaluate_index = evaluate_index
		self.verbose = verbose

	def run_evaluation(self):
		self.network_data.model = load_model(self.model_file)
		self.network_data.dataset.preprocessing_function = None #Change it for use a preprocessing function
		self.network_data.eval_network(verbose=self.verbose)
		if self.evaluate_index:
			run_calculs(network_data=self.network_data,verbose=self.verbose)

if __name__ == '__main__':
	evaluation = pickle.load(open("./evaluation_config.cfg", "rb"))
	evaluation.run_evaluation()
