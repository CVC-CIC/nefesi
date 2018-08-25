"""
This file contains a toy examples to have a first contact with Nefesi, and Keras.
This file has been created with tensorflow (and tensorflow-gpu) 1.8.0, keras 2.2.0, and python 3.6 (with anaconda3 interpreter)
"""

from ..network_data import NetworkData
from ..layer_data import ALL_INDEX_NAMES
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle


class CalculateIndexs:
	def __init__(self, network_data_file, model_file, verbose=True, degrees_orientation_idx= None):
		self.network_data_file = network_data_file
		self.model_file = model_file
		self.verbose = verbose
		self.degrees_orientation_idx = degrees_orientation_idx
	def run_calculs(self,network_data = None):
		if network_data is None:
			network_data = NetworkData.load_from_disk(self.network_data_file, model_file=self.model_file)
			run_calculs(network_data,degrees_orientation_idx=self.degrees_orientation_idx,
									 verbose=self.verbose)

def run_calculs(network_data, degrees_orientation_idx=15, verbose=True):
	index_to_evaluate = ALL_INDEX_NAMES
	if network_data.addmits_concept_selectivity():
		index_to_evaluate+=['concept']
	network_data.get_selectivity_idx(sel_index=index_to_evaluate,layer_name='.*',
									 degrees_orientation_idx=degrees_orientation_idx,verbose=verbose)
	network_data.similarity_idx(layer_name='.*')

def main():
	with open("../nefesi/evaluation_scripts/indexs_config.cfg", "rb") as f:
		indexs_eval = pickle.load(f)
	indexs_eval.run_calculs()