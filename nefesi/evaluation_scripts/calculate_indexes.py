"""
This file contains a toy examples to have a first contact with Nefesi, and Keras.
This file has been created with tensorflow (and tensorflow-gpu) 1.8.0, keras 2.2.0, and python 3.6 (with anaconda3 interpreter)
"""

from ..network_data import NetworkData
import os
import pickle

import nefesi.util.GPUtil as gpu
gpu.assignGPU()

ALL_INDEX_NAMES = ['symmetry', 'orientation', 'color', 'class', 'object', 'part']

class CalculateIndexes:
	def __init__(self, network_data_file, model_file, sel_indexes = ALL_INDEX_NAMES, verbose=True, degrees_orientation_idx= None):
		self.network_data_file = network_data_file
		self.model_file = model_file
		self.verbose = verbose
		self.degrees_orientation_idx = degrees_orientation_idx
		self.sel_indexes = sel_indexes
	def run_calculs(self,network_data = None):
		if network_data is None:
			network_data = NetworkData.load_from_disk(self.network_data_file, model_file=self.model_file)
			run_calculs(network_data, degrees_orientation_idx=self.degrees_orientation_idx, sel_indexes=self.sel_indexes,
						verbose=self.verbose)

def run_calculs(network_data, degrees_orientation_idx=None, sel_indexes = ALL_INDEX_NAMES, verbose=True):
	network_data.save_changes=True
	network_data.indexs_accepted = sel_indexes
	network_data.get_selectivity_idx(sel_index=sel_indexes, layer_name='.*',
									 degrees_orientation_idx=degrees_orientation_idx, verbose=verbose)
	network_data.get_relevance_idx(layer_name='.*')
	network_data.similarity_idx(layer_name='.*')

def main():
	with open("../nefesi/evaluation_scripts/indexes_config.cfg", "rb") as f:
		indexes_eval = pickle.load(f)
	indexes_eval.run_calculs()