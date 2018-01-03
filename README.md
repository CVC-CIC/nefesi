# nefesi
Neuron activity visualization


## Introduction

## Package structure


## Neuron feature (NF)
_**Definition:**_ visualizes the features provoking a high activation of a specific neuron


## Selectivity indexes
_**Definition:**_ a property related to a neuron that measures how much the neuronal response (*i.e.*, the neuron activity) is affected by changes in the stimulus. In this sense, a high selectivity index characterizes a neuron to be highly dependent to a specific property, therefore, when this property is slightly changed, the neuron activity is considerably decreased.

### Selectivity indexes included
* **Color selectivity index**: measures the activity of a neuron to an input stimulus presenting certain color bias. 
  * _High value_: the neuron is sensitive to a color
  * _Low value_: the neuron is not sensitive to a color
  
* **Symmetry selectivity index**: measures the activity of a neuron to an input stimulus presenting a mirror symmetry property.
  * _High value_:  the neuron is sensitive to a mirror symmetry
  * _Low value_: the neuron is not sensitive to a mirror symmetry
  
* **Orientation selectivity index**: measures the activity of a neuron to an input stimulus presenting an orientation bias.
  * _High value_:  the neuron is sensitive to orientation
  * _Low value_: the neuron is not sensitive to orientation
  
* **Class selectivity index**: measures the activity of a neuron to an input stimulus related to a specific class.
  * _High value_: the neuron is sensitive to a feature that belongs to a class
  * _Low value_:  the neuron is sensitive to features found in several classes




## Image decomposition
_**Goal**_: Describe a given image through the set of features that are relevant in the CNN representation.

* _relevant neurons_: set of neurons that are highly activated in each specific position.

* _Application_: Visualize image representation through the set of NFs of the corresponding relevant neurons. It provides an illustration emphasizing the most relevant selected features for describing a specific image through the network. 

 

## Hierarchical decomposition
Editant
