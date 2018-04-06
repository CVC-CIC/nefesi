# nefesi (Neuron Feature and Selectivity Indexes)
Neuron activity visualization


## Introduction

In this package the terms filters and neurons can be used interchangably. While in computer vision, filters are consideres as the kernels that contain the convolutional weights, in our work we are formulating the behaviour of the filters as neuron concept.

Assuming the basic definition of neuron as an entity that spikes given an specific stimulus, when we work in CNN we find out that neurons are usually referred as pixels in an activation/feature maps.  are sharing the filter weights (filter), we provide a tool to analyze the behaviour of all the neurons linked to a given filter. For simplicity, from now on we will call this set of neurons linked to the same filter as Neuron. Consequently, in this package the terms filter and neuron can be used interchangably.  

## Package structure


## Neuron feature (NF)
This package provides the Neuron Feature to understand the internal representations that capture the intrinsic image features, which are a natural consequence of an automatic feature learning to achieve the goal task. Each of the stacked layers of the architecture operates on their inputs to produce a representation change. Taking into account that convolutional layers are the main responsible elements in detecting visual features encoded through their set of neurons, these representation changes are carried out in terms of the features encoded in each layer, likewise each neuron codifies certain features based on the previous layer feature space. Whilst the effects of the first convolutional layer can be easily understood, the understanding of the learned features becomes more difficult in deeper layers when several layers are stacked. This unawareness boosted the interest in understanding and analyzing learned features and several works have proposed different methodologies to address this understanding problem, going beyond proposing different CNN architectures or learning techniques.

_**Definition:**_ visualizes the features provoking a high activation of a specific neuron.
_Properties_: 
   * Image independent
   * Visualization of each neuron independently
   * Realistic appearance

_**Construction:**_ Weighted mean of the set of N-top-scoring images. Each of the set of N-top-scoring images is weighted by the activation provided by the specific neuron on this receptive field.

  * _Top-scoring images_: Set of receptive fields of the training set that gives the highest activations for each neuron with respect to the overall receptive fields analyzed on the training set.


## Selectivity indexes
Selectivity indexes allow us to describe neurons according to their inherent response to a specific property and to rank the whole neuron population proportionally to their related indexes.

_**Definition:**_ a property related to a neuron that measures how much the neuronal response (*i.e.*, the neuron activity) is affected by changes in the stimulus. In this sense, a high selectivity index characterizes a neuron to be highly dependent to a specific property, therefore, when this property is slightly changed, the neuron activity is considerably decreased.

### Selectivity indexes included
In this package we include four different selectivity indexes that allow us to describe each neuron according to color, symmetry, orienation and class properties.

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

* _Relevant neurons_: set of neurons that are highly activated in each specific position.

* _Application_: Visualize image representation through the set of NFs of the corresponding relevant neurons. It provides an illustration emphasizing the most relevant selected features for describing a specific image through the network. 

 

## Hierarchical decomposition
_**Goal**_: Describe a specific neuron of a deep layer through the set of neurons of a previous layer in order to provide insight into how shallower neurons are composed to provide a more complex feature in the neuron of the deeper layer

* _Application_: Visualize the feature composition of a neuron through the set of NFs of the neurons of shallower layers that are highly related to the studied neuron. It provides an illustration emphasizing the simpler features combined to detect more complex features in deeper layers.

